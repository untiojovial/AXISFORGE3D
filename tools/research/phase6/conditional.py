"""Stage D: a learned robustness model P(H|F) and the ordering P(F) * P(H|F).

    python tools/research/phase6/conditional.py --catalogues cat80/ --store store.json \
        --dev research/phase6/dev_parts.json --priors priors.npz --sweep research/phase6/results/sweep.json.gz \
        --fusion research/phase6/results/fusion_benchmark.json \
        --out robustness_prediction.json --manifest-out dynamic_training_manifest.json

Run because the protocol's gate opened: the centring term separates robust
from non-robust viable grasps with AUC 0.57 < 0.70. Same outer folds as the
fusion benchmark (seed 20261130): for each fold, a gradient-boosting classifier
(the Phase V settings, seed 0) is trained on the viable candidates of the
training parts with the label "robust in the six standard conditions" and the
same 19 features; the held-out parts are ordered by p/max * P(H|F). Nothing
is tuned. Contrasts against M0, M1 and M4 (the outer-fold results of the
fusion benchmark), part as the unit.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase6 import fusion, mechanism as mech  # noqa: E402

spec = importlib.util.spec_from_file_location('fusion_bench', ROOT/'tools/research/phase6/fusion_bench.py')
fb = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'dev', 'priors', 'sweep', 'fusion', 'out', 'manifest-out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    pri = np.load(a.priors)
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    ref = json.loads(Path(a.fusion).read_text())
    proto = json.loads((ROOT/'research/phase6/protocol.json').read_text())['fusion']['tuning']
    P = len(dev)
    order = np.random.default_rng(proto['seed']).permutation(P)
    folds = [order[f::proto['outer_folds']] for f in range(proto['outer_folds'])]
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in dev}
    X, Y = {}, {}
    for n in dev:
        cat = cats[n]
        F = bl.features(cat, store[n]['low'], store[n]['high'])
        idx = np.flatnonzero(cat['feasible'])
        X[n] = F
        Y[n] = (idx, np.array([all(c in sw['parts'][n][str(cat['label'][i])] and sw['parts'][n][str(cat['label'][i])][c][0]
                                    for c in std) for i in idx], int))
    out_rows, missing = {}, {}
    oof_h, oof_y = [], []
    manifest = dict(label='robust: held (rise >= 80 mm, slip < 20 mm) in all six standard conditions, v2 physics',
                    features=list(bl.FEATURE_NAMES), model='HistGradientBoostingClassifier, Phase V settings, seed 0',
                    sweep_sha256=hashlib.sha256(Path(a.sweep).read_bytes()).hexdigest(),
                    outer_folds_seed=proto['seed'], folds=[])
    from sklearn.metrics import roc_auc_score
    for k, test in enumerate(folds):
        train = [dev[i] for i in np.setdiff1d(np.arange(P), test)]
        Xt = np.vstack([X[n][Y[n][0]] for n in train]); yt = np.concatenate([Y[n][1] for n in train])
        model = bl.fit(Xt, yt, 0)
        manifest['folds'].append(dict(fold=k, train_parts=len(train), test_parts=[dev[i] for i in test],
                                      train_candidates=int(len(yt)), train_robust_rate=round(float(yt.mean()), 4)))
        for i in test:
            n = dev[i]
            cat = cats[n]
            h = model.predict_proba(X[n])[:, 1]
            oof_h.append(h[Y[n][0]]); oof_y.append(Y[n][1])
            pl = pri[f'{n}|learned'].astype(float); pt = pl/pl.max()
            s = pt*h
            o = fusion.order(cat, s)
            first = str(cat['label'][o[np.flatnonzero(cat['feasible'][o])[0]]])
            if 'load 2 kg' not in sw['parts'][n][first]:
                missing.setdefault(n, []).append(first)
                continue
            out_rows[n] = fb.outcome(cat, o, sw['parts'][n], std)
    if missing:
        Path(a.out).with_name('missing_first_grasps.json').write_text(json.dumps(
            {n: missing.get(n, []) for n in dev}, indent=1))
        raise SystemExit(f'{sum(map(len, missing.values()))} first grasps lack the load conditions: run sweep.py full on '
                         f'{Path(a.out).with_name("missing_first_grasps.json")} and merge, then run this again')
    H, YY = np.concatenate(oof_h), np.concatenate(oof_y)
    res = {k: np.array([out_rows[n][k] if out_rows[n][k] is not None else np.nan for n in dev], float)
           for k in ('nf', 'robust', 'nfr', 'load', 'nominal')}
    has = ~np.isnan(res['nfr'])
    doc = dict(note='Development parts, outer folds of the fusion benchmark; exploratory (stage D gate opened: '
                    'centring AUC for robustness 0.57 < 0.70).',
               p_h_given_f=dict(oof_auc=round(float(roc_auc_score(YY, H)), 4),
                                oof_brier=round(float(np.mean((H-YY)**2)), 4), base_rate=round(float(YY.mean()), 4)),
               summary=dict(N_F_gmean=fb.gmean(res['nf']), robust_rate=round(float(res['robust'].mean()), 4),
                            N_FR_gmean=fb.gmean(res['nfr'][has]), load_tolerant_rate=round(float(res['load'].mean()), 4),
                            nominal_held_rate=round(float(res['nominal'].mean()), 4)),
               contrasts={})
    for b in ('M0', 'M1', 'M4'):
        rb = {k: np.array([ref['per_part'][b][n][k] if ref['per_part'][b][n][k] is not None else np.nan for n in dev], float)
              for k in ('nf', 'robust', 'nfr', 'load')}
        hh = has & ~np.isnan(rb['nfr'])
        doc['contrasts'][f'vs_{b}'] = dict(
            N_F=fb.ratio(res['nf'], rb['nf']), N_FR=fb.ratio(res['nfr'][hh], rb['nfr'][hh]),
            robust=dict(difference=mech.bootstrap_diff(res['robust'], rb['robust']), **mech.mcnemar(res['robust'], rb['robust'])),
            load_tolerant=dict(difference=mech.bootstrap_diff(res['load'], rb['load']), **mech.mcnemar(res['load'], rb['load'])))
    doc['per_part'] = {n: {k: (None if np.isnan(res[k][i]) else int(res[k][i])) for k in res} for i, n in enumerate(dev)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    Path(a.manifest_out).write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in doc.items() if k != 'per_part'})[:2500])


if __name__ == '__main__':
    main()
