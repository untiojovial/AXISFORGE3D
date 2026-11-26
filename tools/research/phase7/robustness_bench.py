"""Development robustness models and FORTIS-R Joint (PHASE7_PROTOCOL.md, I.3 and I.4).

    python tools/research/phase7/robustness_bench.py --catalogues cat80/ --store store.json \
        --dev research/phase6/dev_parts.json --priors priors.npz --sweep sweep.json.gz \
        --out robustness_benchmark.json --missing-out missing.json

Phase VI outer folds (seed 20261130) over the 151 development parts. For each fold, P(H|F) is
trained on the viable candidates of the training parts (label: robust in the six standard
conditions, Phase VI v2), as gradient boosting (M9, historical), MLP(v, u) and MLP(v, u, DINOv3)
with the Phase VII fixed architecture (seed 0). Orderings: p * h (M9 family) and
FORTIS-R Joint = p * h * Q_cen^beta, beta chosen inside each training fold by the protocol's
criterion (standardised mean log N_FR + share of first grasps not load tolerant). First grasps whose
load conditions are missing are written to --missing-out; rerun after simulating them.
"""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase6 import fusion, mechanism as mech  # noqa: E402
from research_grasping.phase7 import representation as rep  # noqa: E402

spec = importlib.util.spec_from_file_location('fb6', ROOT/'tools/research/phase6/fusion_bench.py')
fb6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fb6)
BETAS = (.1, .2, .3, .5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'dev', 'priors', 'sweep', 'out', 'missing-out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    pz = np.load(a.priors)
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in dev}
    pri = {n: pz[f'{n}|learned'].astype(float)/pz[f'{n}|learned'].max() for n in dev}
    F = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in dev}
    data = {}
    for n in dev:
        idx = np.flatnonzero(cats[n]['feasible'])
        lab = [all(c in sw['parts'][n][str(cats[n]['label'][i])] and sw['parts'][n][str(cats[n]['label'][i])][c][0] for c in std)
               for i in idx]
        desc = store[n]['descriptors']
        data[n] = dict(idx=idx, y=np.array(lab, np.float32), v=F[n][:, rep.V_COLUMNS].astype(np.float32),
                       u=np.concatenate([F[n][0, rep.EXTENT_COLUMNS], np.asarray(desc['d2'], float)]).astype(np.float32),
                       d=np.asarray(desc['dinov3'], np.float32))
    order = np.random.default_rng(20261130).permutation(len(dev))
    folds = [order[f::5] for f in range(5)]
    models = ('hgb', 'mlp-geo', 'mlp-geo-dinov3')
    H = {m: {} for m in models}
    for test in folds:
        train = [dev[i] for i in np.setdiff1d(np.arange(len(dev)), test)]
        Xt = np.vstack([F[n][data[n]['idx']] for n in train]); yt = np.concatenate([data[n]['y'] for n in train])
        hgb = bl.fit(Xt, yt.astype(int), 0)
        for i in test:
            H['hgb'][dev[i]] = hgb.predict_proba(F[dev[i]])[:, 1]
        rng = np.random.default_rng(77)
        val = set(rng.choice(train, max(1, len(train)//7), replace=False))
        tr, va = [n for n in train if n not in val], [n for n in train if n in val]
        for m, vis in (('mlp-geo', False), ('mlp-geo-dinov3', True)):
            sub = {n: dict(v=data[n]['v'][data[n]['idx']], y=data[n]['y'], u=data[n]['u'], d=data[n]['d']) for n in train}
            model = rep.MLPModel(geometry=True, visual=vis, seed=0).fit(sub, tr, va)
            full = {dev[i]: dict(v=data[dev[i]]['v'], y=np.zeros(len(data[dev[i]]['v']), np.float32),
                                 u=data[dev[i]]['u'], d=data[dev[i]]['d']) for i in test}
            H[m].update(model.predict(full, [dev[i] for i in test]))
    missing = {}

    def outcome(n, s):
        cat = cats[n]
        o = fusion.order(cat, s)
        first = str(cat['label'][o[np.flatnonzero(cat['feasible'][o])[0]]])
        if 'load 2 kg' not in sw['parts'][n][first]:
            missing.setdefault(n, set()).add(first)
            return None
        return fb6.outcome(cat, o, sw['parts'][n], std)
    res = {}
    for m in models:
        res[f'M9[{m}]'] = {n: outcome(n, pri[n]*H[m][n]) for n in dev}
        for b in BETAS:
            res[f'Joint[{m},b={b}]'] = {n: outcome(n, pri[n]*H[m][n]*np.maximum(fusion.quality(cats[n], 'cen'), 1e-3)**b) for n in dev}
    if missing:
        Path(a.missing_out).write_text(json.dumps({n: sorted(v) for n, v in missing.items()}, indent=1))
        raise SystemExit(f'{sum(map(len, missing.values()))} first grasps lack load conditions -> {a.missing_out}')
    # nested choice of beta per outer fold (criterion fixed in the protocol)
    def crit(rows):
        nfr = np.array([r['nfr'] for r in rows if r['nfr']], float)
        return np.mean(np.log(nfr)), 1 - np.mean([r['load'] for r in rows])
    for m in models:
        chosen, rows = [], {}
        for test in folds:
            train = [dev[i] for i in np.setdiff1d(np.arange(len(dev)), test)]
            c = {b: crit([res[f'Joint[{m},b={b}]'][n] for n in train]) for b in BETAS}
            x = np.array([v[0] for v in c.values()]); y = np.array([v[1] for v in c.values()])
            z = (x-x.mean())/(x.std()+1e-12) + (y-y.mean())/(y.std()+1e-12)
            b = BETAS[int(np.argmin(z))]
            chosen.append(b)
            rows.update({dev[i]: res[f'Joint[{m},b={b}]'][dev[i]] for i in test})
        res[f'Joint[{m}]'] = rows
        res[f'Joint[{m}]::chosen'] = chosen
    p6 = json.loads((ROOT/'research/phase6/results/fusion_benchmark.json').read_text())['per_part']
    ref = {k: {n: p6[k][n] for n in dev} for k in ('M0', 'M1', 'M4')}

    def summ(rows):
        nf = np.array([rows[n]['nf'] for n in dev], float)
        nfr = np.array([rows[n]['nfr'] or np.nan for n in dev], float)
        return dict(N_F_gmean=round(float(np.exp(np.log(nf).mean())), 3),
                    robust_rate=round(float(np.mean([rows[n]['robust'] for n in dev])), 4),
                    N_FR_gmean=round(float(np.exp(np.nanmean(np.log(nfr)))), 3),
                    load_tolerant_rate=round(float(np.mean([rows[n]['load'] for n in dev])), 4))
    allr = {**ref, **{k: v for k, v in res.items() if not k.endswith('::chosen')}}
    out = dict(note='Development (151 parts), Phase VI outer folds; v2 labels (rise and slip). Exploratory.',
               summary={k: summ(v) for k, v in allr.items()},
               joint_beta_chosen={k.split('::')[0]: v for k, v in res.items() if k.endswith('::chosen')},
               p_h_given_f_oof_auc={}, contrasts={})
    from sklearn.metrics import roc_auc_score
    for m in models:
        y = np.concatenate([data[n]['y'] for n in dev]); h = np.concatenate([H[m][n][data[n]['idx']] for n in dev])
        out['p_h_given_f_oof_auc'][m] = round(float(roc_auc_score(y, h)), 4)
    for x in [k for k in allr if k.startswith('M9[') or (k.startswith('Joint[') and ',b=' not in k)]:
        for b in ('M1', 'M4', 'M9[hgb]'):
            if x == b:
                continue
            A, B = allr[x], allr[b]
            out['contrasts'][f'{x} vs {b}'] = dict(
                N_F=fb6.ratio([A[n]['nf'] for n in dev], [B[n]['nf'] for n in dev]),
                robust=mech.bootstrap_diff([A[n]['robust'] for n in dev], [B[n]['robust'] for n in dev]),
                load_tolerant=mech.bootstrap_diff([A[n]['load'] for n in dev], [B[n]['load'] for n in dev]))
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out['summary'].items():
        print(k, v)
    print(out['joint_beta_chosen'], out['p_h_given_f_oof_auc'])


if __name__ == '__main__':
    main()
