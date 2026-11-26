"""Negative transfer and a fallback with a worst-case guarantee (development, no new simulation).

    python tools/research/phase7/negative_transfer.py --physics research/phase7/results/physics_dev_with_joint.json.gz \
        --dev research/phase6/dev_parts.json --catalogues cat80/ --store store.json --priors priors.npz \
        --sweep research/phase7/results/sweep_v2_extended.json.gz --out negative_transfer_analysis.json

Negative transfer on a part: a learned order needs more candidates than the original planner
(N_method > N_M0). The fallback studied is round-robin interleaving of the learned order L and the
original order D (L1, D1, L2, D2, ..., skipping candidates already examined).

Guarantee. Let a = position of the first viable candidate in L and b in D. After 2m steps of
interleaving, the first m candidates of both orders have been examined (or skipped because already
examined), so the hybrid stops no later than step 2 min(a, b) - 1 (if a <= b) or 2 b (if b < a):
N_hybrid <= 2 min(N_L, N_D) <= 2 N_D. With k orders in round robin the same argument gives
N_hybrid <= k min_i N_i. The bound is deterministic and needs no assumption about the prior.
The hybrid's grasp is the first viable candidate it reaches, i.e. that of L or of D; its dynamic
outcome is therefore read from trials already run (v3, torsion 0.005). Unit: the viable development part.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase6 import fusion  # noqa: E402
from research_grasping.phase6.mechanism import bootstrap_diff  # noqa: E402

STANDARD = ('nominal', 'mass x2', 'mass x0.5', 'friction 0.3', 'pose noise 1', 'pose noise 2')
LOADS = ('load 0.5 kg', 'load 1 kg', 'load 2 kg')


def first_viable_pos(cat, o):
    return int(np.flatnonzero(cat['feasible'][o])[0]) + 1


def interleave(cat, learned, default):
    """Round-robin of two orders without repeats; returns (N_hybrid, index of the grasp found, source)."""
    seen = set()
    n = 0
    for k in range(len(learned)):
        for src, o in (('learned', learned), ('default', default)):
            i = int(o[k])
            if i in seen:
                continue
            seen.add(i); n += 1
            if cat['feasible'][i]:
                return n, i, src
    raise ValueError('no viable candidate')


def gmean(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('physics', 'dev', 'catalogues', 'store', 'priors', 'sweep', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    ph = json.load(gzip.open(a.physics, 'rt'))
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    store = json.loads(Path(a.store).read_text())
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in dev}
    pz = np.load(a.priors)
    pri = {n: pz[f'{n}|learned'].astype(float)/pz[f'{n}|learned'].max() for n in dev}
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    betas = ph['joint_betas_by_fold']
    order = np.random.default_rng(20261130).permutation(len(dev))
    folds = [order[f::5] for f in range(5)]
    X = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in dev}
    orders = {n: {} for n in dev}
    for k, test in enumerate(folds):
        train = [dev[i] for i in np.setdiff1d(np.arange(len(dev)), test)]
        Xt, yt = [], []
        for n in train:
            idx = np.flatnonzero(cats[n]['feasible'])
            Xt.append(X[n][idx])
            yt.append([all(c in sw['parts'][n][str(cats[n]['label'][i])] and sw['parts'][n][str(cats[n]['label'][i])][c][0]
                           for c in std) for i in idx])
        model = bl.fit(np.vstack(Xt), np.concatenate(yt).astype(int), 0)
        for i in test:
            n = dev[i]; cat = cats[n]
            h = model.predict_proba(X[n])[:, 1]
            orders[n]['Joint'] = fusion.order(cat, pri[n]*h*np.maximum(fusion.quality(cat, 'cen'), 1e-3)**betas[k])
            orders[n]['M9'] = fusion.order(cat, pri[n]*h)
    for n in dev:
        cat = cats[n]
        orders[n]['M0'] = catalogue.order(cat)
        orders[n]['M1'] = catalogue.order(cat, pri[n], 2.)
        orders[n]['M4'] = fusion.order(cat, fusion.config_score('M4', dict(alpha=.8, q='cen'), cat, pri[n]))
    # check: first viable labels equal the ones simulated in physics_dev
    mismatch = [(n, m) for n in dev for m in ('M0', 'M1', 'M4', 'M9', 'Joint')
                if str(cats[n]['label'][orders[n][m][np.flatnonzero(cats[n]['feasible'][orders[n][m]])[0]]]) != ph['core'][n][m]]

    def outcome(n, label, which):
        t = ph['trials'][n][label]['0.005']
        return int(all(t[c]['held_v3'] for c in (STANDARD if which == 'robust' else LOADS)))

    res = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), parts=len(dev),
               first_grasp_label_mismatches=mismatch, methods={}, hybrids={})
    nd = {n: first_viable_pos(cats[n], orders[n]['M0']) for n in dev}
    for m in ('M1', 'M4', 'M9', 'Joint'):
        nm = {n: first_viable_pos(cats[n], orders[n][m]) for n in dev}
        ratio = np.array([nm[n]/nd[n] for n in dev])
        worse = [n for n in dev if nm[n] > nd[n]]
        res['methods'][m] = dict(
            N_F_gmean=round(gmean(list(nm.values())), 3), N_F_M0_gmean=round(gmean(list(nd.values())), 3),
            negative_transfer_parts=len(worse), better_parts=int(sum(nm[n] < nd[n] for n in dev)),
            ties=int(sum(nm[n] == nd[n] for n in dev)),
            worst_ratio=round(float(ratio.max()), 2), worst_part=dev[int(ratio.argmax())],
            extra_candidates_on_negative_parts=dict(total=int(sum(nm[n]-nd[n] for n in worse)),
                                                    median=float(np.median([nm[n]-nd[n] for n in worse])) if worse else 0.,
                                                    max=int(max([nm[n]-nd[n] for n in worse], default=0))))
        # interleaved fallback
        rows = {}
        for n in dev:
            cat = cats[n]
            nh, idx, src = interleave(cat, orders[n][m], orders[n]['M0'])
            lab = str(cat['label'][idx])
            assert nh <= 2*min(nm[n], nd[n]), (n, nh, nm[n], nd[n])
            rows[n] = dict(n=nh, label=lab, source=src)
        nh = {n: rows[n]['n'] for n in dev}
        sim = [n for n in dev if rows[n]['label'] in ph['trials'][n]]
        hyb = dict(
            N_F_gmean=round(gmean(list(nh.values())), 3),
            bound_holds_all_parts=True, max_ratio_to_M0=round(max(nh[n]/nd[n] for n in dev), 2),
            max_ratio_to_learned=round(max(nh[n]/nm[n] for n in dev), 2),
            parts_worse_than_M0=int(sum(nh[n] > nd[n] for n in dev)),
            grasp_source=dict(learned=int(sum(rows[n]['source'] == 'learned' for n in dev)),
                              default=int(sum(rows[n]['source'] == 'default' for n in dev))),
            parts_with_simulated_grasp=len(sim))
        if len(sim) == len(dev):
            for which in ('robust', 'load'):
                H = [outcome(n, rows[n]['label'], which) for n in dev]
                L = [outcome(n, ph['core'][n][m], which) for n in dev]
                D = [outcome(n, ph['core'][n]['M0'], which) for n in dev]
                hyb[f'{which}_rate'] = round(float(np.mean(H)), 4)
                hyb[f'{which}: hybrid - {m}'] = bootstrap_diff(H, L)
                hyb[f'{which}: hybrid - M0'] = bootstrap_diff(H, D)
        res['hybrids'][f'interleave({m}, M0)'] = hyb
    res['per_part_N_F'] = {n: {m: first_viable_pos(cats[n], orders[n][m]) for m in ('M0', 'M1', 'M4', 'M9', 'Joint')}
                           for n in dev}
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res['methods'], indent=1)); print(json.dumps(res['hybrids'], indent=1)); print('mismatch', mismatch)


if __name__ == '__main__':
    main()
