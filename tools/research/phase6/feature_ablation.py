"""Feature-group ablation of the learned classifier (stage B).

    python tools/research/phase6/feature_ablation.py --catalogues cat80/ --store store.json \
        --library research/phase5/library_parts.txt --dev research/phase6/dev_parts.json \
        --sweep research/phase6/results/sweep.json.gz --out feature_ablation.json

Each variant drops one group of the 19 features, retrains on the library (seed
0) and orders the development parts as Phase V did (preliminary + 2 * p/max).
Reported: candidate AUC, N_F, robustness of the first viable grasp (six
standard conditions, from the v2 sweep) and N_FR, with ratios against the
full model over parts.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import baselines as bl, catalogue  # noqa: E402
from research_grasping.phase5.stats import _boot  # noqa: E402
from research_grasping.phase6 import mechanism as mech  # noqa: E402

GROUPS = {
    'none (full model)': [],
    'centre_offset': ['centre_offset'],
    'preliminary + makeable': ['preliminary', 'makeable'],
    'height + rel_z': ['height', 'rel_z'],
    'approach': ['approach_x', 'approach_y', 'approach_z'],
    'closing': ['closing_x', 'closing_y', 'closing_z'],
    'part size (log_extent)': ['log_extent_x', 'log_extent_y', 'log_extent_z'],
    'antipodal': ['antipodal_a', 'antipodal_b'],
    'width': ['width'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'store', 'library', 'dev', 'sweep', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    store = json.loads(Path(a.store).read_text())
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in library+dev}
    F = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in library+dev}
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    XL = np.vstack([F[n] for n in library]); yL = np.concatenate([cats[n]['feasible'] for n in library]).astype(int)
    from sklearn.metrics import roc_auc_score
    names = list(bl.FEATURE_NAMES)
    res = {}
    for g, drop in GROUPS.items():
        keep = [j for j, f in enumerate(names) if f not in drop]
        model = bl.fit(XL[:, keep], yL, 0)
        nf, rob, nfr, ps, ys = [], [], [], [], []
        for n in dev:
            cat = cats[n]
            pr = model.predict_proba(F[n][:, keep])[:, 1]
            ps.append(pr); ys.append(cat['feasible'].astype(int))
            order = catalogue.order(cat, pr/pr.max(), 2.)
            first, fr = None, None
            for k, i in enumerate(order):
                if not cat['feasible'][i]:
                    continue
                d = sw['parts'][n][str(cat['label'][i])]
                r = all(c in d and d[c][0] for c in std)
                if first is None:
                    first = (k+1, r)
                if r:
                    fr = k+1
                    break
            nf.append(first[0]); rob.append(first[1]); nfr.append(fr)
        res[g] = dict(nf=np.array(nf, float), robust=np.array(rob, float), nfr=np.array([x if x else np.nan for x in nfr]),
                      auc=float(roc_auc_score(np.concatenate(ys), np.concatenate(ps))))
    base = res['none (full model)']
    ok = ~np.isnan(base['nfr'])
    out = dict(note='Development parts; classifier retrained on the library without each feature group.', variants={})
    for g, r in res.items():
        d = np.log(r['nf']/base['nf'])
        dfr = np.log(r['nfr'][ok]/base['nfr'][ok])
        out['variants'][g] = dict(
            dropped=GROUPS[g], auc=round(r['auc'], 4), N_F_gmean=round(float(np.exp(np.log(r['nf']).mean())), 3),
            robust_rate=round(float(r['robust'].mean()), 4),
            N_FR_gmean=round(float(np.exp(np.nanmean(np.log(r['nfr'])))), 3),
            N_F_ratio_vs_full=dict(ratio=round(float(np.exp(d.mean())), 4), ci95=[round(x, 4) for x in _boot(d, 95)]),
            N_FR_ratio_vs_full=dict(ratio=round(float(np.exp(dfr.mean())), 4), ci95=[round(x, 4) for x in _boot(dfr, 95)]),
            robust_minus_full=mech.bootstrap_diff(r['robust'], base['robust']))
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for g, v in out['variants'].items():
        print(f"{g:24s} AUC {v['auc']} N_F {v['N_F_gmean']} x{v['N_F_ratio_vs_full']['ratio']} {v['N_F_ratio_vs_full']['ci95']} "
              f"R {v['robust_rate']} N_FR {v['N_FR_gmean']}")


if __name__ == '__main__':
    main()
