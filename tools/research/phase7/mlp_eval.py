"""Evaluate the representation variants (PHASE7_PROTOCOL.md, I.3).

    python tools/research/phase7/mlp_eval.py --pred-dir mlp/ --catalogues cat80/ \
        --dev research/phase6/dev_parts.json --sweep research/phase6/results/sweep.json.gz --out-dir results/

Per variant and seed, every held-out part is ordered as M1 orders it (preliminary + 2 p/max, gate
first) with the exact planner-order simulator. Per-part metrics are averaged over seeds; contrasts
are paired by part. Writes mlp_benchmark.json, linear_projection_benchmark.json,
visual_ablation.json and ranking_loss_benchmark.json.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import catalogue  # noqa: E402
from research_grasping.phase5.stats import _boot, holm  # noqa: E402

KS = (1, 5, 10, 20, 50)
CONTRASTS = [('L8', 'L2'), ('L7', 'L2'), ('L8', 'L10'), ('L8', 'L9'), ('L8', 'L8p'), ('L2', 'L0'), ('L11', 'L2')]
LINEAR = [('L4', 'L6'), ('L3', 'L5'), ('L10', 'L8'), ('L1', 'L0'), ('L2', 'L1')]
VISUAL = [('L5', 'L2v'), ('L6', 'L2v'), ('L7', 'L2'), ('L8', 'L2'), ('L9', 'L2'), ('L11', 'L2'), ('L8p', 'L2'),
          ('L8', 'L8p'), ('L8', 'L9'), ('L6', 'L5'), ('L8', 'L7'), ('L10', 'L2'), ('L10', 'L10p'), ('L10', 'L10r'),
          ('L10p', 'L2'), ('L10r', 'L2')]  # the L10 rows were added after L10 turned out best (post hoc)
RANKING = [('L2-rank', 'L2'), ('L8-rank', 'L8'), ('L8-rank', 'L2-rank')]
LOCAL = [('LG', 'L2'), ('LD', 'L2'), ('LGD', 'L2'), ('LD', 'LG'), ('LGD', 'LG'), ('LGD', 'LD')]
SENS = [('L2-k8', 'L2'), ('L2-k32', 'L2'), ('L6-k8', 'L6'), ('L6-k32', 'L6'), ('L8-k8', 'L8'), ('L8-k32', 'L8')]


def ece(p, y, bins=15):
    idx = np.clip((p*bins).astype(int), 0, bins-1)
    return float(sum((idx == b).mean()*abs(y[idx == b].mean()-p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def part_metrics(cat, p, dyn, std):
    order = catalogue.order(cat, p/max(float(p.max()), 1e-12), 2.)
    feas = cat['feasible'][order]
    hits = np.flatnonzero(feas)
    total = int(feas.sum())
    m = dict(nf=int(hits[0])+1 if total else None)
    for k in KS:
        m[f's@{k}'] = int(total > 0 and hits[0] < k)
        m[f'p@{k}'] = float(feas[:k].mean())
        m[f'r@{k}'] = float(feas[:k].sum()/total) if total else None
    if dyn is not None and total:
        first, nfr = None, None
        for pos in hits:
            d = dyn[str(cat['label'][order[pos]])]
            r = all(c in d and d[c][0] for c in std)
            if first is None:
                first = int(r)
            if r:
                nfr = int(pos)+1
                break
        m['robust'], m['nfr'] = first, nfr
    return m


def ratio(a, b):
    d = np.log(np.asarray(a)) - np.log(np.asarray(b))
    nz = d[d != 0]
    return dict(ratio=round(float(np.exp(d.mean())), 4), ci95=[round(x, 4) for x in _boot(d, 95)], parts=int(len(d)),
                better=int((d < 0).sum()), worse=int((d > 0).sum()),
                wilcoxon_p=float(st.wilcoxon(nz).pvalue) if len(nz) else 1.)


def diff(a, b, seed=20261128, n=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    dd = (a[idx]-b[idx]).mean(1)
    d = a-b
    nz = d[d != 0]
    return dict(difference=round(float(d.mean()), 4), ci95=[round(float(np.percentile(dd, 2.5)), 4),
                                                           round(float(np.percentile(dd, 97.5)), 4)],
                parts=int(len(a)), wilcoxon_p=float(st.wilcoxon(nz).pvalue) if len(nz) else 1.)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('pred-dir', 'catalogues', 'dev', 'sweep', 'out-dir'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--local-dir', help='local_features.py output, for coverage statistics')
    a = p.parse_args()
    pd = Path(a.pred_dir)
    dev = set(json.loads(Path(a.dev).read_text())['parts'])
    sw = json.load(gzip.open(a.sweep, 'rt'))
    std = sw['standard']
    files = sorted(pd.glob('*-s*.npz'))
    variants = sorted({f.stem.rsplit('-s', 1)[0] for f in files})
    names = None
    cats = {}
    res, info = {}, {}
    for v in variants:
        seeds = sorted(f for f in files if f.stem.rsplit('-s', 1)[0] == v)
        per_seed, aucs, lls, eces, meta = [], [], [], [], []
        for f in seeds:
            z = np.load(f)
            names = sorted(z.files) if names is None else names
            for n in names:
                if n not in cats:
                    cats[n] = catalogue.load(Path(a.catalogues)/f'{n}.npz')
            P = np.concatenate([z[n] for n in names]).astype(float)
            Y = np.concatenate([cats[n]['feasible'] for n in names]).astype(int)
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(Y, P))
            pc = np.clip(P, 1e-6, 1-1e-6)
            lls.append(float(-np.mean(Y*np.log(pc)+(1-Y)*np.log(1-pc))))
            eces.append(ece(P, Y))
            per_seed.append({n: part_metrics(cats[n], z[n].astype(float), sw['parts'].get(n) if n in dev else None, std)
                             for n in names})
            meta.append(json.loads(f.with_suffix('.json').read_text()))
        # seed-averaged per-part values
        agg = {}
        for n in names:
            ms = [s[n] for s in per_seed]
            row = {}
            if ms[0]['nf'] is not None:
                row['lognf'] = float(np.mean([np.log(m['nf']) for m in ms]))
            for k in KS:
                row[f's@{k}'] = float(np.mean([m[f's@{k}'] for m in ms]))
                row[f'p@{k}'] = float(np.mean([m[f'p@{k}'] for m in ms]))
                if ms[0][f'r@{k}'] is not None:
                    row[f'r@{k}'] = float(np.mean([m[f'r@{k}'] for m in ms]))
            if 'robust' in ms[0]:
                row['robust'] = float(np.mean([m['robust'] for m in ms]))
                if all(m['nfr'] for m in ms):
                    row['lognfr'] = float(np.mean([np.log(m['nfr']) for m in ms]))
            agg[n] = row
        res[v] = agg
        viable = [n for n in names if 'lognf' in agg[n]]
        seed_nf = [float(np.exp(np.mean([np.log(s[n]['nf']) for n in viable]))) for s in per_seed]
        info[v] = dict(
            seeds=len(seeds), config=meta[0]['config'],
            auc=round(float(np.mean(aucs)), 4), auc_seeds=[round(x, 4) for x in aucs],
            log_loss=round(float(np.mean(lls)), 4), ece=round(float(np.mean(eces)), 4),
            N_F_gmean=round(float(np.exp(np.mean([agg[n]['lognf'] for n in viable]))), 3),
            N_F_gmean_by_seed=[round(x, 3) for x in seed_nf],
            **{f'success@{k}': round(float(np.mean([agg[n][f's@{k}'] for n in names])), 4) for k in KS},
            **{f'precision@{k}': round(float(np.mean([agg[n][f'p@{k}'] for n in names])), 4) for k in KS},
            **{f'recall@{k}': round(float(np.mean([agg[n][f'r@{k}'] for n in viable])), 4) for k in KS},
            robust_first_dev=round(float(np.mean([agg[n]['robust'] for n in names if 'robust' in agg[n]])), 4),
            N_FR_gmean_dev=round(float(np.exp(np.mean([agg[n]['lognfr'] for n in names if 'lognfr' in agg[n]]))), 3),
            params=meta[0]['folds'][0].get('params'),
            train_s_per_fold=round(float(np.mean([f['train_s'] for m in meta for f in m['folds']])), 2),
            predict_ms_per_part=round(1e3*float(np.mean([f['predict_s'] for m in meta for f in m['folds']]))/(len(names)/5), 3),
            epochs=[f.get('epochs') for f in meta[0]['folds']])

    def contrasts(pairs):
        out = {'N_F': [], 'success@1': [], 'success@5': [], 'robust_first_dev': []}
        for x, y in pairs:
            if x not in res or y not in res:
                continue
            both = [n for n in names if 'lognf' in res[x][n] and 'lognf' in res[y][n]]
            out['N_F'].append(dict(a=x, b=y, **ratio(np.exp([res[x][n]['lognf'] for n in both]),
                                                    np.exp([res[y][n]['lognf'] for n in both]))))
            for k in (1, 5):
                out[f'success@{k}'].append(dict(a=x, b=y, **diff([res[x][n][f's@{k}'] for n in names],
                                                                 [res[y][n][f's@{k}'] for n in names])))
            dv = [n for n in names if 'robust' in res[x][n]]
            out['robust_first_dev'].append(dict(a=x, b=y, **diff([res[x][n]['robust'] for n in dv],
                                                                 [res[y][n]['robust'] for n in dv])))
        for key in out:
            if out[key]:
                holm(out[key])
        return out
    od = Path(a.out_dir)
    main_c = contrasts(CONTRASTS)
    adds = {}
    for vis in ('L7', 'L8', 'L11'):
        c2 = next((c for c in main_c['N_F'] if c['a'] == vis and c['b'] == 'L2'), None)
        cp = next((c for c in contrasts([(vis, 'L8p')])['N_F']), None) if vis in res and 'L8p' in res else None
        if c2:
            adds[vis] = dict(N_F_vs_L2_ci95=c2['ci95'], N_F_vs_permuted_ci95=cp['ci95'] if cp else None,
                             adds_value=bool(c2['ci95'][1] < 1 and cp is not None and cp['ci95'][1] < 1))
    common = dict(note='Development only (316 parts: 116 library + 200 Phase V test). Out-of-fold predictions, '
                       '5 folds grouped by product name, 3 seeds for MLPs (1 for L0/L1); per-part metrics averaged over '
                       'seeds. Robustness and N_FR on the 151 parts with Phase VI v2 dynamics.',
                  parts=len(names), viable_parts=len([n for n in names if cats[n]['feasible'].any()]))
    (od/'mlp_benchmark.json').write_text(json.dumps(dict(**common, variants=info, contrasts=main_c,
                                                         visual_adds_value_rule=adds,
                                                         sensitivity=contrasts(SENS)), indent=1) + '\n')
    (od/'linear_projection_benchmark.json').write_text(json.dumps(dict(**common, contrasts=contrasts(LINEAR),
                                                                       variants={k: info[k] for k in ('L0', 'L1', 'L3', 'L4', 'L5', 'L6', 'L8', 'L10') if k in info}), indent=1) + '\n')
    (od/'visual_ablation.json').write_text(json.dumps(dict(**common, contrasts=contrasts(VISUAL),
                                                           variants={k: info[k] for k in info if k in {x for p in VISUAL for x in p}}), indent=1) + '\n')
    (od/'ranking_loss_benchmark.json').write_text(json.dumps(dict(**common, contrasts=contrasts(RANKING),
                                                                  variants={k: info[k] for k in ('L2', 'L8', 'L2-rank', 'L8-rank') if k in info}), indent=1) + '\n')
    if any(v in info for v in ('LG', 'LD', 'LGD')):
        cov = None
        if a.local_dir:
            vis, off, nv = [], [], 0
            for f in sorted(Path(a.local_dir).glob('*.npz')):
                z = np.load(f)
                vis.append(np.concatenate([z['vis_a'] > 0, z['vis_b'] > 0])); off.append(np.concatenate([z['offset_a'], z['offset_b']]))
                nv += 1
            vis, off = np.concatenate(vis), np.concatenate(off)
            cov = dict(parts=nv, contacts=int(len(vis)), visible_in_at_least_one_view=round(float(vis.mean()), 4),
                       snap_offset_mm=dict(median=round(float(np.median(off))*1e3, 2), p90=round(float(np.percentile(off, 90))*1e3, 2)))
        (od/'local_features_benchmark.json').write_text(json.dumps(dict(
            **common, design='PHASE7_PROTOCOL.md I.3 (local visual features): DINOv3 patch tokens sampled at the two contacts '
                             '(snapped to the render surface; 8 views, 224 px), PCA 16 per contact fitted on the training fold; '
                             'classical local geometry (eigenvalue ratios and normal angle within 10 mm); appended to v; '
                             'same MLP with u as L2', coverage=cov, contrasts=contrasts(LOCAL),
            variants={k: info[k] for k in ('L2', 'LG', 'LD', 'LGD') if k in info}), indent=1) + '\n')
    for v in sorted(info):
        i = info[v]
        print(f"{v:8s} AUC {i['auc']} ece {i['ece']} N_F {i['N_F_gmean']} s@1 {i['success@1']} s@5 {i['success@5']} "
              f"R {i['robust_first_dev']} N_FR {i['N_FR_gmean_dev']} params {i['params']} train {i['train_s_per_fold']}s")


if __name__ == '__main__':
    main()
