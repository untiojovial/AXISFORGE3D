"""Paired, descriptive comparison of the Optuna-tuned MLP families with the Phase VII references (exploratory).

    python tools/research/phase7_extension/optuna_compare.py --optuna-dir optuna/ --mlp-dir mlp/ \
        --catalogues cat80/ --out research/phase7_extension/results/optuna_contrasts.json

Defined after the search had started and before any out-of-fold metric of the tuned models was computed. It is
descriptive: it supports no selection and no claim. Conventions follow tools/research/phase7/mlp_eval.py, so the
numbers are comparable with mlp_benchmark.json:

- each part is ordered as M1 orders it (preliminary + 2 p/max) with the exact planner-order simulator;
- N_F is the 1-based position of the first feasible candidate; parts without a feasible candidate are left out;
- per-part log N_F is averaged over seeds (5 for the tuned families, 3 for the fixed MLPs, 1 for L0);
- a contrast A / B is the geometric mean of the per-part N_F ratios, paired by part, with the percentile
  bootstrap of research_grasping.phase5.stats (same seed and draws), the counts of parts better and worse, a
  Wilcoxon signed-rank p and Holm's adjustment within each group of contrasts.

Groups:

- tuned vs fixed: each tuned family against the fixed-architecture MLP with the same inputs (OpenCLIP has no
  fixed counterpart with the same inputs: L11 also has the geometry block);
- tuned vs L0: each tuned family against gradient boosting, the model family of the final method;
- visual vs control, tuned: each tuned family with frozen features against the tuned family without them, and the
  tuned combined family against its tuned permuted and untrained-ViT controls.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.stats as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import catalogue  # noqa: E402
from research_grasping.phase5.stats import _boot, holm  # noqa: E402

FIXED = {'candidate-only': 'L2v', 'geometry': 'L2', 'dinov2': 'L5', 'dinov3': 'L6', 'combined': 'L8', 'linear': 'L10',
         'random-vit': 'L9', 'permuted': 'L8p'}
VISUAL = [('dinov2', 'candidate-only'), ('dinov3', 'candidate-only'), ('openclip', 'candidate-only'),
          ('combined', 'geometry'), ('linear', 'geometry'), ('combined', 'permuted'), ('combined', 'random-vit')]


def lognf(cat, p):
    order = catalogue.order(cat, p/max(float(p.max()), 1e-12), 2.)
    hits = np.flatnonzero(cat['feasible'][order])
    return float(np.log(hits[0] + 1)) if len(hits) else None


def per_part(preds, cats):
    """preds: list (one per seed) of {part: probabilities} -> {part: seed-averaged log N_F} on viable parts."""
    out = {}
    for n, cat in cats.items():
        v = [lognf(cat, s[n].astype(float)) for s in preds]
        if v[0] is not None:
            out[n] = float(np.mean(v))
    return out


def contrast(a, b):
    parts = sorted(set(a) & set(b))
    d = np.array([a[p] - b[p] for p in parts])
    nz = d[d != 0]
    return dict(ratio=round(float(np.exp(d.mean())), 4), ci95=[round(x, 4) for x in _boot(d, 95)], parts=len(parts),
                better=int((d < 0).sum()), worse=int((d > 0).sum()),
                wilcoxon_p=float(st.wilcoxon(nz).pvalue) if len(nz) else 1.)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('optuna-dir', 'mlp-dir', 'catalogues', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    od, md = Path(a.optuna_dir), Path(a.mlp_dir)
    names = sorted(np.load(md/'L0-s0.npz').files)
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in names}
    tuned = {}
    for fam in ['candidate-only', 'geometry', 'dinov2', 'dinov3', 'openclip', 'combined', 'linear', 'random-vit', 'permuted']:
        files = [od/f'{fam}-fold{k}-oof.npz' for k in range(5)]
        if not all(f.exists() for f in files):
            continue
        seeds = {}
        for f in files:
            z = np.load(f)
            for key in z.files:
                part, s = key.rsplit('|s', 1)
                seeds.setdefault(int(s), {})[part] = z[key]
        assert all(set(s) == set(names) for s in seeds.values()), fam
        tuned[fam] = per_part([seeds[s] for s in sorted(seeds)], cats)
    refs = {}
    for v in sorted(set(FIXED.values()) | {'L0'}):
        refs[v] = per_part([np.load(f) for f in sorted(md.glob(f'{v}-s[0-9].npz'))], cats)
    groups = {'tuned vs fixed': [(f, FIXED[f]) for f in tuned if f in FIXED],
              'tuned vs L0': [(f, 'L0') for f in tuned],
              'visual vs control, tuned': [(x, y) for x, y in VISUAL if x in tuned and y in tuned]}
    out = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), development_only=True, exploratory=True,
               families_complete=sorted(tuned),
               N_F_gmean={k: round(float(np.exp(np.mean(list(v.values())))), 4) for k, v in {**tuned, **refs}.items()},
               viable_parts={k: len(v) for k, v in {**tuned, **refs}.items()}, contrasts={})
    for g, pairs in groups.items():
        rows = {f'{x} / {y}': contrast(tuned[x], tuned[y] if y in tuned else refs[y]) for x, y in pairs}
        holm(list(rows.values()))  # adds holm_p to each row, in place
        out['contrasts'][g] = rows
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for g, rows in out['contrasts'].items():
        print(g)
        for k, r in rows.items():
            print(f'  {k:28s} {r["ratio"]:.3f} [{r["ci95"][0]:.3f}, {r["ci95"][1]:.3f}] better {r["better"]} worse {r["worse"]} '
                  f'holm {r["holm_p"]:.3f}')


if __name__ == '__main__':
    main()
