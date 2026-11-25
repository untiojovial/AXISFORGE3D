"""Paired statistics for a grasp-prior evaluation, as fixed in the protocol.

For each contrast A vs B and each target (first, third feasible grasp):
the geometric mean over held-out parts of examined(A) / examined(B), a 95 %
percentile bootstrap interval (parts resampled with replacement, 10 000
draws, fixed seed), parts better / equal / worse, and a two-sided Wilcoxon
signed-rank test on log(A/B) (zero differences dropped). The primary
contrasts are Holm-adjusted together. Methods with several seeds (random
neighbours) enter as the per-part mean.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SEED = 20261126
DRAWS = 10_000


def per_part(rows, method, target):
    values = {}
    for r in rows:
        if r['method'] == method and r['nr_grasps'] == target:
            values.setdefault(r['part'], []).append(r['examined'])
    return {p: float(np.mean(v)) for p, v in values.items()}


def contrast(rows, a, b, target):
    va, vb = per_part(rows, a, target), per_part(rows, b, target)
    parts = sorted(set(va) & set(vb))
    if not parts:
        return None
    d = np.log(np.array([va[p] for p in parts])/np.array([vb[p] for p in parts]))
    rng = np.random.default_rng(SEED)
    boot = np.exp(d[rng.integers(0, len(d), size=(DRAWS, len(d)))].mean(1))
    nonzero = d[d != 0]
    p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 1 else 1.0
    return dict(a=a, b=b, target=target, parts=len(parts), ratio=float(np.exp(d.mean())),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                better=int((d < 0).sum()), equal=int((d == 0).sum()), worse=int((d > 0).sum()),
                wilcoxon_p=p, median_a=float(np.median([va[q] for q in parts])),
                median_b=float(np.median([vb[q] for q in parts])))


def holm(results):
    order = sorted(range(len(results)), key=lambda i: results[i]['wilcoxon_p'])
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(results)-rank)*results[i]['wilcoxon_p']))
        results[i]['holm_p'] = running
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--primary', nargs=2, action='append', metavar=('A', 'B'), default=[])
    p.add_argument('--secondary', nargs=2, action='append', metavar=('A', 'B'), default=[])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rows = json.loads(Path(a.evaluation).read_text())['rows']
    targets = sorted({r['nr_grasps'] for r in rows})
    primary = [c for c in (contrast(rows, x, y, 1) for x, y in a.primary) if c]
    holm(primary)
    secondary = [c for x, y in a.primary + a.secondary for t in targets
                 if (c := contrast(rows, x, y, t)) and not (t == 1 and [x, y] in a.primary)]
    out = dict(seed=SEED, draws=DRAWS, primary=primary, secondary=secondary)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for c in primary + secondary:
        tag = 'PRIMARY ' if c in primary else ''
        extra = f"  Holm p {c['holm_p']:.4f}" if 'holm_p' in c else ''
        print(f"{tag}{c['a']} vs {c['b']} (to {c['target']}): ratio {c['ratio']:.3f} "
              f"[{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]  {c['better']}/{c['equal']}/{c['worse']} of {c['parts']}"
              f"  p {c['wilcoxon_p']:.4f}{extra}")


if __name__ == '__main__':
    main()
