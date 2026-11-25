"""Phase V statistics on per-part outcomes (the part is the unit; rows that share a
part are never treated as independent).

* contrast: paired geometric-mean ratio of candidates examined, percentile
  bootstrap intervals (95 % and 90 %), parts better/equal/worse, two-sided
  Wilcoxon signed-rank p (zero differences dropped), and an equivalence
  verdict: equivalent when the 95 % interval lies inside [1/margin, margin]
  (two one-sided tests at 2.5 % each, conservative), plus paired-t TOST p.
* budget: share of all test parts (no-viable ones count as failures) with a
  feasible grasp within B candidates, with bootstrap intervals.
* holm: step-down adjustment within a family.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

SEED = 20261128
DRAWS = 10_000
BUDGETS = (1, 5, 10, 20, 50, 100)


def select(rows, experiment, gripper):
    return [r for r in rows if r['experiment'] == experiment and r['gripper'] == gripper]


def per_part(rows, method, target=1):
    """{part: examined} for parts with at least ``target`` feasible grasps.

    ``visual:vit-random`` and ``fusion:vit-random`` pool the five random
    initialisations; seeded methods average their seeds."""
    key = f'examined{target}'
    values = {}
    for r in rows:
        m = r['method']
        if method.endswith(':vit-random'):
            if not m.startswith(method+'-s'):
                continue
        elif m != method:
            continue
        if r['feasible'] >= target:
            values.setdefault(r['part'], []).append(r[key])
    return {p: float(np.mean(v)) for p, v in values.items()}


def _boot(d, level):
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), size=(DRAWS, len(d)))].mean(1)
    lo, hi = np.percentile(means, [(100-level)/2, 100-(100-level)/2])
    return [float(np.exp(lo)), float(np.exp(hi))]


def contrast(rows, a, b, target=1, margin=1.25):
    va, vb = per_part(rows, a, target), per_part(rows, b, target)
    parts = sorted(set(va) & set(vb))
    if len(parts) < 2:
        return None
    d = np.log(np.array([va[p] for p in parts])/np.array([vb[p] for p in parts]))
    nonzero = d[d != 0]
    ci95, ci90 = _boot(d, 95), _boot(d, 90)
    sd = d.std(ddof=1)
    se = sd/np.sqrt(len(d)) if sd > 0 else np.inf
    lm = np.log(margin)
    t_low = st.t.sf((d.mean()+lm)/se, len(d)-1) if np.isfinite(se) else 0.
    t_high = st.t.cdf((d.mean()-lm)/se, len(d)-1) if np.isfinite(se) else 0.
    return dict(a=a, b=b, target=target, parts=len(parts), ratio=float(np.exp(d.mean())), ci95=ci95, ci90=ci90,
                better=int((d < 0).sum()), equal=int((d == 0).sum()), worse=int((d > 0).sum()),
                wilcoxon_p=float(st.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
                margin=margin, equivalent=bool(ci95[0] >= 1/margin and ci95[1] <= margin),
                tost_p=float(max(t_low, t_high)),
                median_a=float(np.median([va[p] for p in parts])), median_b=float(np.median([vb[p] for p in parts])))


def holm(results, key='wilcoxon_p', out='holm_p'):
    order = sorted(range(len(results)), key=lambda i: results[i][key])
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(results)-rank)*results[i][key]))
        results[i][out] = running
    return results


def budget(rows, method, budgets=BUDGETS):
    """Success within each budget over every test part (no viable grasp = failure)."""
    per = {}
    for r in rows:
        m = r['method']
        if (method.endswith(':vit-random') and m.startswith(method+'-s')) or m == method:
            per.setdefault(r['part'], []).append((r['found1'], r['examined1']))
    parts = sorted(per)
    out = {}
    rng = np.random.default_rng(SEED)
    for b in budgets:
        s = np.array([np.mean([f and e <= b for f, e in per[p]]) for p in parts])
        boot = s[rng.integers(0, len(s), size=(DRAWS, len(s)))].mean(1)
        out[str(b)] = dict(rate=float(s.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
    return dict(method=method, parts=len(parts), budgets=out)


def negative_transfer(rows, method, reference='default', target=1):
    """Share of viable parts where the method examines more than the reference."""
    va, vb = per_part(rows, method, target), per_part(rows, reference, target)
    parts = sorted(set(va) & set(vb))
    worse = [p for p in parts if va[p] > vb[p]]
    return dict(method=method, reference=reference, parts=len(parts), worse=len(worse),
                rate=len(worse)/len(parts) if parts else None)
