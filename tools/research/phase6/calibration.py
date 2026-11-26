"""Calibration of the feasibility classifier (stage D, always run: it is cheap).

    python tools/research/phase6/calibration.py --catalogues cat80/ --library research/phase5/library_parts.txt \
        --dev research/phase6/dev_parts.json --priors priors.npz --out calibration_results.json

Raw probabilities of the Phase V classifier: out of fold on the library (5
folds by part) and on the development parts (trained on the whole library).
Platt and isotonic maps are fitted on the library out-of-fold predictions only
and applied to the development parts. Brier score, negative log-likelihood and
expected calibration error (15 equal-width bins), with bootstrap intervals over
parts. The planner's prior is p / max(p) per part, which is not a probability;
its calibration is reported only to show that.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue  # noqa: E402

EPS = 1e-6


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins+1)
    idx = np.clip(np.digitize(p, edges)-1, 0, bins-1)
    total = 0.
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean()*abs(y[m].mean() - p[m].mean())
    return float(total)


def metrics(p, y):
    p = np.clip(p, EPS, 1-EPS)
    return dict(brier=float(np.mean((p-y)**2)), nll=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))), ece=ece(p, y),
                mean_p=float(p.mean()), rate=float(y.mean()))


def by_part_ci(ps, ys, fn, seed=20261128, draws=2000):
    rng = np.random.default_rng(seed)
    n = len(ps)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        vals.append(fn(np.concatenate([ps[i] for i in idx]), np.concatenate([ys[i] for i in idx])))
    return [round(float(np.percentile(vals, 2.5)), 5), round(float(np.percentile(vals, 97.5)), 5)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('catalogues', 'library', 'dev', 'priors', 'out'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    library = [s.strip() for s in Path(a.library).read_text().splitlines() if s.strip()]
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    pri = np.load(a.priors)
    lab = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz')['feasible'].astype(float) for n in library+dev}
    pl = [pri[f'{n}|oof'].astype(float) for n in library]
    yl = [lab[n] for n in library]
    pd = [pri[f'{n}|learned'].astype(float) for n in dev]
    yd = [lab[n] for n in dev]
    PL, YL = np.concatenate(pl), np.concatenate(yl)
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    logit = lambda x: np.log(np.clip(x, EPS, 1-EPS)/(1-np.clip(x, EPS, 1-EPS)))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(PL)[:, None], YL)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(PL, YL)
    maps = {'raw': lambda x: x, 'platt': lambda x: platt.predict_proba(logit(x)[:, None])[:, 1],
            'isotonic': lambda x: iso.predict(x), 'planner prior p/max (not a probability)': None}
    out = dict(note='Library: out-of-fold (5 folds by part). Development: classifier trained on the whole library; '
                    'calibration maps fitted on the library out-of-fold predictions only.',
               library=dict(parts=len(library), candidates=int(len(YL)), raw=metrics(PL, YL)),
               development={})
    for name, fn in maps.items():
        if fn is None:
            q = [x/max(x.max(), EPS) for x in pd]
        else:
            q = [fn(x) for x in pd]
        Q, Y = np.concatenate(q), np.concatenate(yd)
        m = metrics(Q, Y)
        m['ci95'] = {k: by_part_ci(q, yd, lambda pp, yy, k=k: metrics(pp, yy)[k]) for k in ('brier', 'ece')}
        # reliability table
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(Q, edges)-1, 0, 9)
        m['reliability'] = [dict(bin=f'{edges[b]:.1f}-{edges[b+1]:.1f}', n=int((idx == b).sum()),
                                 mean_p=round(float(Q[idx == b].mean()), 4) if (idx == b).any() else None,
                                 rate=round(float(Y[idx == b].mean()), 4) if (idx == b).any() else None)
                            for b in range(10)]
        out['development'][name] = m
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
