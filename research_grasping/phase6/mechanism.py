"""Stage B helpers: why the learned ordering's first grasps were less robust.

Everything here works per part: grasps and perturbations of one part are never
counted as independent parts, and every interval is a bootstrap over parts.
"""
from __future__ import annotations

import numpy as np

MAX_OPENING = .08
FINGER_DEPTH = .045


def preliminary_terms(cat, i, up=(0., 0., 1.)):
    """The four terms of the planner's preliminary score for candidate ``i`` (80 mm gripper).

    width and centring are exact; the force term is near 1 for a 0.1 kg part
    (it needs about 2 N of the 90 N available) and is recovered from the score.
    """
    width = float(np.clip((MAX_OPENING - cat['width'][i])/MAX_OPENING, 0, 1))
    co = float(np.nan_to_num(cat['center_offset'][i]))
    centre = float(np.clip(1 - co/FINGER_DEPTH, 0, 1))
    above = float(.5*(1 - np.dot(cat['approach'][i], up)))
    pre = float(cat['preliminary'][i])
    force = (pre*.9 - .15*width - .25*centre - .3*above)/.2
    return dict(width=width, force=force, centre=centre, above=above, preliminary=pre, centre_offset_mm=co*1e3)


def bootstrap_diff(a, b, seed=20261128, n=10000):
    """Paired difference of means a - b over parts, with a percentile 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    return dict(difference=round(float((a-b).mean()), 4),
                ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], n=int(len(a)))


def mcnemar(a, b):
    """Exact McNemar on paired booleans: counts where only a holds, only b holds, and the two-sided p."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(x, x+y, .5).pvalue if x+y else 1.
    return dict(only_a=x, only_b=y, p=round(float(p), 6))


def grouped_auc(X, y, groups, folds=5, seed=0):
    """Out-of-fold AUC of a logistic model, folds by part."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    X = np.asarray(X, float).reshape(len(y), -1)
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred)), pred


def failure_mode(trial):
    """'held', 'dropped' (no rise), 'pivoted' (rose but turned or slid) from a v1/v2 trial record."""
    if trial['held']:
        return 'held'
    rise = trial['rise'] if 'rise' in trial else trial[1]/1e3
    return 'dropped' if rise < .02 else 'pivoted or slid'
