"""Stage C: simple fusions of a feasibility prior p and a quality term Q.

Every fusion returns a score per candidate (higher is examined first). The
planner receives it as the prior (s + 2*makeable)/3 with weight 1e6, so the
stroke/force gate stays first and the preliminary score only breaks ties
(see PHASE6_PROTOCOL.md, section 4).
"""
from __future__ import annotations

import numpy as np

from research_grasping.phase5 import catalogue

FINGER_DEPTH = .045
WEIGHT = 1e6
Q_FLOOR = 1e-3


def minmax(x):
    x = np.asarray(x, float)
    span = np.ptp(x)
    return (x - x.min())/span if span > 0 else np.zeros_like(x)


def quality(cat, kind):
    """Per-part normalised quality: 'pre' (preliminary score) or 'cen' (centring term)."""
    if kind == 'pre':
        return minmax(cat['preliminary'])
    if kind == 'cen':
        co = np.nan_to_num(np.asarray(cat['center_offset'], float))
        return minmax(np.clip(1 - co/FINGER_DEPTH, 0, 1))
    raise ValueError(kind)


def pareto_fronts(p, q):
    """Front index (0 = non-dominated) of each point when maximising both p and q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    # dominated[i, j]: j dominates i
    dominated = (p[None, :] >= p[:, None]) & (q[None, :] >= q[:, None]) & \
                ((p[None, :] > p[:, None]) | (q[None, :] > q[:, None]))
    fronts = np.full(len(p), -1)
    left = np.ones(len(p), bool)
    k = 0
    while left.any():
        front = left & ~(dominated & left[None, :]).any(1)
        fronts[front] = k
        left &= ~front
        k += 1
    return fronts


def score(method, p, q, alpha=None, theta=None, tau=None):
    """Fused score for one part. p and q in [0, 1]."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if method == 'M3':
        return alpha*p + (1-alpha)*q
    if method == 'M4':
        return p**alpha * np.maximum(q, Q_FLOOR)**(1-alpha)
    if method == 'M5':
        return np.where(p >= theta, 2 + q, p)
    if method == 'M6':
        return np.where(q >= tau, 2 + p, p)
    if method == 'M7':
        from scipy.stats import rankdata
        n = len(p)
        return alpha*rankdata(p)/n + (1-alpha)*rankdata(q)/n
    if method == 'M8':
        f = pareto_fronts(p, q)
        return (f.max() - f) + .999*p
    raise ValueError(method)


def planner_prior(cat, s):
    """The prior handed to the planner: normalised score with the gate folded in."""
    return (minmax(s) + 2*np.asarray(cat['makeable'], float))/3


def order(cat, s):
    return catalogue.order(cat, planner_prior(cat, s), WEIGHT)


PLANNER_MAX_WEIGHT = 5.0  # the native planner accepts priorWeight in [0, 5]


def encoded_prior(cat, examined_order):
    """A prior that makes the native planner (weight 5) examine candidates in ``examined_order``.

    The planner ranks by preliminary + 5*prior (+10 if makeable). Giving the k-th
    candidate of the order the target T_k = 1 + 4*(n-1-k)/(n-1), strictly
    decreasing in [1, 5], and prior = (T_k - preliminary)/5 in [0, 1] reproduces
    the order exactly, because the order already puts every makeable candidate first.
    """
    n = len(examined_order)
    pre = np.asarray(cat['preliminary'], float)
    target = 1 + 4*np.arange(n-1, -1, -1)/max(n-1, 1)
    prior = np.empty(n)
    prior[np.asarray(examined_order)] = (target - pre[np.asarray(examined_order)])/PLANNER_MAX_WEIGHT
    return prior


GRID = {
    'M3': [dict(alpha=a/10, q=k) for k in ('pre', 'cen') for a in range(11)],
    'M4': [dict(alpha=a/10, q=k) for k in ('pre', 'cen') for a in range(11)],
    'M5': [dict(theta=t, q=k) for k in ('pre', 'cen') for t in (.3, .5, .7, .9)],
    'M6': [dict(tau=t, q=k) for k in ('pre', 'cen') for t in (.3, .5, .7, .9)],
    'M7': [dict(alpha=a/10, q=k) for k in ('pre', 'cen') for a in range(1, 10)],
    'M8': [dict(q=k) for k in ('pre', 'cen')],
}


def config_score(method, cfg, cat, p):
    q = quality(cat, cfg['q'])
    return score(method, p, q, **{k: v for k, v in cfg.items() if k != 'q'})
