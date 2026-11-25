"""Priors for Phase V: geometric heuristics, random order, a learned candidate
classifier, a metric 3D shape descriptor, and adaptive transfer (AFAT).

Every prior is an array aligned with a part's catalogue, with values in
[0, 1]; ``catalogue.order`` turns it into the planner's examination order.
"""
from __future__ import annotations

import numpy as np

from research_grasping.phase4 import visual_prior as vp

MAX_OPENING = {40: .04, 80: .08, 120: .12}


def _unit(x):
    x = np.asarray(x, float)
    span = x.max()-x.min() if len(x) else 0.
    return (x-x.min())/span if span > 0 else np.zeros_like(x)


# --- geometric heuristics: no library, no learning -------------------------------------------

def heuristic(name, cat, low, high):
    """com: grasp centre near the box centre; narrow: smallest opening first;
    antipodal: contact normals along the closing axis; topdown: approach
    pointing down."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    if name == 'com':
        return 1-_unit(np.linalg.norm(cat['center']-(low+high)/2, axis=1))
    if name == 'narrow':
        return 1-_unit(cat['width'])
    if name == 'antipodal':
        k = cat['closing']/np.linalg.norm(cat['closing'], axis=1, keepdims=True)
        return (np.abs((cat['normalA']*k).sum(1))+np.abs((cat['normalB']*k).sum(1)))/2
    if name == 'topdown':
        return (1-cat['approach'][:, 2]/np.linalg.norm(cat['approach'], axis=1))/2
    raise ValueError(name)


HEURISTICS = ('com', 'narrow', 'antipodal', 'topdown')

# Large enough that the planner's own estimate (at most 1) only breaks exact ties:
# a uniformly random order within the stroke gate.
RANDOM_WEIGHT = 1e6


def random_order(cat, seed):
    return np.random.default_rng(seed).random(len(cat['label']))


# --- metric 3D shape descriptor ----------------------------------------------------------------

D2_BINS = np.linspace(0, .40, 33)


def d2_descriptor(vertices, faces, points=2048, pairs=200_000, seed=0):
    """Osada et al.'s D2 shape distribution in metres (not normalised for size),
    as a unit vector of square-rooted bin frequencies (cosine = Bhattacharyya)."""
    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    pick = rng.choice(len(tri), size=points, p=area/area.sum())
    u, v = rng.random(points), rng.random(points)
    flip = u+v > 1
    u[flip], v[flip] = 1-u[flip], 1-v[flip]
    p = tri[pick, 0]+u[:, None]*(tri[pick, 1]-tri[pick, 0])+v[:, None]*(tri[pick, 2]-tri[pick, 0])
    i, j = rng.integers(0, points, size=(2, pairs))
    d = np.linalg.norm(p[i]-p[j], axis=1)
    h, _ = np.histogram(np.clip(d, 0, D2_BINS[-1]-1e-9), bins=D2_BINS)
    h = np.sqrt(h/h.sum())
    return h/np.linalg.norm(h)


# --- library entries straight from catalogues --------------------------------------------------

def entry(name, cat, low, high, descriptors, project=None):
    """A visual_prior library entry whose good grasps are the catalogue's feasible ones."""
    good = [c for c, f in zip(_candidates(cat), cat['feasible']) if f]
    e = dict(name=name, project=project, body=name, low=list(map(float, low)), high=list(map(float, high)),
             candidates_total=int(len(cat['label'])), feasible=int(cat['feasible'].sum()), good=good,
             descriptors={k: np.asarray(v, float).tolist() for k, v in descriptors.items()})
    # Signatures of the good grasps, computed once (visual_prior.prior recomputes them per call).
    e['_sig'] = vp.signatures(good, e['low'], e['high']) if good else None
    return e


def _candidates(cat):
    from research_grasping.phase5 import catalogue
    return catalogue.candidates(cat)


def query_signature(cat, low, high):
    return vp.signatures(_candidates(cat), low, high)


def transfer(method, query, library, cat, model=None, seed=0, qsig=None):
    """Transfer prior (array aligned with the catalogue) from library neighbours.

    Same arithmetic as visual_prior.prior (weighted mean of each neighbour's best
    affinity, divided by its maximum, rounded to 6 decimals), with cached signatures.
    """
    chosen = vp.neighbours(method, query, library, seed=seed, model=model)
    if not chosen:
        return np.zeros(len(cat['label']))
    if qsig is None:
        qsig = query_signature(cat, query['low'], query['high'])
    total = np.zeros(len(cat['label'])); norm = 0.
    for entry, w in chosen:
        sig = entry.get('_sig') or vp.signatures(entry['good'], entry['low'], entry['high'])
        total += w*vp.affinity(qsig, sig).max(1)
        norm += w
    score = total/norm
    if score.max() > 0:
        score = score/score.max()
    return np.array([round(float(x), 6) for x in score])


# --- learned candidate classifier --------------------------------------------------------------

def features(cat, low, high):
    """Per-candidate features available before any evaluation."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    extent = np.maximum(high-low, 1e-6)
    a = cat['approach']/np.linalg.norm(cat['approach'], axis=1, keepdims=True)
    k = cat['closing']/np.linalg.norm(cat['closing'], axis=1, keepdims=True)
    rel = (cat['center']-low)/extent
    n = len(a)
    return np.column_stack([
        a, np.abs(k), cat['width'], rel, cat['center'][:, 2]-low[2],
        np.tile(np.log(extent), (n, 1)),
        np.abs((cat['normalA']*k).sum(1)), np.abs((cat['normalB']*k).sum(1)),
        np.nan_to_num(cat['center_offset']), cat['preliminary'], cat['makeable'].astype(float)])


FEATURE_NAMES = ['approach_x', 'approach_y', 'approach_z', 'closing_x', 'closing_y', 'closing_z', 'width',
                 'rel_x', 'rel_y', 'rel_z', 'height', 'log_extent_x', 'log_extent_y', 'log_extent_z',
                 'antipodal_a', 'antipodal_b', 'centre_offset', 'preliminary', 'makeable']


def classifier(seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=.1, max_leaf_nodes=31,
                                          l2_regularization=1., random_state=seed)


def fit(X, y, seed=0):
    model = classifier(seed)
    model.fit(X, y)
    return model


def scores(model, X):
    """Probabilities as a prior: divided by the part's maximum, so in [0, 1]."""
    p = model.predict_proba(X)[:, 1]
    return p/p.max() if p.max() > 0 else p


# --- adaptive transfer -------------------------------------------------------------------------

def similarity_features(query, library, models):
    """How close the part is to what the library knows, per source: these let the
    stacked model trust a source only where it is informative."""
    pool = [e for e in library if e['name'] != query['name'] and e['feasible'] > 0]
    out = []
    size = lambda e: np.log(np.maximum(np.subtract(e['high'], e['low']), 1e-6))
    d = np.sort([np.linalg.norm(size(e)-size(query)) for e in pool])
    out += [d[0], d[:3].mean()]
    for m in models:
        q = np.asarray(query['descriptors'][m], float)
        sim = np.sort([float(np.dot(q, e['descriptors'][m])) for e in pool])[::-1]
        out += [sim[0], sim[:3].mean()]
    return np.array(out, float)


def afat_features(cat, low, high, sources, similarity):
    """Candidate features + every source's transfer score + the part's similarity profile."""
    base = features(cat, low, high)
    n = len(base)
    return np.column_stack([base, np.column_stack(sources) if sources else np.zeros((n, 0)),
                            np.tile(similarity, (n, 1))])
