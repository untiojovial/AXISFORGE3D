"""Phase V evaluation engine: every ordering, every test part, from catalogues.

Nothing here runs the planner: ``catalogue.order`` reproduces its examination
order exactly (checked against 6 788 recorded planner runs), and each part's
catalogue already holds every candidate's verdict.

Experiments (``library_for``):
  B  frozen library -> new parts; no test part is ever in the library
  A  only the 16 YCB parts as library -> new parts (cross-dataset)
  C  as B, but the library leaves out every part of the query's GSO category
"""
from __future__ import annotations

import numpy as np

from research_grasping.phase5 import baselines as bl
from research_grasping.phase5 import catalogue

RANDOM_ORDER_SEEDS = 20
RANDOM_NEIGHBOUR_SEEDS = 10
RANDOM_VITS = tuple(f'vit-random-s{s}' for s in range(5))
VISUAL_MODELS = ('dinov2', 'dinov3', 'dinov3-metric', 'd2', *RANDOM_VITS)
FUSION_MODELS = VISUAL_MODELS
AFAT_SOURCES = (('box', None), ('pooled', None), ('visual', 'dinov3'), ('visual', 'd2'))
AFAT_SIMILARITY = ('dinov3', 'd2')
# Ablation: the same stacker without anything from the image model.
NOVIS_SOURCES = (('box', None), ('pooled', None), ('visual', 'd2'))
NOVIS_SIMILARITY = ('d2',)


def library_for(experiment, query, store, library_names):
    if experiment == 'A':
        return [n for n in library_names if store[n]['source'] == 'ycb']
    if experiment == 'C':
        category = store[query]['category']
        return [n for n in library_names if store[n]['category'] != category]
    return list(library_names)


def entries(names, store, cats):
    return [bl.entry(n, cats[n], store[n]['low'], store[n]['high'], store[n]['descriptors']) for n in names]


def query_entry(name, store, cats):
    return bl.entry(name, cats[name], store[name]['low'], store[name]['high'], store[name]['descriptors'])


CHASE = 20


def positions(cat, prior=None, weight=2.0):
    """Examined until the 1st and 3rd feasible grasp, the first one's label, and the
    first CHASE feasible grasps as (label, candidates examined) for dynamic checks."""
    order = catalogue.order(cat, prior, weight)
    hits = np.flatnonzero(cat['feasible'][order])
    n = len(order)
    first = str(cat['label'][order[hits[0]]]) if len(hits) else None
    chase = [(str(cat['label'][order[h]]), int(h)+1) for h in hits[:CHASE]]
    return (int(hits[0])+1 if len(hits) else n, bool(len(hits)), int(hits[2])+1 if len(hits) > 2 else n,
            len(hits) > 2, first, chase)


def source_priors(query, library, cat, qsig=None, sources=AFAT_SOURCES):
    qsig = qsig if qsig is not None else bl.query_signature(cat, query['low'], query['high'])
    return [bl.transfer(m, query, library, cat, model=model, qsig=qsig) for m, model in sources]


def stacked_features(query, library, cat, low, high, qsig=None):
    """Features for AFAT and for its no-image ablation (sharing the common sources)."""
    full = source_priors(query, library, cat, qsig)
    by_source = dict(zip(AFAT_SOURCES, full))
    novis = [by_source[s] for s in NOVIS_SOURCES]
    return (bl.afat_features(cat, low, high, full, bl.similarity_features(query, library, AFAT_SIMILARITY)),
            bl.afat_features(cat, low, high, novis, bl.similarity_features(query, library, NOVIS_SIMILARITY)))


def afat_rows(name, library, store, cats):
    """Training rows for one library part, its sources computed without it."""
    q = query_entry(name, store, cats)
    others = [e for e in library if e['name'] != name]
    cat = cats[name]
    X, Xn = stacked_features(q, others, cat, store[name]['low'], store[name]['high'])
    return X, Xn, cat['feasible'].astype(int)


def train(library, store, cats, seed=0):
    """The learned candidate classifier and the AFAT stacker, on library parts only."""
    names = [e['name'] for e in library]
    X = np.vstack([bl.features(cats[n], store[n]['low'], store[n]['high']) for n in names])
    y = np.concatenate([cats[n]['feasible'].astype(int) for n in names])
    learned = bl.fit(X, y, seed)
    rows = [afat_rows(n, library, store, cats) for n in names]
    yy = np.concatenate([r[2] for r in rows])
    afat = bl.fit(np.vstack([r[0] for r in rows]), yy, seed)
    novis = bl.fit(np.vstack([r[1] for r in rows]), yy, seed)
    return learned, (afat, novis)


def evaluate_query(name, library, store, cats, learned, afat, chase=()):
    """Rows for every ordering of one held-out part."""
    cat = cats[name]
    q = query_entry(name, store, cats)
    low, high = store[name]['low'], store[name]['high']
    qs = bl.query_signature(cat, low, high)
    priors = [('default', None, None, 2.0)]
    priors += [('random-order', s, bl.random_order(cat, s), bl.RANDOM_WEIGHT) for s in range(RANDOM_ORDER_SEEDS)]
    priors += [(f'heuristic:{h}', None, bl.heuristic(h, cat, low, high), 2.0) for h in bl.HEURISTICS]
    X = bl.features(cat, low, high)
    priors.append(('learned', None, bl.scores(learned, X), 2.0))
    priors.append(('box', None, bl.transfer('box', q, library, cat, qsig=qs), 2.0))
    priors.append(('pooled', None, bl.transfer('pooled', q, library, cat, qsig=qs), 2.0))
    priors += [('random', s, bl.transfer('random', q, library, cat, seed=s, qsig=qs), 2.0)
               for s in range(RANDOM_NEIGHBOUR_SEEDS)]
    priors += [(f'visual:{m}', None, bl.transfer('visual', q, library, cat, model=m, qsig=qs), 2.0) for m in VISUAL_MODELS]
    priors += [(f'fusion:{m}', None, bl.transfer('fusion', q, library, cat, model=m, qsig=qs), 2.0) for m in FUSION_MODELS]
    Xa, Xn = stacked_features(q, library, cat, low, high, qs)
    priors.append(('afat', None, bl.scores(afat[0], Xa), 2.0))
    priors.append(('afat-novis', None, bl.scores(afat[1], Xn), 2.0))
    priors.append(('self', None, bl.transfer('self', q, [q], cat, qsig=qs), 2.0))
    rows = []
    for method, seed, prior, weight in priors:
        e1, f1, e3, f3, first, order = positions(cat, prior, weight)
        row = dict(part=name, method=method, seed=seed, candidates=int(len(cat['label'])),
                   feasible=int(cat['feasible'].sum()), examined1=e1, found1=f1, examined3=e3, found3=f3, first=first)
        if method in chase:
            row['chase'] = order
        rows.append(row)
    return rows
