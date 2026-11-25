"""Labelled candidate catalogues and an exact simulator of the planner's order.

A catalogue holds, for every grasp candidate the native planner generates for
one part, its geometry, the planner's own ranking inputs (preliminary score and
the stroke/grip gate) and the verdict of an exhaustive plan (feasible, or the
first reason it was refused). With it, the number of candidates the planner
examines under any prior can be computed without running the planner:

    score = preliminary + weight * prior   (then + 10 if makeable)

sorted by decreasing score, ties kept in generator order (Rust's stable
``sort_by``). A candidate's verdict does not depend on the order it is
examined in; ``validate`` checks this against real planner runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

FIELDS = ('center', 'approach', 'closing', 'normalA', 'normalB')


def _vec(v):
    return [v['x'], v['y'], v['z']]


def axisforge(binary, *args, timeout=1800):
    done = subprocess.run([str(binary), *map(str, args)], check=True, capture_output=True, text=True,
                          timeout=timeout)
    return json.loads(done.stdout)


def build(binary, project, body):
    """Exhaustive verdicts for every candidate of one part, in generator order."""
    start = time.perf_counter()
    records = axisforge(binary, 'research-candidates', project, body)
    generate_s = time.perf_counter()-start
    start = time.perf_counter()
    report = axisforge(binary, 'plan', '--json', '--nr-grasps', 1_000_000, project, body)
    solve_s = time.perf_counter()-start
    labels = [r['candidate']['label'] for r in records]
    if len(set(labels)) != len(labels):
        raise ValueError('candidate labels are not unique')
    verdict = {p['candidate']['label']: 'feasible' for p in report['plans']}
    for r in report['rejections']:
        verdict[r['label']] = r['reason']
    if set(verdict) != set(labels) or report['candidatesExamined'] != len(labels):
        raise ValueError('exhaustive plan does not cover the catalogue')
    pose = np.array([[*_vec(r['graspPose']['translation']), *[r['graspPose']['rotation'][k] for k in 'xyzw']]
                     for r in records], float)
    arrays = dict(
        label=np.array(labels), preliminary=np.array([r['preliminaryScore'] for r in records], float),
        makeable=np.array([r['makeable'] for r in records], bool),
        feasible=np.array([verdict[l] == 'feasible' for l in labels], bool),
        reason=np.array([verdict[l] for l in labels]),
        width=np.array([r['candidate']['width'] for r in records], float),
        center_offset=np.array([r['candidate'].get('centerOffset', np.nan) for r in records], float),
        grasp_pose=pose,
        **{f: np.array([_vec(r['candidate'][f]) for r in records], float) for f in FIELDS})
    meta = dict(body=body, project_sha256=hashlib.sha256(Path(project).read_bytes()).hexdigest(),
                candidates=len(labels), feasible=int(arrays['feasible'].sum()),
                generate_s=round(generate_s, 3), solve_s=round(solve_s, 3))
    return arrays, meta


def save(path, arrays, meta):
    np.savez_compressed(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files if k != 'meta'}
        out['meta'] = json.loads(str(f['meta']))
    return out


def candidates(cat):
    """The catalogue's candidates as the dictionaries ``visual_prior.prior`` reads."""
    xyz = lambda a: dict(x=float(a[0]), y=float(a[1]), z=float(a[2]))
    return [dict(label=str(l), center=xyz(c), approach=xyz(a), closing=xyz(k), width=float(w))
            for l, c, a, k, w in zip(cat['label'], cat['center'], cat['approach'], cat['closing'], cat['width'])]


def order(cat, prior=None, weight=2.0):
    """Indices in the order the planner examines them.

    ``prior`` maps labels to scores in [0, 1] (as the --prior file), or is an
    array aligned with the catalogue. The arithmetic follows the planner's:
    preliminary, plus weight times prior, plus 10 when makeable.
    """
    score = cat['preliminary'].astype(float).copy()
    if prior is not None:
        if isinstance(prior, dict):
            p = np.array([prior.get(str(l), np.nan) for l in cat['label']], float)
            has = ~np.isnan(p)
            score[has] = score[has] + weight*p[has]
        else:
            score = score + weight*np.asarray(prior, float)
    score = np.where(cat['makeable'], score+10.0, score)
    # Python's sort is stable, like Rust's sort_by: equal scores keep generator order.
    return np.array(sorted(range(len(score)), key=lambda i: -score[i]), int)


def examined(cat, prior=None, weight=2.0, target=1):
    """Candidates examined until ``target`` feasible ones, and whether they were found."""
    hits = np.flatnonzero(cat['feasible'][order(cat, prior, weight)])
    if len(hits) < target:
        return len(cat['label']), False
    return int(hits[target-1])+1, True


def prior_doc_scores(doc):
    """{label: score} from a --prior document."""
    return {c['label']: float(c['score']) for c in doc['candidates']}
