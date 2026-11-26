"""Phase VII development physics: v3 outcomes of first grasps, and torsional-friction sensitivity.

    python tools/research/phase7/physics_dev.py --dev research/phase6/dev_parts.json --catalogues cat80/ \
        --collision COLLISION_DIR --projects POOL --store store.json --priors priors.npz \
        --sweep research/phase6/results/sweep.json.gz --first research/phase6/... first_grasps.json \
        --missing research/phase6/results/missing_first_grasps.json --out physics_dev.json.gz --jobs 2

1. The first viable grasp of M0, M1, M4 (alpha 0.8, centring) and M9 (p * P(H|F), gradient boosting
   retrained exactly as in Phase VI's conditional.py, out of fold) for every development part.
2. v3 trials (research_grasping/phase7/physics.py) in the nine conditions for every grasp in
   first_grasps.json + missing_first_grasps.json at torsion 0.005, and for the four core methods'
   first grasps also at torsion 0.001, 0.0025 and 0.02.
Output: {core: {part: {method: label}}, trials: {part: {label: {torsion: {condition: record}}}}}.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import baselines as bl, catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import fusion, physics as v2  # noqa: E402
from research_grasping.phase7 import physics as v3  # noqa: E402

CONDITIONS = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)),
              ('pose noise 1', dict(noise_seed=1)), ('pose noise 2', dict(noise_seed=2)),
              ('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))
TORSIONS = (.001, .0025, .005, .02)
STA = {}


def core_first(dev, cats, store, pri, sw):
    """First viable labels of M0, M1, M4, M9 (M9 with Phase VI's out-of-fold P(H|F))."""
    std = sw['standard']
    order = np.random.default_rng(20261130).permutation(len(dev))
    folds = [order[f::5] for f in range(5)]
    X = {n: bl.features(cats[n], store[n]['low'], store[n]['high']) for n in dev}
    first = lambda cat, o: str(cat['label'][o[np.flatnonzero(cat['feasible'][o])[0]]])
    out = {n: {} for n in dev}
    for test in folds:
        train = [dev[i] for i in np.setdiff1d(np.arange(len(dev)), test)]
        Xt, yt = [], []
        for n in train:
            idx = np.flatnonzero(cats[n]['feasible'])
            Xt.append(X[n][idx])
            yt.append([all(c in sw['parts'][n][str(cats[n]['label'][i])] and sw['parts'][n][str(cats[n]['label'][i])][c][0]
                           for c in std) for i in idx])
        model = bl.fit(np.vstack(Xt), np.concatenate(yt).astype(int), 0)
        for i in test:
            n = dev[i]
            h = model.predict_proba(X[n])[:, 1]
            out[n]['M9'] = (first(cats[n], fusion.order(cats[n], pri[n]*h)))
    for n in dev:
        cat = cats[n]
        out[n]['M0'] = first(cat, catalogue.order(cat))
        out[n]['M1'] = first(cat, catalogue.order(cat, pri[n], 2.))
        out[n]['M4'] = first(cat, fusion.order(cat, fusion.config_score('M4', dict(alpha=.8, q='cen'), cat, pri[n])))
    return out


def init(a):
    STA.update(vars(a))


def job(task):
    part, jobs = task
    s = STA
    store = json.loads(Path(s['store']).read_text())
    R, t, g, tt = v1.part_pose(Path(s['projects'])/store[part]['project'], part)
    pieces = [v1.read_obj(p) for p in sorted((Path(s['collision'])/part).glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(s['catalogues'])/f'{part}.npz')
    com, _ = v2.planner_com(cat)
    cands = {c['label']: c for c in catalogue.candidates(cat)}
    out = {}
    start = time.perf_counter()
    for label, torsions in jobs:
        out[label] = {}
        for tor in torsions:
            out[label][str(tor)] = {name: v3.trial(pieces, R, t, tt, cands[label], g, com=com, torsion=tor, **kw)
                                    for name, kw in CONDITIONS}
    return part, out, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('dev', 'catalogues', 'collision', 'projects', 'store', 'priors', 'sweep', 'first', 'missing', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args()
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    store = json.loads(Path(a.store).read_text())
    cats = {n: catalogue.load(Path(a.catalogues)/f'{n}.npz') for n in dev}
    pz = np.load(a.priors)
    pri = {n: pz[f'{n}|learned'].astype(float)/pz[f'{n}|learned'].max() for n in dev}
    sw = json.load(gzip.open(a.sweep, 'rt'))
    core = core_first(dev, cats, store, pri, sw)
    labels = json.loads(Path(a.first).read_text())
    miss = json.loads(Path(a.missing).read_text())
    tasks = []
    for n in dev:
        want = {lab: {.005} for lab in set(labels.get(n, [])) | set(miss.get(n, []))}
        for lab in core[n].values():
            want.setdefault(lab, set()).update(TORSIONS)
        tasks.append((n, sorted((lab, sorted(ts)) for lab, ts in want.items())))
    tasks.sort(key=lambda t: -sum(len(ts) for _, ts in t[1]))
    trials, secs = {}, {}
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for part, out, s in pool.map(job, tasks):
            trials[part], secs[part] = out, round(s, 1)
            print(part, len(out), round(s), flush=True)
    doc = dict(model='v3: research_grasping/phase7/physics.py', turn_limit_deg=v3.TURN_LIMIT_DEG,
               conditions={n: kw for n, kw in CONDITIONS}, torsions=list(TORSIONS),
               standard=[n for n, _ in CONDITIONS[:6]], loads=[n for n, _ in CONDITIONS[6:]],
               core=core, seconds=secs, trials=trials)
    with gzip.open(a.out, 'wt') as f:
        json.dump(doc, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
