"""Phase VII extension (development only): v3 labels for viable candidates of the 151 dynamic development parts.

    python tools/research/phase7_extension/v3_sweep.py --dev research/phase6/dev_parts.json --catalogues cat80/ \
        --collision COLLISION_DIR --projects POOL --store store.json \
        --physics research/phase7/results/physics_dev_with_joint.json.gz --out-dir v3sweep/ --cap 30 --jobs 3

Candidates per part: a seeded random sample of at most --cap viable candidates (seed 20261210 + part index)
plus the first viable grasps of M0, M1, M4, M9 and Joint (already simulated at torsion 0.005 in Phase VII;
re-simulated here so that every label comes from one run of this script). Trials per candidate (v3 criterion,
research_grasping/phase7/physics.py):

- standard six conditions at torsional friction 0.005, 0.001 and 0.02 m (scenarios, not calibrated values);
- the three loads (0.5, 1, 2 kg) at 0.005 m;
- extended scenarios at 0.005 m: tangential friction 0.7, centre of mass shifted by +/-5 mm along world x.

One JSON per part, written when the part is complete (restart-safe). No EGAD object is touched.
With --labels-json FILE ({part: [label, ...]}) exactly those candidates are simulated instead (used for the
first grasps of new development orders that the sample did not contain); write them to another --out-dir.
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
from research_grasping.phase5 import catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import physics as v2  # noqa: E402
from research_grasping.phase7 import physics as v3  # noqa: E402

STANDARD = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)),
            ('pose noise 1', dict(noise_seed=1)), ('pose noise 2', dict(noise_seed=2)))
LOADS = (('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))
EXTENDED = (('friction 0.7', dict(friction=.7)), ('com +5 mm x', dict(com_shift=(.005, 0., 0.))),
            ('com -5 mm x', dict(com_shift=(-.005, 0., 0.))))
PLAN = {'0.005': STANDARD + LOADS + EXTENDED, '0.001': STANDARD, '0.02': STANDARD}
STA = {}


def init(a):
    STA.update(vars(a))


def job(task):
    part, labels = task
    s = STA
    path = Path(s['out_dir'])/f'{part}.json'
    if path.exists():
        return part, 0.
    store = json.loads(Path(s['store']).read_text())
    R, t, g, tt = v1.part_pose(Path(s['projects'])/store[part]['project'], part)
    pieces = [v1.read_obj(p) for p in sorted((Path(s['collision'])/part).glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(s['catalogues'])/f'{part}.npz')
    com, _ = v2.planner_com(cat)
    cands = {c['label']: c for c in catalogue.candidates(cat)}
    start = time.perf_counter()
    out = {}
    for lab in labels:
        out[lab] = {}
        for tor, conds in PLAN.items():
            res = {}
            for name, kw in conds:
                kw = dict(kw)
                shift = kw.pop('com_shift', None)
                c = com if shift is None else np.asarray(com) + np.asarray(shift)
                res[name] = v3.trial(pieces, R, t, tt, cands[lab], g, com=c, torsion=float(tor), **kw)
            out[lab][tor] = res
    path.write_text(json.dumps(dict(seconds=round(time.perf_counter()-start, 1), trials=out), separators=(',', ':')))
    return part, time.perf_counter()-start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('dev', 'catalogues', 'collision', 'projects', 'store', 'physics', 'out-dir'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--cap', type=int, default=30)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--labels-json')
    p.add_argument('--reverse', action='store_true', help='process parts in reverse order (a second runner on the same out-dir)')
    a = p.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    dev = sorted(json.loads(Path(a.dev).read_text())['parts'])
    core = json.load(gzip.open(a.physics, 'rt'))['core']
    tasks, manifest = [], {}
    if a.labels_json:
        want = json.loads(Path(a.labels_json).read_text())
        tasks = [(n, sorted(set(ls))) for n, ls in want.items() if ls and n in set(dev)]
        (Path(a.out_dir)/'manifest.json').write_text(json.dumps(dict(labels_json=a.labels_json, plan={k: [c for c, _ in v] for k, v in PLAN.items()},
                                                                     parts={n: ls for n, ls in tasks}), indent=1))
        dev = []
    for i, n in enumerate(dev):
        cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
        viable = [str(x) for x in cat['label'][cat['feasible']]]
        rng = np.random.default_rng(20261210 + i)
        sample = [viable[j] for j in sorted(rng.permutation(len(viable))[:a.cap])]
        labels = sorted(set(sample) | set(core[n].values()))
        manifest[n] = dict(viable=len(viable), sampled=sample, core=core[n], simulated=labels)
        tasks.append((n, labels))
    if not a.labels_json and not a.reverse:
        (Path(a.out_dir)/'manifest.json').write_text(json.dumps(dict(cap=a.cap, seed_base=20261210,
                                                                     plan={k: [c for c, _ in v] for k, v in PLAN.items()}, parts=manifest), indent=1))
    tasks.sort(key=lambda t: -len(t[1]))
    if a.reverse:
        tasks.reverse()
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for part, sec in pool.map(job, tasks):
            print(part, round(sec), flush=True)


if __name__ == '__main__':
    main()
