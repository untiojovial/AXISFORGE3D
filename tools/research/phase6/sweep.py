"""Dynamic outcome of every viable candidate of the development parts (v2 physics).

    python tools/research/phase6/sweep.py --dev research/phase6/dev_parts.json --catalogues cat80/ \
        --collision COLLISION_DIR --projects POOL --store store.json --out sweep.json.gz --jobs 4

Every candidate the exhaustive planner accepted (80 mm gripper) is tried under
the six standard conditions with the intended masses (0.1 kg nominal, 0.2 and
0.05 kg) and under three heavy loads (0.5, 1 and 2 kg; Phase V's 'mass x0.5'
and 'mass x2' were in fact 0.5 and 2 kg). With this table, any ordering's
first viable grasp can be looked up instead of simulated again.

Output: {part: {label: {condition: [held, rise_mm, slip_mm, turn_deg, touching]}}}.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import physics  # noqa: E402

CONDITIONS = (
    ('nominal', dict()),
    ('mass x2', dict(mass=.2)),
    ('mass x0.5', dict(mass=.05)),
    ('friction 0.3', dict(friction=.3)),
    ('pose noise 1', dict(noise_seed=1)),
    ('pose noise 2', dict(noise_seed=2)),
    ('load 0.5 kg', dict(mass=.5)),
    ('load 1 kg', dict(mass=1.)),
    ('load 2 kg', dict(mass=2.)),
)
STANDARD = tuple(n for n, _ in CONDITIONS[:6])
STA = {}


def init(a):
    STA.update(vars(a))


def job(part):
    s = STA
    store = json.loads(Path(s['store']).read_text())
    proj = Path(s['projects'])/store[part]['project']
    R, t, g, tt = v1.part_pose(proj, part)
    pieces = [v1.read_obj(p) for p in sorted((Path(s['collision'])/part).glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(s['catalogues'])/f'{part}.npz')
    com, residual = physics.planner_com(cat)
    out = {}
    start = time.perf_counter()
    for c in catalogue.candidates(cat):
        i = list(cat['label']).index(c['label'])
        if not cat['feasible'][i]:
            continue
        res = {}
        for name, kw in CONDITIONS:
            r = physics.trial(pieces, R, t, tt, c, g, com=com, **kw)
            res[name] = [int(r['held']), round(r.get('rise', 0)*1e3, 1), round(r.get('slip', 0)*1e3, 1),
                         r.get('turn_deg'), int(r.get('touching', False))]
        out[c['label']] = res
    return part, out, round(time.perf_counter()-start, 1), residual


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('dev', 'catalogues', 'collision', 'projects', 'store', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    dev = json.loads(Path(a.dev).read_text())['parts']
    parts = sorted(dev, key=lambda n: -dev[n]['feasible'])  # longest first
    start = time.perf_counter()
    table, secs, resid = {}, {}, {}
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for part, out, s, r in pool.map(job, parts):
            table[part], secs[part], resid[part] = out, s, r
            print(part, len(out), s, flush=True)
    doc = dict(model='phase6 v2: research_grasping/phase6/physics.py (planner centre of mass)',
               conditions={n: kw for n, kw in CONDITIONS}, standard=list(STANDARD),
               fields=['held', 'rise_mm', 'slip_mm', 'turn_deg', 'touching'],
               wall_s=round(time.perf_counter()-start, 1), seconds=secs, com_fit_residual_m=resid,
               parts=dict(sorted(table.items())))
    with gzip.open(a.out, 'wt') as f:
        json.dump(doc, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
