"""v3 trials (nine conditions) for given candidates of development parts or EGAD objects.

    python tools/research/phase7_extension/extra_physics.py --labels needed.json --kind development \
        --catalogues cat80/ --store store.json --projects POOL --collision COLLISION_DIR --out-dir out/ --jobs 2
    python tools/research/phase7_extension/extra_physics.py --labels needed.json --kind egad \
        --catalogues CONF/cat80 --store CONF/store.json --projects EGAD_POOL --out-dir out/ --jobs 2

--labels is a JSON {object: [label, ...]}. Conditions and the criterion are those of the Phase VII
confirmation (tools/research/phase7/egad_confirm.py CONDITIONS, research_grasping/phase7/physics.py v3),
at the torsions given by --torsions (default 0.005). One JSON per object, written when complete.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import physics as v2  # noqa: E402
from research_grasping.phase7 import physics as v3  # noqa: E402

CONDITIONS = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)),
              ('pose noise 1', dict(noise_seed=1)), ('pose noise 2', dict(noise_seed=2)),
              ('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))
STA = {}


def init(a):
    STA.update(vars(a))


def job(task):
    name, labels = task
    s = STA
    path = Path(s['out_dir'])/f'{name}.json'
    if path.exists():
        return name, 0.
    store = json.loads(Path(s['store']).read_text())
    R, t, g, tt = v1.part_pose(Path(s['projects'])/store[name]['project'], name)
    coll = Path(s['projects'])/name/'collision' if s['kind'] == 'egad' else Path(s['collision'])/name
    pieces = [v1.read_obj(p) for p in sorted(coll.glob('model_collision_*.obj'), key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(s['catalogues'])/f'{name}.npz')
    com, _ = v2.planner_com(cat)
    cands = {c['label']: c for c in catalogue.candidates(cat)}
    start = time.perf_counter()
    res = {}
    for label in labels:
        res[label] = {tor: {c: v3.trial(pieces, R, t, tt, cands[label], g, com=com, torsion=float(tor), **kw) for c, kw in CONDITIONS}
                      for tor in s['torsions']}
    path.write_text(json.dumps(dict(seconds=round(time.perf_counter()-start, 1), trials=res), separators=(',', ':')))
    return name, time.perf_counter()-start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('labels', 'catalogues', 'store', 'projects', 'out-dir'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--kind', choices=('development', 'egad'), required=True)
    p.add_argument('--collision'); p.add_argument('--torsions', nargs='+', default=['0.005'])
    p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args()
    if a.kind == 'development' and not a.collision:
        raise SystemExit('--collision is required for development parts')
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    want = json.loads(Path(a.labels).read_text())
    tasks = sorted(((n, sorted(set(ls))) for n, ls in want.items() if ls), key=lambda t: -len(t[1]))
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for n, sec in pool.map(job, tasks):
            print(n, round(sec, 1), flush=True)


if __name__ == '__main__':
    main()
