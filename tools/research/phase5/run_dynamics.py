"""Dynamic checks (MuJoCo) of the grasps each ordering finds first.

    python tools/research/phase5/run_dynamics.py --rows test.json --catalogues cat80/ \
        --collision COLLISION_DIR --projects POOL --store store.json --out dynamics.json --jobs 4

For every viable test part (experiment B, 80 mm) and every method with a
kept 'chase' list: the first feasible grasp is tried under all six
perturbations; then grasps are tried in that method's order, nominal only,
until one holds (at most 20). Results per grasp are cached and shared
between methods.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue, dynamics  # noqa: E402

STATE = {}


def init(collision, projects, cats_dir, store):
    STATE.update(collision=Path(collision), projects=Path(projects), cats=Path(cats_dir), store=store)


def part_job(task):
    name, methods = task
    st = STATE
    pieces = [dynamics.read_obj(p) for p in sorted((st['collision']/name).glob('model_collision_*.obj'),
                                                    key=lambda p: int(p.stem.split('_')[-1]))]
    if not pieces:
        return name, dict(error='no collision pieces')
    project = st['projects']/st['store'][name]['project']
    R, t, gripper, table_top = dynamics.part_pose(project, name)
    cat = catalogue.load(st['cats']/f'{name}.npz')
    cands = {c['label']: c for c in catalogue.candidates(cat)}
    cache = {}

    def run(label, perturbation='nominal'):
        key = (label, perturbation)
        if key not in cache:
            kw = dict(dynamics.PERTURBATIONS)[perturbation]
            start = time.perf_counter()
            r = dynamics.trial(pieces, R, t, table_top, cands[label], gripper, **kw)
            r['seconds'] = round(time.perf_counter()-start, 3)
            cache[key] = r
        return cache[key]

    out = {}
    for method, chase in methods.items():
        if not chase:
            out[method] = dict(first=None)
            continue
        first, _ = chase[0]
        trials = {p: run(first, p) for p, _ in dynamics.PERTURBATIONS}
        held_at = None
        for label, examined in chase:
            if run(label)['held']:
                held_at = dict(label=label, examined=examined)
                break
        out[method] = dict(first=first, first_trials=trials, first_held=trials['nominal']['held'],
                           first_robust=all(v['held'] for v in trials.values()),
                           first_held_count=sum(v['held'] for v in trials.values()), first_dynamic=held_at,
                           chased=len(chase))
    return name, dict(methods=out, pieces=len(pieces), sims=len(cache),
                      sim_seconds=round(sum(v['seconds'] for v in cache.values()), 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rows', required=True); p.add_argument('--catalogues', required=True)
    p.add_argument('--collision', required=True); p.add_argument('--projects', required=True)
    p.add_argument('--store', required=True); p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    rows = json.loads(Path(a.rows).read_text())['rows']
    store = json.loads(Path(a.store).read_text())
    tasks = {}
    for r in rows:
        if r['experiment'] == 'B' and r['gripper'] == 80 and 'chase' in r and r['feasible'] > 0:
            tasks.setdefault(r['part'], {})[r['method']] = r['chase']
    start = time.perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init,
                             initargs=(a.collision, a.projects, a.catalogues, store)) as pool:
        for name, res in pool.map(part_job, sorted(tasks.items())):
            results[name] = res
            print(name, res.get('sims'), res.get('sim_seconds'), flush=True)
    Path(a.out).write_text(json.dumps(dict(wall_s=round(time.perf_counter()-start, 1),
                                           schedule=dynamics.SCHEDULE,
                                           perturbations=[n for n, _ in dynamics.PERTURBATIONS],
                                           parts=results), indent=1) + '\n')


if __name__ == '__main__':
    main()
