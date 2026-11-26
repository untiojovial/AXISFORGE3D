"""Dynamic outcomes of the development parts' viable candidates (v2 physics).

    # pass 1: every viable candidate, the six standard conditions, stopping at the first failure
    python tools/research/phase6/sweep.py robust --dev research/phase6/dev_parts.json --catalogues cat80/ \
        --collision COLLISION_DIR --projects POOL --store store.json --cache DIR --jobs 4
    # pass 2: all nine conditions for the given grasps (the first viable grasp of every ordering)
    python tools/research/phase6/sweep.py full --labels first_grasps.json ... --cache DIR
    # merge both into one table
    python tools/research/phase6/sweep.py merge --dev ... --cache DIR --out sweep.json.gz

Standard conditions use the intended masses (0.1 kg nominal, 0.2 and 0.05 kg);
the heavy loads (0.5, 1 and 2 kg) are the masses Phase V really applied plus
1 kg. A candidate is robust when it holds in all six standard conditions, so
pass 1 stops at its first failure (order below, most frequent failures first):
the conditions it did not run are absent, never guessed. One JSON per part in
the cache, so a run can resume.

Merged output: {part: {label: {condition: [held, rise_mm, slip_mm, turn_deg, touching]}}}.
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
EARLY_EXIT_ORDER = ('nominal', 'pose noise 2', 'friction 0.3', 'mass x2', 'pose noise 1', 'mass x0.5')
STA = {}


def init(a):
    STA.update(vars(a))


def record(r):
    return [int(r['held']), round(r.get('rise', 0)*1e3, 1), round(r.get('slip', 0)*1e3, 1),
            r.get('turn_deg'), int(r.get('touching', False))]


def job(task):
    part, labels = task
    s = STA
    path = Path(s['cache'])/s['mode']/f'{part}.json'
    if path.exists():
        return part, 'cached'
    store = json.loads(Path(s['store']).read_text())
    R, t, g, tt = v1.part_pose(Path(s['projects'])/store[part]['project'], part)
    pieces = [v1.read_obj(p) for p in sorted((Path(s['collision'])/part).glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(s['catalogues'])/f'{part}.npz')
    com, residual = physics.planner_com(cat)
    cands = {c['label']: c for c in catalogue.candidates(cat)}
    kw = dict(CONDITIONS)
    start = time.perf_counter()
    out = {}
    for label in labels:
        res = {}
        names = EARLY_EXIT_ORDER if s['mode'] == 'robust' else [n for n, _ in CONDITIONS]
        for name in names:
            res[name] = record(physics.trial(pieces, R, t, tt, cands[label], g, com=com, **kw[name]))
            if s['mode'] == 'robust' and not res[name][0]:
                break
        out[label] = res
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(seconds=round(time.perf_counter()-start, 1), com_fit_residual_m=residual,
                                    grasps=out), separators=(',', ':')))
    return part, len(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('robust', 'full', 'merge'))
    for k in ('dev', 'cache'):
        p.add_argument(f'--{k}', required=True)
    for k in ('catalogues', 'collision', 'projects', 'store', 'labels', 'out'):
        p.add_argument(f'--{k}')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--supplement', action='append', default=[], help='extra cache dirs merged into the full pass')
    a = p.parse_args()
    dev = json.loads(Path(a.dev).read_text())['parts']
    if a.mode == 'merge':
        table, secs = {}, {}
        for part in sorted(dev):
            r = json.loads((Path(a.cache)/'robust'/f'{part}.json').read_text())
            f = json.loads((Path(a.cache)/'full'/f'{part}.json').read_text())
            secs[part] = r['seconds'] + f['seconds']
            table[part] = r['grasps']
            for label, res in f['grasps'].items():
                table[part][label] = {**table[part].get(label, {}), **res}
            for extra in a.supplement:
                path = Path(extra)/'full'/f'{part}.json'
                if path.exists():
                    s = json.loads(path.read_text())
                    secs[part] += s['seconds']
                    for label, res in s['grasps'].items():
                        table[part][label] = {**table[part].get(label, {}), **res}
        doc = dict(model='phase6 v2: research_grasping/phase6/physics.py (planner centre of mass)',
                   conditions={n: kw for n, kw in CONDITIONS}, standard=list(STANDARD),
                   early_exit_order=list(EARLY_EXIT_ORDER),
                   fields=['held', 'rise_mm', 'slip_mm', 'turn_deg', 'touching'],
                   note='every viable candidate has the standard conditions up to its first failure; the first '
                        'viable grasp of every ordering has all nine conditions',
                   seconds=secs, parts=table)
        with gzip.open(a.out, 'wt') as fh:
            json.dump(doc, fh, separators=(',', ':'))
        return
    if a.mode == 'robust':
        tasks = []
        for part in dev:
            cat = catalogue.load(Path(a.catalogues)/f'{part}.npz')
            tasks.append((part, [str(lab) for lab, ok in zip(cat['label'], cat['feasible']) if ok]))
    else:
        want = json.loads(Path(a.labels).read_text())
        tasks = [(part, sorted(want[part])) for part in dev if want.get(part)]
    tasks.sort(key=lambda t: -len(t[1]))
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for part, n in pool.map(job, tasks):
            print(part, n, flush=True)


if __name__ == '__main__':
    main()
