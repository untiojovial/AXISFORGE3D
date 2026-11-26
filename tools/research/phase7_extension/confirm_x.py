"""Run the new confirmation of PHASE7X_PROTOCOL.md (EGAD positions 300-499), stage by stage, each stage once.

    python tools/research/phase7_extension/confirm_x.py prepare  --pool NEW_POOL --binary BIN --out-dir OUT --jobs 3
    python tools/research/phase7_extension/confirm_x.py orders   --pool NEW_POOL --binary BIN --out-dir OUT --freeze FREEZE.json --jobs 3
    python tools/research/phase7_extension/confirm_x.py physics  --pool NEW_POOL --out-dir OUT --jobs 3
    python tools/research/phase7_extension/confirm_x.py timing   --pool NEW_POOL --binary BIN --out-dir OUT --freeze FREEZE.json --cpu 3
    python tools/research/phase7_extension/confirm_x.py analyse  --out-dir OUT --freeze FREEZE.json --out research/phase7_extension/results/confirmation_x.json

Stages:

- **prepare.** The frozen Phase VII catalogue procedure (egad_confirm.prepare): one exhaustively labelled
  80 mm catalogue per object, plus store.json and status.json.
- **orders.** Decisions of the native planner, `plan --json --nr-grasps 1 [--learned F --order MODE]`, for
  every method of the freeze. That gives N_F, the first grasp and Success@k. The Python replay (catalogue order,
  fusion orders) is computed too, as a check; mismatches are recorded. GPD+gate comes from the adapter scores
  in OUT/gpd. One JSON per object (restart-safe).
- **physics.** v3 trials of every method's first grasp: the nine conditions at torsion 0.005 m for all methods,
  and also at 0.001 and 0.02 m for the methods named in the freeze's torsion grid. One JSON per object.
- **timing.** Full cold-process cost per object and mode, 3 repetitions, one thread pinned to one CPU, seeded
  random order (integrated_benchmark.py logic).
- **analyse.** The protocol's tests, read from the freeze record, which is hash-checked before use.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools/research/phase7')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from research_grasping.phase5 import catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import physics as v2  # noqa: E402
from research_grasping.phase7 import physics as v3  # noqa: E402

CONDITIONS = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)),
              ('pose noise 1', dict(noise_seed=1)), ('pose noise 2', dict(noise_seed=2)),
              ('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))
BUDGETS = (1, 5, 10, 20, 50)
STA = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_freeze(path):
    fr = json.loads(Path(path).read_text())
    for f, h in fr['files_sha256'].items():
        if sha(ROOT/f) != h:
            raise SystemExit(f'frozen file changed: {f}')
    return fr


# ------------------------------------------------------------------------------------------ prepare
def prepare(a):
    import egad_confirm as ec
    ec.prepare(argparse.Namespace(pool=a.pool, binary=a.binary, out_dir=a.out_dir, jobs=a.jobs))


# ------------------------------------------------------------------------------------------- orders
def plan_decision(binary, proj, name, models=None, mode=None):
    args = [binary, 'plan', '--json', '--nr-grasps', '1', str(proj), name]
    if models:
        args[3:3] = ['--learned', str(ROOT/models), '--order', mode]
    rep = json.loads(subprocess.run(args, check=True, capture_output=True, text=True, timeout=3600,
                                    env={'RAYON_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'}).stdout)
    return rep['candidatesExamined'], (rep['plans'][0]['candidate']['label'] if rep['plans'] else None)


def orders_one(task):
    n, a_ = task
    out = Path(a_['out_dir'])/'orders'/f'{n}.json'
    if out.exists():
        return n, 'cached'
    fr = STA['freeze']
    store = json.loads((Path(a_['out_dir'])/'store.json').read_text())
    proj = Path(a_['pool'])/store[n]['project']
    cat = catalogue.load(Path(a_['out_dir'])/'cat80'/f'{n}.npz')
    viable = bool(cat['feasible'].any())
    row = dict(viable=viable, candidates=int(len(cat['label'])), methods={})
    for name, m in fr['methods'].items():
        if m['kind'] == 'native':
            t0 = time.perf_counter()
            nf, first = plan_decision(a_['binary'], proj, n, m.get('models'), m.get('mode'))
            row['methods'][name] = dict(nf=nf if first else None, first=first, examined=nf, seconds=time.perf_counter()-t0)
    # Python replay check (M0 = catalogue order)
    o = catalogue.order(cat)
    ref = C.first(cat, o)
    row['replay_M0'] = list(ref)
    row['replay_M0_match'] = (row['methods']['M0']['nf'], row['methods']['M0']['first']) == ref if viable else row['methods']['M0']['first'] is None
    # GPD+gate (secondary)
    g = Path(a_['out_dir'])/'gpd'/f'{n}.json'
    if g.exists() and 'scores' in json.loads(g.read_text()):
        from research_grasping.phase6 import fusion
        s = np.asarray(json.loads(g.read_text())['scores'], float)
        go = fusion.order(cat, (s - s.min())/(s.max() - s.min()) if s.max() > s.min() else np.zeros_like(s))
        nf, first = C.first(cat, go)
        hy = C.first(cat, C.interleave(cat, go, o))
        row['methods']['GPD+gate'] = dict(nf=nf, first=first)
        row['methods']['hybrid GPD/M0'] = dict(nf=hy[0], first=hy[1])
    out.write_text(json.dumps(row))
    return n, 'ok'


def orders(a):
    fr = load_freeze(a.freeze)
    status = json.loads((Path(a.out_dir)/'status.json').read_text())
    names = sorted(n for n, s in status.items() if s['status'] in ('viable', 'no viable candidate'))
    (Path(a.out_dir)/'orders').mkdir(exist_ok=True)
    STA['freeze'] = fr
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=STA.update, initargs=(dict(freeze=fr),)) as pool:
        for n, st in pool.map(orders_one, [(n, dict(out_dir=a.out_dir, pool=a.pool, binary=a.binary)) for n in names]):
            print(n, st, flush=True)


# ------------------------------------------------------------------------------------------ physics
def physics_one(task):
    n, a_, jobs = task
    out = Path(a_['out_dir'])/'trials'/f'{n}.json'
    if out.exists():
        return n, 0.
    store = json.loads((Path(a_['out_dir'])/'store.json').read_text())
    R, t, g, tt = v1.part_pose(Path(a_['pool'])/store[n]['project'], n)
    pieces = [v1.read_obj(p) for p in sorted((Path(a_['pool'])/n/'collision').glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(a_['out_dir'])/'cat80'/f'{n}.npz')
    com, _ = v2.planner_com(cat)
    cands = {c['label']: c for c in catalogue.candidates(cat)}
    t0 = time.perf_counter()
    res = {label: {tor: {c: v3.trial(pieces, R, t, tt, cands[label], g, com=com, torsion=float(tor), **kw) for c, kw in CONDITIONS}
                   for tor in tors} for label, tors in jobs}
    out.write_text(json.dumps(dict(seconds=round(time.perf_counter()-t0, 1), trials=res)))
    return n, time.perf_counter()-t0


def physics(a):
    fr = load_freeze(a.freeze)
    (Path(a.out_dir)/'trials').mkdir(exist_ok=True)
    tasks = []
    for f in sorted((Path(a.out_dir)/'orders').glob('*.json')):
        row = json.loads(f.read_text())
        if not row['viable']:
            continue
        want = {}
        for m, r in row['methods'].items():
            if r.get('first'):
                want.setdefault(r['first'], set()).add('0.005')
                if m in fr['torsion_grid_methods']:
                    want[r['first']].update(('0.001', '0.02'))
        tasks.append((f.stem, dict(out_dir=a.out_dir, pool=a.pool), sorted((lab, sorted(ts)) for lab, ts in want.items())))
    tasks.sort(key=lambda t: -sum(len(ts) for _, ts in t[2]))
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for n, sec in pool.map(physics_one, tasks):
            print(n, round(sec), flush=True)


# ------------------------------------------------------------------------------------------- timing
def timing(a):
    import integrated_benchmark as ib
    fr = load_freeze(a.freeze)
    store = json.loads((Path(a.out_dir)/'store.json').read_text())
    names = sorted(f.stem for f in (Path(a.out_dir)/'orders').glob('*.json'))
    raw_dir = Path(a.out_dir)/'timing'; raw_dir.mkdir(exist_ok=True)
    modes = {name: m for name, m in fr['methods'].items() if m['kind'] == 'native' and name in fr['timed_methods']}
    order_rng = random.Random(fr['timing_seed']); order_rng.shuffle(names)
    for i, n in enumerate(names):
        path = raw_dir/f'{n}.json'
        if path.exists():
            continue
        rnd = random.Random(f"{fr['timing_seed']}-{n}")
        proj = Path(a.pool)/store[n]['project']
        rec = dict(load_before=__import__('os').getloadavg()[0], cold={k: [] for k in modes})
        for _ in range(fr['timing_reps']):
            ms = list(modes); rnd.shuffle(ms)
            for name in ms:
                m = modes[name]
                args = [a.binary, 'plan', '--json', '--nr-grasps', '1', str(proj), n]
                if m.get('models'):
                    args[3:3] = ['--learned', str(ROOT/m['models']), '--order', m['mode']]
                out, wall, cpu, _ = ib.spawn(args, a.cpu)
                rec['cold'][name].append(dict(wall=wall, cpu=cpu, peak_kib=out.get('peakResidentKiB'), examined=out['candidatesExamined'],
                                              first=out['plans'][0]['candidate']['label'] if out['plans'] else None,
                                              timings=out['timings'], project_load=out.get('projectLoadSeconds'),
                                              model_load=out.get('modelLoadSeconds', 0.)))
        rec['load_after'] = __import__('os').getloadavg()[0]
        path.write_text(json.dumps(rec))
        print(i+1, len(names), n, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('prepare', 'orders', 'physics', 'timing'):
        s = sub.add_parser(name)
        s.add_argument('--pool', required=True); s.add_argument('--out-dir', required=True)
        s.add_argument('--binary'); s.add_argument('--freeze'); s.add_argument('--jobs', type=int, default=3); s.add_argument('--cpu', type=int, default=3)
    a = p.parse_args()
    dict(prepare=prepare, orders=orders, physics=physics, timing=timing)[a.cmd](a)


if __name__ == '__main__':
    main()
