"""Articulated-arm pilot on the development first grasps (no EGAD data).

    python tools/research/phase7/articulated_pilot.py run --binary target/release/axisforge --store store.json \
        --projects POOL --collision COLLISION_DIR --catalogues cat80/ \
        --physics research/phase7/results/physics_dev_with_joint.json.gz --out-dir art/ --jobs 1
    python tools/research/phase7/articulated_pilot.py report --physics ... --out-dir art/ \
        --out research/phase7/results/articulated_simulator_validation.json

For every development part, the first viable grasp of M0, M1, M4, M9 and Joint (unique labels) is
planned again with the native planner (an encoded prior puts that label first) to get its joint
values, then simulated with research_grasping/phase7/articulated.py in seven conditions (nominal,
mass x2, mass x0.5, friction 0.3, loads 0.5/1/2 kg; torsion 0.005; pose noise is not applied at
this level). The report compares the articulated verdicts with the floating-gripper v3 verdicts of
the same grasps and conditions. Development evidence; a pilot, not a calibrated arm model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import physics as v2  # noqa: E402
from research_grasping.phase6.mechanism import bootstrap_diff, mcnemar  # noqa: E402
from research_grasping.phase7 import articulated as art  # noqa: E402

CONDITIONS = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)),
              ('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))
STANDARD4 = ('nominal', 'mass x2', 'mass x0.5', 'friction 0.3')
LOADS = ('load 0.5 kg', 'load 1 kg', 'load 2 kg')
METHODS = ('M0', 'M1', 'M4', 'M9', 'Joint')
STA = {}


def init(a):
    STA.update(vars(a))


def job(task):
    part, labels = task
    s = STA
    out_path = Path(s['out_dir'])/f'{part}.json'
    if out_path.exists():
        return part, 0.
    store = json.loads(Path(s['store']).read_text())
    proj = Path(s['projects'])/store[part]['project']
    R, t, g, tt = v1.part_pose(proj, part)
    pieces = [v1.read_obj(p) for p in sorted((Path(s['collision'])/part).glob('model_collision_*.obj'),
                                              key=lambda p: int(p.stem.split('_')[-1]))]
    cat = catalogue.load(Path(s['catalogues'])/f'{part}.npz')
    com, _ = v2.planner_com(cat)
    project = art.load_project(proj)
    start = time.perf_counter()
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for lab in labels:
            prior = Path(tmp)/'prior.json'
            prior.write_text(json.dumps(dict(weight=5., candidates=[dict(label=lab, score=1.)])))
            rep = json.loads(subprocess.run([s['binary'], 'plan', '--json', '--nr-grasps', '1', '--prior', str(prior),
                                             str(proj), part], check=True, capture_output=True, text=True,
                                            env={'RAYON_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'}).stdout)
            if not rep['plans'] or rep['plans'][0]['candidate']['label'] != lab:
                res[lab] = dict(error='planner did not return this label first',
                                got=rep['plans'][0]['candidate']['label'] if rep['plans'] else None)
                continue
            plan = rep['plans'][0]
            res[lab] = {name: art.trial(project, pieces, R, t, tt, plan, g, com=com, **kw) for name, kw in CONDITIONS}
    out_path.write_text(json.dumps(dict(seconds=round(time.perf_counter()-start, 1), trials=res)))
    return part, time.perf_counter()-start


def run(a):
    ph = json.load(gzip.open(a.physics, 'rt'))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    tasks = [(n, sorted(set(ph['core'][n][m] for m in METHODS))) for n in sorted(ph['core'])]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for part, sec in pool.map(job, tasks):
            print(part, round(sec), flush=True)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean()); pa, pb = a.mean(), b.mean()
    pe = pa*pb + (1-pa)*(1-pb)
    return round((po-pe)/(1-pe), 4) if pe < 1 else None


def report(a):
    ph = json.load(gzip.open(a.physics, 'rt'))
    parts = sorted(ph['core'])
    art_res = {n: json.loads((Path(a.out_dir)/f'{n}.json').read_text()) for n in parts if (Path(a.out_dir)/f'{n}.json').exists()}
    errors = {n: {lab: r for lab, r in d['trials'].items() if 'error' in r} for n, d in art_res.items()}
    errors = {n: e for n, e in errors.items() if e}
    agree = {}
    pairs = {c: ([], []) for c, _ in CONDITIONS}
    for n, d in art_res.items():
        for lab, r in d['trials'].items():
            if 'error' in r:
                continue
            v3r = ph['trials'][n][lab]['0.005']
            for c, _ in CONDITIONS:
                pairs[c][0].append(bool(v3r[c]['held_v3'])); pairs[c][1].append(bool(r[c]['held_art']))
    for c, (x, y) in pairs.items():
        x, y = np.array(x), np.array(y)
        agree[c] = dict(grasps=int(len(x)), v3_held=round(float(x.mean()), 4), articulated_held=round(float(y.mean()), 4),
                        agreement=round(float((x == y).mean()), 4), kappa=kappa(x, y),
                        v3_only=int((x & ~y).sum()), articulated_only=int((~x & y).sum()))
    ok_parts = [n for n in parts if n in art_res and all('error' not in art_res[n]['trials'][ph['core'][n][m]] for m in METHODS)]

    def outcome(n, m, which, level):
        lab = ph['core'][n][m]
        conds = STANDARD4 if which == 'robust4' else LOADS
        if level == 'articulated':
            r = art_res[n]['trials'][lab]
            return int(all(r[c]['held_art'] for c in conds))
        return int(all(ph['trials'][n][lab]['0.005'][c]['held_v3'] for c in conds))
    rates, contrasts = {}, {}
    for level in ('v3 floating gripper', 'articulated'):
        lv = 'articulated' if level == 'articulated' else 'v3'
        rates[level] = {m: {w: round(float(np.mean([outcome(n, m, w, lv) for n in ok_parts])), 4) for w in ('robust4', 'load')}
                        for m in METHODS}
        contrasts[level] = {}
        for x, y in (('Joint', 'M0'), ('Joint', 'M1'), ('M4', 'M1'), ('M9', 'M1')):
            for w in ('robust4', 'load'):
                A = [outcome(n, x, w, lv) for n in ok_parts]; B = [outcome(n, y, w, lv) for n in ok_parts]
                contrasts[level][f'{x} - {y}, {w}'] = dict(**bootstrap_diff(A, B), mcnemar=mcnemar(A, B))
    track = [r[c]['max_tracking_error_rad'] for d in art_res.values() for r in d['trials'].values() if 'error' not in r
             for c, _ in CONDITIONS if 'max_tracking_error_rad' in r[c]]
    sat = [r[c]['torque_saturated_steps'] for d in art_res.values() for r in d['trials'].values() if 'error' not in r
           for c, _ in CONDITIONS if 'torque_saturated_steps' in r[c]]
    unstable = sum(1 for d in art_res.values() for r in d['trials'].values() if 'error' not in r for c, _ in CONDITIONS
                   if r[c].get('unstable'))
    out = dict(status='executed (pilot)', note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '),
               model='research_grasping/phase7/articulated.py', parts_simulated=len(art_res), parts_all_methods=len(ok_parts),
               planner_label_errors=errors,
               robustness_definition='robust4 = held in nominal, mass x2, mass x0.5 and friction 0.3 (pose-noise conditions '
                                     'are not simulated at the articulated level, so robust4 differs from the six-condition '
                                     'robustness of v3 and the two are compared on the same four conditions)',
               agreement_by_condition=agree, rates=rates, contrasts=contrasts,
               controller=dict(max_tracking_error_rad=dict(median=round(float(np.median(track)), 4),
                                                           p99=round(float(np.percentile(track, 99)), 4),
                                                           max=round(float(np.max(track)), 4)),
                               trials_with_torque_saturation=int(sum(1 for x in sat if x > 0)), trials=len(sat),
                               unstable_trials=unstable),
               seconds_per_part_median=float(np.median([d['seconds'] for d in art_res.values()])),
               limitations=['the robot is the cell model (link masses and inertias from the project), not a calibrated '
                            'industrial arm', 'arm links have no collision geometry', 'joint-space lift instead of a '
                            'Cartesian one; shake through the base joint', 'computed-torque control with exact model, no '
                            'joint friction, backlash or sensor noise', 'same contact model and uncalibrated torsional '
                            'friction as v3', 'simulation, not a real robot'])
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('parts_simulated', 'parts_all_methods', 'agreement_by_condition', 'rates',
                                         'controller')}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    for k in ('binary', 'store', 'projects', 'collision', 'catalogues', 'physics', 'out-dir'):
        r.add_argument(f'--{k}', required=True)
    r.add_argument('--jobs', type=int, default=1)
    e = sub.add_parser('report')
    for k in ('physics', 'out-dir', 'out'):
        e.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
