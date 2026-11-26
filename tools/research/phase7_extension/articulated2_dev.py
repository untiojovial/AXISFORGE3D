"""Articulated model v2 on the development first grasps: load tolerance, timing sensitivity and controls.

    python tools/research/phase7_extension/articulated2_dev.py run --binary target/release/axisforge --store store.json \
        --projects POOL --collision COLLISION_DIR --catalogues cat80/ --physics research/phase7/results/physics_dev_with_joint.json.gz \
        --out-dir art2/ --jobs 2 [--labels-json extra.json]
    python tools/research/phase7_extension/articulated2_dev.py report --physics ... --out-dir art2/ \
        --out research/phase7_extension/results/articulated2_development.json

Development only (151 dynamic parts). Grasps: the first viable grasp of M0, M1 and Joint (Phase VII out-of-fold
orders), plus any labels in --labels-json (e.g. the selected method's). Each is re-planned with the native planner
(a one-label prior) to get its joint values. It is then simulated with research_grasping/phase7x/articulated2.py:

- standard: nominal, mass x2, mass x0.5, friction 0.3 at the pilot's timing;
- loads 0.5, 1 and 2 kg at the pilot's timing and with three timing variants: lift 0.5 s (fast), lift 2.0 s
  (slow), close 0.25 s (fast closing);
- controls on M0's first grasp: jaws kept open (must not lift), frictionless contacts (must slip),
  gravity off (must hold).

Torsion 0.005 m. Pose noise is not applied at this level. Telemetry per trial: rise, slip, rotation, contact,
pad normal forces, friction use, grip-force and torque saturation, tracking error, relative motion, unexpected
link-table/link-part and palm-table contacts.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from research_grasping.phase5 import catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import physics as v2  # noqa: E402
from research_grasping.phase7 import articulated as art  # noqa: E402
from research_grasping.phase7x import articulated2 as art2  # noqa: E402

STANDARD = (('nominal', {}), ('mass x2', dict(mass=.2)), ('mass x0.5', dict(mass=.05)), ('friction 0.3', dict(friction=.3)))
LOADS = (('load 0.5 kg', dict(mass=.5)), ('load 1 kg', dict(mass=1.)), ('load 2 kg', dict(mass=2.)))
TIMING = (('', None), (' | lift 0.5 s', dict(lift=.5)), (' | lift 2.0 s', dict(lift=2.)), (' | close 0.25 s', dict(close=.25)))
CONTROLS = (('control: open', dict(control='open')), ('control: frictionless', dict(control='frictionless')),
            ('control: zero-g', dict(control='zero-g', mass=2.)))
METHODS = ('M0', 'M1', 'Joint')
STA = {}


def init(a):
    STA.update(vars(a))


def job(task):
    part, labels, control_label = task
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
            rep = json.loads(subprocess.run([s['binary'], 'plan', '--json', '--nr-grasps', '1', '--prior', str(prior), str(proj), part],
                                            check=True, capture_output=True, text=True,
                                            env={'RAYON_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'}).stdout)
            if not rep['plans'] or rep['plans'][0]['candidate']['label'] != lab:
                res[lab] = dict(error='planner did not return this label first'); continue
            plan = rep['plans'][0]
            r = {name: art2.trial(project, pieces, R, t, tt, plan, g, com=com, **kw) for name, kw in STANDARD}
            for suffix, sched in TIMING:
                for name, kw in LOADS:
                    r[name + suffix] = art2.trial(project, pieces, R, t, tt, plan, g, com=com, schedule=sched, **kw)
            if lab == control_label:
                for name, kw in CONTROLS:
                    r[name] = art2.trial(project, pieces, R, t, tt, plan, g, com=com, **kw)
            res[lab] = r
    out_path.write_text(json.dumps(dict(seconds=round(time.perf_counter()-start, 1), trials=res)))
    return part, time.perf_counter()-start


def run(a):
    ph = json.load(gzip.open(a.physics, 'rt'))
    extra = json.loads(Path(a.labels_json).read_text()) if a.labels_json else {}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    tasks = [(n, sorted(set(ph['core'][n][m] for m in METHODS) | set(extra.get(n, []))), ph['core'][n]['M0']) for n in sorted(ph['core'])]
    if a.limit:
        tasks = tasks[:a.limit]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for part, sec in pool.map(job, tasks):
            print(part, round(sec), flush=True)


def report(a):
    ph = json.load(gzip.open(a.physics, 'rt'))
    extra = json.loads(Path(a.labels_json).read_text()) if a.labels_json else {}
    docs = {f.stem: json.loads(f.read_text())['trials'] for f in Path(a.out_dir).glob('*.json')}
    methods = {m: {n: ph['core'][n][m] for n in docs} for m in METHODS}
    del extra  # extra labels are reported by the analysis that requested them (v3_models.py)
    res = dict(note=__doc__.strip().split('\n\n', 1)[1].replace('\n', ' '), parts=len(docs))
    # controls
    ctl = {}
    for name, _ in CONTROLS:
        rs = [docs[n][ph['core'][n]['M0']][name] for n in docs if name in docs[n].get(ph['core'][n]['M0'], {})]
        lifted = [r['rise'] >= .8*r['planned_rise'] for r in rs if not r.get('unstable')]
        ctl[name] = dict(trials=len(rs), held_art=int(sum(r.get('held_art', False) for r in rs)), lifted=int(sum(lifted)),
                         unstable=int(sum(r.get('unstable', False) for r in rs)))
    res['controls'] = ctl
    # rates per method and condition, and telemetry summaries
    conds = [c for c, _ in STANDARD] + [c + s for s, _ in TIMING for c, _ in LOADS]
    rates, tele = {}, {}
    for m, lab in methods.items():
        rates[m], tele[m] = {}, {}
        for c in conds:
            rs = [docs[n][lab[n]][c] for n in docs if 'error' not in docs[n].get(lab[n], {'error': 1}) and c in docs[n][lab[n]]]
            rates[m][c] = dict(n=len(rs), held_art=float(np.mean([r.get('held_art', False) for r in rs])) if rs else None)
            ok = [r for r in rs if not r.get('unstable')]
            if ok:
                tele[m][c] = dict(peak_friction_use_median=float(np.median([r['peak_friction_use'] for r in ok])),
                                  grip_saturated_share=float(np.mean([r['grip_force_saturated_steps'] > 0 for r in ok])),
                                  torque_saturated_share=float(np.mean([r['torque_saturated_steps'] > 0 for r in ok])),
                                  unexpected_link_contact_share=float(np.mean([bool(r['unexpected_link_contacts']) for r in ok])),
                                  palm_table_share=float(np.mean([r['palm_table_contact'] for r in ok])),
                                  max_relative_motion_mm_median=float(np.median([r['max_relative_motion_mm'] for r in ok])),
                                  tracking_error_rad_median=float(np.median([r['max_tracking_error_rad'] for r in ok])))
    res['held_art_rate'] = rates
    res['telemetry'] = tele
    # paired load tolerance Joint - M0 at each timing (all three loads held)
    pair = {}
    for suffix, _ in TIMING:
        keys = [c + suffix for c, _ in LOADS]

        def tol(n, m):
            r = docs[n].get(methods[m][n], {})
            return None if 'error' in r or any(k not in r for k in keys) else int(all(r[k].get('held_art', False) for k in keys))
        both = [(tol(n, 'Joint'), tol(n, 'M0')) for n in docs]
        both = [b for b in both if None not in b]
        pair[f'loads{suffix or " | pilot timing"}'] = C.boot_diff(*zip(*both)) if both else None
    res['joint_minus_m0_load_tolerance'] = pair
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(dict(controls=ctl, pair={k: (round(v['difference'], 3), [round(x, 3) for x in v['ci95']]) for k, v in pair.items() if v}), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    for k in ('binary', 'store', 'projects', 'collision', 'catalogues', 'physics', 'out-dir'):
        r.add_argument(f'--{k}', required=True)
    r.add_argument('--jobs', type=int, default=2); r.add_argument('--labels-json'); r.add_argument('--limit', type=int)
    q = sub.add_parser('report')
    for k in ('physics', 'out-dir', 'out'):
        q.add_argument(f'--{k}', required=True)
    q.add_argument('--labels-json')
    a = p.parse_args()
    dict(run=run, report=report)[a.cmd](a)


if __name__ == '__main__':
    main()
