"""Physics controls for the Phase VI dynamic test (v2).

    python tools/research/phase6/physics_controls.py --dev dev_parts.json --catalogues cat80/ \
        --collision COLLISION_DIR --projects POOL --store store.json --phase5-dynamics dynamics.json.gz \
        --out physics_validation.json --sample 60

Synthetic controls with known answers (free fall, energy and angular momentum
in free flight, open pads, the grip force and friction thresholds, a centred
box under every perturbation, the lever arm a torsional grip can carry) and
sensitivity checks on real grasps (time step, solver, torsional friction,
centre of mass, penetration). The sample of real grasps is the first viable
grasp of the learned ordering on the first ``--sample`` development parts in
name order.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_grasping.phase5 import catalogue, dynamics as v1  # noqa: E402
from research_grasping.phase6 import physics  # noqa: E402

G = 9.81
GRIPPER = None  # the cell's gripper, read from the first real part


def box(sx, sy, sz):
    return np.array([[x, y, z] for x in (-sx/2, sx/2) for y in (-sy/2, sy/2) for z in (0., sz)], float)


def top_grasp(offset_x=0., width=.04, height=.04):
    return dict(center=dict(x=offset_x, y=0., z=height/2), approach=dict(x=0., y=0., z=-1.),
                closing=dict(x=0., y=1., z=0.), width=width)


def synthetic(pieces, cand, **kw):
    return physics.trial(pieces, np.eye(3), np.zeros(3), 0., cand, GRIPPER, **kw)


def free_flight():
    """Free fall from rest and torque-free spin: integrator checks with exact answers."""
    import mujoco
    xml = """<mujoco><option timestep=".001" integrator="implicitfast"/><worldbody>
<body name="b" pos="0 0 1"><freejoint/><inertial pos="0 0 0" mass=".1" diaginertia="1e-4 2e-4 3e-4"/></body>
</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    T = .3
    for _ in range(int(T/m.opt.timestep)):
        mujoco.mj_step(m, d)
    drop = 1-d.qpos[2]
    fall = dict(t_s=T, simulated_drop_m=round(float(drop), 6), exact_drop_m=round(.5*G*T*T, 6),
                relative_error=round(float(abs(drop-.5*G*T*T)/(.5*G*T*T)), 5))
    m.opt.gravity[:] = 0
    d = mujoco.MjData(m); d.qvel[3:6] = [1., 2., .5]
    mujoco.mj_forward(m, d)
    I = np.array([1e-4, 2e-4, 3e-4])

    def state(d):
        R = d.xmat[1].reshape(3, 3)
        w_body = d.qvel[3:6]  # free joint angular velocity is in the body frame
        L = R@(I*w_body)
        return L, .5*float(np.sum(I*w_body**2))
    L0, E0 = state(d)
    for _ in range(2000):
        mujoco.mj_step(m, d)
    L1, E1 = state(d)
    spin = dict(t_s=2., angular_momentum_drift=round(float(np.linalg.norm(L1-L0)/np.linalg.norm(L0)), 5),
                energy_drift=round(float(abs(E1-E0)/E0), 5))
    return dict(free_fall=fall, torque_free_spin=spin,
                verdict='pass' if fall['relative_error'] < .01 and spin['angular_momentum_drift'] < .02 else 'fail')


def threshold(values, held):
    """Smallest value from which every larger value held, or None."""
    for i, v in enumerate(values):
        if all(held[i:]):
            return v
    return None


def force_and_friction():
    cube = [box(.04, .04, .04)]
    cand = top_grasp()
    # peak demand: weight plus the lift's peak acceleration (6*0.1/1^2) and the shake's (0.02*(6 pi)^2)
    a_lift, a_shake = .6, .02*(2*np.pi*3)**2
    static = .1*G
    dynamic = .1*np.hypot(G+a_lift, a_shake)
    forces = [.25, .5, .75, 1., 1.25, 1.5, 2., 3., 5.]
    held_f = [synthetic(cube, cand, force=f)['held'] for f in forces]
    mus = [.03, .05, .08, .1, .12, .15, .2, .3]
    held_m = [synthetic(cube, cand, force=5., friction=mu)['held'] for mu in mus]
    f_star, m_star = threshold(forces, held_f), threshold(mus, held_m)
    band_f = [round(static/(2*.5), 3), round(dynamic/(2*.5), 3)]
    band_m = [round(static/(2*5.), 3), round(dynamic/(2*5.), 3)]
    return dict(
        insufficient_force=dict(part='40 mm cube, 0.1 kg, mu 0.5, centred top grasp', force_per_pad_N=forces,
                                held=held_f, simulated_threshold_N=f_star, coulomb_band_N=band_f,
                                note='band: weight only .. weight with the peak lift and shake accelerations',
                                verdict='pass' if f_star is not None and band_f[0]*.7 <= f_star <= band_f[1]*1.6 else 'fail'),
        low_friction=dict(part='same cube, 5 N per pad', mu=mus, held=held_m, simulated_threshold=m_star,
                          coulomb_band=band_m,
                          verdict='pass' if m_star is not None and band_m[0]*.7 <= m_star <= band_m[1]*1.6 else 'fail'))


def simple_object():
    cube = [box(.04, .04, .04)]
    res = {name: synthetic(cube, top_grasp(), **kw) for name, kw in PERTURBATIONS_V2}
    return dict(part='40 mm cube, centred top grasp', results={k: v['held'] for k, v in res.items()},
                verdict='pass' if all(v['held'] for v in res.values()) else 'fail')


def open_pads():
    cube = [box(.04, .04, .04)]
    r = synthetic(cube, top_grasp(), close=False)
    return dict(part='40 mm cube, pads never close', held=r['held'], rise_m=r['rise'], touching=r['touching'],
                verdict='pass' if not r['held'] and r['rise'] < .005 else 'fail')


def lever_arm():
    """A 160 mm bar gripped at a distance d from its centre: the load it can carry before it turns."""
    bar = [box(.16, .03, .03)]
    out = []
    masses = [.05, .1, .2, .4, .8, 1.6]
    for d in (0., .02, .04, .06):
        cand = top_grasp(offset_x=d, width=.03, height=.03)
        rows = [synthetic(bar, cand, mass=m) for m in masses]
        out.append(dict(offset_m=d, mass_kg=masses, held=[r['held'] for r in rows],
                        turn_deg=[r.get('turn_deg') for r in rows]))
    return dict(part='160x30x30 mm bar, 45 N per pad, mu 0.5, torsional coefficient 0.005 m', rows=out,
                note='torque to carry: m g d; tangential capacity 2 mu F = 45 N is never the limit here')


PERTURBATIONS_V2 = (
    ('nominal', dict()),
    ('mass x2', dict(mass=.2)),
    ('mass x0.5', dict(mass=.05)),
    ('friction 0.3', dict(friction=.3)),
    ('pose noise 1', dict(noise_seed=1)),
    ('pose noise 2', dict(noise_seed=2)),
)


def real_sample(a):
    dev = json.loads(Path(a.dev).read_text())['parts']
    store = json.loads(Path(a.store).read_text())
    d5 = json.load(gzip.open(a.phase5_dynamics))['parts']
    out = []
    for part in sorted(dev)[:a.sample]:
        first = d5[part]['methods']['learned']['first']
        proj = Path(a.projects)/store[part]['project']
        R, t, g, tt = v1.part_pose(proj, part)
        pieces = [v1.read_obj(p) for p in sorted((Path(a.collision)/part).glob('model_collision_*.obj'),
                                                  key=lambda p: int(p.stem.split('_')[-1]))]
        cat = catalogue.load(Path(a.catalogues)/f'{part}.npz')
        cand = {c['label']: c for c in catalogue.candidates(cat)}[first]
        com, _ = physics.planner_com(cat)
        out.append((part, pieces, R, t, tt, g, cand, com))
    return out


def agreement(sample, base, variant, conditions=(('nominal', dict()), ('mass x2', dict(mass=.2)), ('2 kg', dict(mass=2.)))):
    same, n, flips = 0, 0, []
    for part, pieces, R, t, tt, g, cand, com in sample:
        for name, kw in conditions:
            x = physics.trial(pieces, R, t, tt, cand, g, com=com, **{**base, **kw})
            y = physics.trial(pieces, R, t, tt, cand, g, com=com, **{**variant, **kw})
            n += 1; same += x['held'] == y['held']
            if x['held'] != y['held']:
                flips.append([part, name, x['held'], y['held']])
    return dict(trials=n, agree=same, agreement=round(same/n, 4), flips=flips)


def main():
    global GRIPPER
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('dev', 'catalogues', 'collision', 'projects', 'store', 'phase5-dynamics', 'out'):
        p.add_argument(f'--{k}', required=True)
    p.add_argument('--sample', type=int, default=60)
    a = p.parse_args()
    start = time.perf_counter()
    sample = real_sample(a)
    GRIPPER = sample[0][5]
    out = dict(model='phase6 v2 (planner centre of mass); research_grasping/phase6/physics.py',
               gripper=dict(fingerWidth=GRIPPER['fingerWidth'], fingerDepth=GRIPPER['fingerDepth'],
                            maxGripForce=GRIPPER['maxGripForce']))
    out['free_flight'] = free_flight()
    out['open_pads'] = open_pads()
    out.update(force_and_friction())
    out['simple_object'] = simple_object()
    out['lever_arm'] = lever_arm()
    # real grasps: pads open must drop every part
    opened = [physics.trial(pc, R, t, tt, cand, g, com=com, close=False) for _, pc, R, t, tt, g, cand, com in sample]
    out['open_pads_real'] = dict(grasps=len(opened), held=sum(r['held'] for r in opened),
                                 max_rise_m=max(r['rise'] for r in opened),
                                 verdict='pass' if not any(r['held'] for r in opened) else 'fail')
    out['timestep'] = agreement(sample, dict(), dict(timestep=.0005))
    out['solver_cg'] = agreement(sample, dict(), dict(solver='CG'))
    out['solver_pgs'] = agreement(sample, dict(), dict(solver='PGS'))
    out['torsion_0.001'] = agreement(sample, dict(), dict(torsion=.001))
    out['torsion_0.02'] = agreement(sample, dict(), dict(torsion=.02))
    out['com_v1_vs_v2'] = dict(note='same grasps with v1 (vertex-mean) centre of mass', **_com_compare(sample))
    pen = [physics.trial(pc, R, t, tt, cand, g, com=com)['max_penetration_mm'] for _, pc, R, t, tt, g, cand, com in sample]
    out['penetration'] = dict(grasps=len(pen), median_mm=round(float(np.median(pen)), 3),
                              p95_mm=round(float(np.percentile(pen, 95)), 3), max_mm=round(float(np.max(pen)), 3))
    out['sample'] = dict(parts=[s[0] for s in sample], grasp='first viable grasp of the learned ordering')
    out['wall_s'] = round(time.perf_counter()-start, 1)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


def _com_compare(sample):
    same, n, flips = 0, 0, []
    for part, pieces, R, t, tt, g, cand, com in sample:
        for name, kw in (('nominal', dict()), ('mass x2', dict(mass=.2)), ('2 kg', dict(mass=2.))):
            x = physics.trial(pieces, R, t, tt, cand, g, com=None, **kw)
            y = physics.trial(pieces, R, t, tt, cand, g, com=com, **kw)
            n += 1; same += x['held'] == y['held']
            if x['held'] != y['held']:
                flips.append([part, name, x['held'], y['held']])
    return dict(trials=n, agree=same, agreement=round(same/n, 4), flips=flips)


if __name__ == '__main__':
    main()
