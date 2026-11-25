"""Dynamic check of a planned grasp in MuJoCo: close, lift, hold, shake.

The part is a free body made of its published convex collision pieces, resting
on the table where the cell puts it. The gripper is reduced to its two pads
(finger width x finger depth, 5 mm thick) on a kinematically driven palm, so
the arm and palm are left out: this tests whether the grasp holds the part
against gravity and a shake with friction, not whether the arm can do it
(the planner already checked reach and collisions).

Schedule (1 ms steps): settle 0.3 s; close 0.5 s to 15 mm past the part;
hold 0.2 s; lift 0.10 m in 1 s; hold 0.5 s; shake +-20 mm at 3 Hz for 0.5 s;
hold 0.3 s. Held: the part rose at least 80 mm and moved less than 20 mm
relative to the palm between the end of the lift and the end.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PAD_THICKNESS = .005
SQUEEZE = .015
SCHEDULE = dict(settle=.3, close=.5, grip=.2, lift=1.0, hold=.5, shake=.5, rest=.3)
PERTURBATIONS = (
    ('nominal', dict()),
    ('mass x2', dict(mass=2.)),
    ('mass x0.5', dict(mass=.5)),
    ('friction 0.3', dict(friction=.3)),
    ('pose noise 1', dict(noise_seed=1)),
    ('pose noise 2', dict(noise_seed=2)),
)


def read_obj(path):
    v = [list(map(float, l.split()[1:4])) for l in Path(path).read_text().splitlines() if l.startswith('v ')]
    return np.asarray(v, float)


def _quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def part_pose(project, body):
    """World rotation and translation of the body's visual mesh frame."""
    from research_grasping.phase4.visual_prior import _transform
    doc = json.loads(Path(project).read_text())['project']
    b = next(x for x in doc['environment']['bodies'] if x['name'] == body)
    rb, tb = _transform(b['pose'])
    g = next(x for x in b['geometries'] if x.get('role', 'both') in ('visual', 'both') and x['shape']['kind'] == 'mesh')
    rg, tg = _transform(g['origin'])
    gripper = doc['robot']['gripper']
    table_top = max(  # the table's top face, from its box geometry
        bb['pose']['translation']['z'] + gg['origin']['translation']['z'] + gg['shape']['size']['z']/2
        for bb in doc['environment']['bodies'] if bb['name'] == 'table'
        for gg in bb['geometries'] if gg['shape']['kind'] == 'box' and gg['name'] == 'top')
    return rb@rg, rb@tg+tb, gripper, table_top


def scene(pieces, R, t, table_top, candidate, gripper, mass=.1, friction=.5, noise_seed=None):
    """MJCF for one trial; ``candidate`` is the planner's grasp (world frame)."""
    vec = lambda d: np.array([d['x'], d['y'], d['z']], float)
    c, a, k = vec(candidate['center']), vec(candidate['approach']), vec(candidate['closing'])
    a /= np.linalg.norm(a); k -= a*np.dot(a, k); k /= np.linalg.norm(k)
    x = np.cross(k, a)
    Rp = np.column_stack([x, k, a])
    if noise_seed is not None:
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(noise_seed)
        c = c + Rp@np.array([rng.uniform(-.003, .003), 0., rng.uniform(-.003, .003)])
        Rp = Rp@Rotation.from_euler('z', rng.uniform(-3, 3), degrees=True).as_matrix()
    world_pieces = [(p@R.T)+t for p in pieces]
    allv = np.vstack(world_pieces)
    com = allv.mean(0)
    ext = allv.max(0)-allv.min(0)
    inertia = mass/12*np.array([ext[1]**2+ext[2]**2, ext[0]**2+ext[2]**2, ext[0]**2+ext[1]**2])
    open_half = candidate['width']/2 + .005 + PAD_THICKNESS/2
    fw, fd = gripper['fingerWidth']/2, gripper['fingerDepth']/2
    force = gripper['maxGripForce']/2
    meshes = ''.join(f'<mesh name="p{i}" vertex="{" ".join(f"{v:.6f}" for v in (p-com).ravel())}"/>'
                     for i, p in enumerate(world_pieces))
    geoms = ''.join(f'<geom type="mesh" mesh="p{i}" mass="0" condim="4" friction="{friction} .005 .0001"/>'
                    for i in range(len(world_pieces)))
    q = ' '.join(f'{v:.8f}' for v in _quat_wxyz(Rp))
    pad = (f'<geom type="box" size="{fw} {PAD_THICKNESS/2} {fd}" condim="4" friction="{friction} .005 .0001" '
           f'mass=".1"/>')
    # Stiff contacts (2 ms time constant) and several contact points per convex pair:
    # with MuJoCo's defaults, 45 N pads sink through a 0.1 kg mesh part.
    return f"""<mujoco><option timestep=".001" gravity="0 0 -9.81" cone="elliptic" impratio="10" integrator="implicitfast">
<flag multiccd="enable"/></option>
<default><geom solref=".002 1"/></default>
<asset>{meshes}</asset><worldbody>
<geom name="table" type="plane" pos="0 0 {table_top}" size="2 2 .01" friction="{friction} .005 .0001"/>
<body name="part" pos="{com[0]} {com[1]} {com[2]}"><freejoint/>
<inertial pos="0 0 0" mass="{mass}" diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>{geoms}</body>
<body name="target" mocap="true" pos="{c[0]} {c[1]} {c[2]}" quat="{q}"/>
<body name="palm" pos="{c[0]} {c[1]} {c[2]}" quat="{q}" gravcomp="1"><freejoint/>
<inertial pos="0 0 0" mass=".5" diaginertia=".001 .001 .001"/>
<body name="left" pos="0 {open_half} 0" gravcomp="1"><joint name="left" type="slide" axis="0 -1 0" damping="20"/>{pad}</body>
<body name="right" pos="0 {-open_half} 0" gravcomp="1"><joint name="right" type="slide" axis="0 1 0" damping="20"/>{pad}</body>
</body></worldbody>
<equality><weld body1="palm" body2="target" solref=".005 1"/></equality>
<actuator><position joint="left" kp="3000" forcelimited="true" forcerange="-{force} {force}"/>
<position joint="right" kp="3000" forcelimited="true" forcerange="-{force} {force}"/></actuator></mujoco>""", open_half


def trial(pieces, R, t, table_top, candidate, gripper, **perturbation):
    import mujoco
    xml, open_half = scene(pieces, R, t, table_top, candidate, gripper, **perturbation)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    part = model.body('part').id
    palm = model.body('palm').id
    start_palm = data.mocap_pos[0].copy()
    z0 = data.xpos[part][2]
    target = open_half - PAD_THICKNESS/2 + SQUEEZE
    s = SCHEDULE
    marks = np.cumsum([s['settle'], s['close'], s['grip'], s['lift'], s['hold'], s['shake'], s['rest']])
    steps = int(round(marks[-1]/model.opt.timestep))
    relative_after_lift = None
    for i in range(steps):
        time = i*model.opt.timestep
        closing = np.clip((time-marks[0])/s['close'], 0, 1)
        data.ctrl[:] = closing*target
        lift = np.clip((time-marks[2])/s['lift'], 0, 1)
        lift = .10*(3*lift**2-2*lift**3)
        shake = .02*np.sin(2*np.pi*3*(time-marks[4])) if marks[4] <= time < marks[5] else 0.
        data.mocap_pos[0] = start_palm+np.array([shake, 0., lift])
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            return dict(held=False, unstable=True)
        if relative_after_lift is None and time >= marks[3]:
            relative_after_lift = data.xpos[part]-data.xpos[palm]
    rise = float(data.xpos[part][2]-z0)
    slip = float(np.linalg.norm((data.xpos[part]-data.xpos[palm])-relative_after_lift))
    return dict(held=bool(rise >= .08 and slip < .02), rise=round(rise, 4), slip=round(slip, 4), unstable=False)


def evaluate(pieces, project, body, candidate):
    """Every perturbation for one grasp: {name: trial result}."""
    R, t, gripper, table_top = part_pose(project, body)
    return {name: trial(pieces, R, t, table_top, candidate, gripper, **kw) for name, kw in PERTURBATIONS}
