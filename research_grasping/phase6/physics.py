"""Phase VI dynamics (v2): the Phase V lift-hold-shake test, with the part's
centre of mass where the planner puts it, telemetry, and the settings the
physics controls vary.

Phase V (``research_grasping.phase5.dynamics``, v1) put the part's centre of
mass at the mean of its collision-piece vertices. The planner measures every
grasp's lever arm (``center_offset``) from the mesh's own centroid, and the two
points differ by 6 mm in the median part and by up to 51 mm. v2 keeps v1's
geometry, contacts and schedule and only moves the centre of mass to the
planner's, so the lever arm the planner scores is the one the simulation
loads. v1 is left as it was; both are reported.

Everything else is v1: pads only (no palm or arm), 1 ms steps, stiff contacts,
elliptic friction cones with torsional friction (condim 4), a 0.1 kg part with
a box inertia, 45 N per pad. ``trial`` also returns how far the part turned
relative to the palm, whether the pads still touch it at the end, and the
deepest penetration seen.
"""
from __future__ import annotations

import numpy as np

from research_grasping.phase5 import dynamics as v1

PERTURBATIONS = v1.PERTURBATIONS
SCHEDULE = v1.SCHEDULE
PAD_THICKNESS = v1.PAD_THICKNESS
SQUEEZE = v1.SQUEEZE


def planner_com(cat):
    """The planner's centre of mass, recovered from a catalogue.

    Every candidate stores its closing line and the distance from the
    planner's centre of mass to that line; the point that fits all of them
    is the planner's centre of mass (residuals are below a micrometre).
    """
    from scipy.optimize import least_squares
    co = np.asarray(cat['center_offset'], float)
    ok = np.isfinite(co)
    P = np.asarray(cat['center'], float)[ok]
    K = np.asarray(cat['closing'], float)[ok]
    K = K/np.linalg.norm(K, axis=1, keepdims=True)
    co = co[ok]

    def residual(c):
        v = c - P
        v = v - (v*K).sum(1, keepdims=True)*K
        return np.linalg.norm(v, axis=1) - co
    sol = least_squares(residual, P.mean(0))
    return sol.x, float(np.abs(sol.fun).max())


def _quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def scene(pieces, R, t, table_top, candidate, gripper, mass=.1, friction=.5, noise_seed=None, com=None,
          torsion=.005, timestep=.001, solver='Newton', force=None, condim=4):
    """MJCF for one trial. ``com`` (world frame) moves the centre of mass; None keeps v1's vertex mean."""
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
    origin = allv.mean(0)  # body frame: v1's, so the geometry is unchanged
    inertial = np.zeros(3) if com is None else np.asarray(com, float)-origin
    ext = allv.max(0)-allv.min(0)
    inertia = mass/12*np.array([ext[1]**2+ext[2]**2, ext[0]**2+ext[2]**2, ext[0]**2+ext[1]**2])
    open_half = candidate['width']/2 + .005 + PAD_THICKNESS/2
    fw, fd = gripper['fingerWidth']/2, gripper['fingerDepth']/2
    force = gripper['maxGripForce']/2 if force is None else force
    fr = f'{friction} {torsion} .0001'
    meshes = ''.join(f'<mesh name="p{i}" vertex="{" ".join(f"{v:.6f}" for v in (p-origin).ravel())}"/>'
                     for i, p in enumerate(world_pieces))
    geoms = ''.join(f'<geom name="part{i}" type="mesh" mesh="p{i}" mass="0" condim="{condim}" friction="{fr}"/>'
                    for i in range(len(world_pieces)))
    q = ' '.join(f'{v:.8f}' for v in _quat_wxyz(Rp))
    pad = lambda n: (f'<geom name="{n}" type="box" size="{fw} {PAD_THICKNESS/2} {fd}" condim="{condim}" '
                     f'friction="{fr}" mass=".1"/>')
    xml = f"""<mujoco><option timestep="{timestep}" gravity="0 0 -9.81" cone="elliptic" impratio="10"
integrator="implicitfast" solver="{solver}"><flag multiccd="enable"/></option>
<default><geom solref=".002 1"/></default>
<asset>{meshes}</asset><worldbody>
<geom name="table" type="plane" pos="0 0 {table_top}" size="2 2 .01" friction="{fr}"/>
<body name="part" pos="{origin[0]} {origin[1]} {origin[2]}"><freejoint/>
<inertial pos="{inertial[0]} {inertial[1]} {inertial[2]}" mass="{mass}"
 diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>{geoms}</body>
<body name="target" mocap="true" pos="{c[0]} {c[1]} {c[2]}" quat="{q}"/>
<body name="palm" pos="{c[0]} {c[1]} {c[2]}" quat="{q}" gravcomp="1"><freejoint/>
<inertial pos="0 0 0" mass=".5" diaginertia=".001 .001 .001"/>
<body name="left" pos="0 {open_half} 0" gravcomp="1"><joint name="left" type="slide" axis="0 -1 0" damping="20"/>{pad('padL')}</body>
<body name="right" pos="0 {-open_half} 0" gravcomp="1"><joint name="right" type="slide" axis="0 1 0" damping="20"/>{pad('padR')}</body>
</body></worldbody>
<equality><weld body1="palm" body2="target" solref=".005 1"/></equality>
<actuator><position joint="left" kp="3000" forcelimited="true" forcerange="-{force} {force}"/>
<position joint="right" kp="3000" forcelimited="true" forcerange="-{force} {force}"/></actuator></mujoco>"""
    return xml, open_half


def _angle(q1, q2):
    """Angle in degrees between two unit quaternions (w, x, y, z)."""
    return float(np.degrees(2*np.arccos(min(1., abs(float(np.dot(q1, q2)))))))


def trial(pieces, R, t, table_top, candidate, gripper, close=True, **settings):
    """One lift-hold-shake trial. ``close=False`` leaves the pads open (a control)."""
    import mujoco
    xml, open_half = scene(pieces, R, t, table_top, candidate, gripper, **settings)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    part, palm = model.body('part').id, model.body('palm').id
    pads = {model.geom('padL').id, model.geom('padR').id}
    part_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == part}
    start_palm = data.mocap_pos[0].copy()
    z0 = data.xipos[part][2]
    target = open_half - PAD_THICKNESS/2 + SQUEEZE
    s = SCHEDULE
    marks = np.cumsum([s['settle'], s['close'], s['grip'], s['lift'], s['hold'], s['shake'], s['rest']])
    dt = model.opt.timestep
    steps = int(round(marks[-1]/dt))
    rel, rel_q, depth = None, None, 0.
    every = max(1, int(round(.01/dt)))
    for i in range(steps):
        time = i*dt
        closing = np.clip((time-marks[0])/s['close'], 0, 1) if close else 0.
        data.ctrl[:] = closing*target
        lift = np.clip((time-marks[2])/s['lift'], 0, 1)
        lift = .10*(3*lift**2-2*lift**3)
        shake = .02*np.sin(2*np.pi*3*(time-marks[4])) if marks[4] <= time < marks[5] else 0.
        data.mocap_pos[0] = start_palm+np.array([shake, 0., lift])
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            return dict(held=False, unstable=True)
        if i % every == 0 and time >= marks[1]:
            for j in range(data.ncon):
                con = data.contact[j]
                if con.geom1 in part_geoms or con.geom2 in part_geoms:
                    depth = max(depth, -float(con.dist))
        if rel is None and time >= marks[3]:
            rel = data.xpos[part]-data.xpos[palm]
            rel_q = _rel_quat(data.xquat[palm], data.xquat[part])
    # v1's measures, taken at the body frame origin (v1's centre of mass) so they mean the same in v2
    rise = float(data.xpos[part][2]-model.body_pos[part][2])
    slip = float(np.linalg.norm((data.xpos[part]-data.xpos[palm])-rel))
    turn = _angle(rel_q, _rel_quat(data.xquat[palm], data.xquat[part]))
    touching = any((data.contact[j].geom1 in pads and data.contact[j].geom2 in part_geoms) or
                   (data.contact[j].geom2 in pads and data.contact[j].geom1 in part_geoms) for j in range(data.ncon))
    com_rise = float(data.xipos[part][2]-z0)
    return dict(held=bool(rise >= .08 and slip < .02), rise=round(rise, 4), slip=round(slip, 4),
                com_rise=round(com_rise, 4), turn_deg=round(turn, 1), touching=bool(touching),
                max_penetration_mm=round(depth*1e3, 3), unstable=False)


def _rel_quat(q_palm, q_part):
    import mujoco
    inv = np.zeros(4); mujoco.mju_negQuat(inv, q_palm)
    out = np.zeros(4); mujoco.mju_mulQuat(out, inv, q_part)
    return out


def evaluate(pieces, R, t, table_top, candidate, gripper, com=None, perturbations=PERTURBATIONS, **settings):
    """Every perturbation for one grasp: {name: trial result}."""
    return {name: trial(pieces, R, t, table_top, candidate, gripper, com=com, **{**settings, **kw})
            for name, kw in perturbations}
