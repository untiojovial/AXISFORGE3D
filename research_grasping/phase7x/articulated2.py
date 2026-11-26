"""Articulated model, second version (Phase VII extension): the Phase VII pilot plus the checks it lacked.

Changes with respect to research_grasping/phase7/articulated.py (which is not modified):

1. Arm links collide. Every box collision geometry of every robot link is added, so the model sees
   link-table and link-part contacts. Robot geometries do not collide with each other. Any such contact
   is reported as an unexpected collision: the planner proved the arm clear only kinematically.
2. Contact telemetry on the pads, during hold and shake:
   - peak normal force per pad;
   - peak tangential-to-normal ratio, i.e. friction use; values near the friction coefficient mean
     incipient slip;
   - steps in which the jaw actuators run at their force limit (grip-force saturation).
3. Arm tracking:
   - maximum joint tracking error;
   - steps with the computed torque at maxEffort (as before);
   - part-palm relative motion over the whole lift.
4. Timing as a parameter. The schedule (settle, close, grip, lift, hold, shake, rest) can be overridden
   to test sensitivity to closing time and lift speed.
5. Controls:
   - `control='open'` (negative): the jaws never close, and the part must not be lifted;
   - `control='frictionless'` (negative): friction 1e-4, and the part must slip out;
   - `control='zero-g'` (positive): gravity off, and any grasp that closed must hold.

Joint friction is not added. No measured value for the cell's arm exists, and inventing one would add a
parameter without evidence. Damping and armature are the pilot's. This is still an uncalibrated simulation of
the project's arm, not a model of a real robot.
"""
from __future__ import annotations

import numpy as np

from research_grasping.phase5 import dynamics as v1
from research_grasping.phase6 import physics as v2
from research_grasping.phase7 import articulated as art

TURN_LIMIT_DEG = art.TURN_LIMIT_DEG
OMEGA = art.OMEGA


def _link_geoms(project, friction_attr, condim):
    """Collision boxes of the non-gripper links, keyed by link name (MJCF snippets, robot-only collision class)."""
    r = project['robot']
    skip = set(r['gripper']['fingerLinkIds']) | {r['gripper']['palmLinkId']}
    out = {}
    for link in r['links']:
        if link['id'] in skip:
            continue
        gs = []
        for k, g in enumerate(link['geometries']):
            if g.get('role', 'both') not in ('collision', 'both') or g['shape']['kind'] != 'box':
                continue
            s = g['shape']['size']
            gs.append(f'<geom name="link_{link["name"].replace(" ", "_")}_{k}" type="box" pos="{art._v(g["origin"]["translation"])}" '
                      f'quat="{art._q(g["origin"]["rotation"])}" size="{s["x"]/2} {s["y"]/2} {s["z"]/2}" contype="2" conaffinity="0" '
                      f'condim="{condim}" friction="{friction_attr}" mass="0"/>')
        if gs:
            out[link['name'].replace(' ', '_')] = ''.join(gs)
    return out


def scene(project, pieces, R, t, table_top, mass=.1, friction=.5, com=None, torsion=.005, condim=4, timestep=.001, gravity=9.81):
    xml, arm, jaws = art.scene(project, pieces, R, t, table_top, mass=mass, friction=friction, com=com, torsion=torsion,
                               condim=condim, timestep=timestep)
    fr = f'{friction} {torsion} .0001'
    for name, g in _link_geoms(project, fr, condim).items():
        tag = f'<body name="{name}" '
        i = xml.find(tag)
        if i < 0:
            continue
        j = xml.find('>', i) + 1  # MJCF accepts body children in any order: insert right after the opening tag
        xml = xml[:j] + g + xml[j:]
    # collision bits: links (contype 2, conaffinity 0) touch only the table and the part (conaffinity 1|2); pads and
    # palm keep the pilot's defaults (1, 1), so links never touch the gripper or each other
    xml = xml.replace('<geom name="table" ', '<geom name="table" conaffinity="3" ').replace('<geom name="part', '<geom conaffinity="3" name="part')
    xml = xml.replace('gravity="0 0 -9.81"', f'gravity="0 0 {-gravity}"')
    return xml, arm, jaws


def trial(project, pieces, R, t, table_top, plan, gripper, mass=.1, friction=.5, com=None, torsion=.005, schedule=None, control=None):
    import mujoco
    if control == 'frictionless':
        friction = 1e-4
    xml, arm, jaws = scene(project, pieces, R, t, table_top, mass=mass, friction=friction, com=com, torsion=torsion,
                           gravity=0. if control == 'zero-g' else 9.81)
    force_limit = gripper['maxGripForce']/2
    xml = xml.replace('{force}', str(force_limit))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joints = [j['name'].replace(' ', '_') for j in project['robot']['joints'] if j['kind'] in ('revolute', 'prismatic')]
    names = arm + jaws
    assert joints == names, (joints, names)
    q_open, q_closed, q_lift = (np.asarray(plan[k], float) for k in ('graspOpen', 'graspClosed', 'lift'))
    qa = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    da = np.array([model.jnt_dofadr[model.joint(n).id] for n in arm])
    na = len(arm)
    limits = np.array([j['maxEffort'] for j in project['robot']['joints'] if j['kind'] == 'revolute'], float)
    data.qpos[qa] = q_open
    mujoco.mj_forward(model, data)
    part = model.body('part').id
    palm = model.body(next(l['name'] for l in project['robot']['links']
                           if l['id'] == project['robot']['gripper']['palmLinkId']).replace(' ', '_')).id
    pads = {model.geom('padL').id: 'L', model.geom('padR').id: 'R'}
    part_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == part}
    table = model.geom('table').id
    link_geoms = {g for g in range(model.ngeom) if model.geom(g).name.startswith('link_')}
    palm_geoms = {g for g in range(model.ngeom) if model.geom(g).name.startswith('palm_')}
    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos; probe.qpos[qa] = q_closed; mujoco.mj_kinematics(model, probe)
    z_closed = probe.xpos[palm][2]; tcp_r = float(np.linalg.norm(probe.xpos[palm][:2]))
    probe.qpos[qa] = q_lift; mujoco.mj_kinematics(model, probe)
    planned_rise = float(probe.xpos[palm][2] - z_closed)
    jaw_target = q_open[na:] if control == 'open' else np.clip(q_closed[na:] - v1.SQUEEZE, 0, None)
    s = dict(v1.SCHEDULE, **(schedule or {}))
    marks = np.cumsum([s['settle'], s['close'], s['grip'], s['lift'], s['hold'], s['shake'], s['rest']])
    dt = model.opt.timestep
    kp, kd = OMEGA**2, 2*OMEGA
    shake_amp = .02/max(tcp_r, .1)
    M = np.zeros((model.nv, model.nv))
    rel = q_grip = rel0 = None
    track_err, saturated, grip_sat, steps = 0., 0, 0, int(round(marks[-1]/dt))
    peak_n = {'L': 0., 'R': 0.}; peak_mu = 0.; link_hits = set(); palm_table = False; max_rel = 0.
    f6 = np.zeros(6)
    for i in range(steps):
        time = i*dt
        closing = np.clip((time-marks[0])/s['close'], 0, 1)
        data.ctrl[:] = q_open[na:] + closing*(jaw_target - q_open[na:])
        u = np.clip((time-marks[2])/s['lift'], 0, 1); u = 3*u**2-2*u**3
        target = q_closed[:na] + u*(q_lift[:na]-q_closed[:na]) if time >= marks[2] else q_open[:na]
        if marks[4] <= time < marks[5]:
            target = target.copy(); target[0] += shake_amp*np.sin(2*np.pi*3*(time-marks[4]))
        mujoco.mj_fullM(model, M, data.qM)
        Ma = M[np.ix_(da, da)]
        e = target - data.qpos[qa[:na]]
        tau = Ma@(kp*e - kd*data.qvel[da]) + data.qfrc_bias[da]
        saturated += int(np.any(np.abs(tau) > limits))
        data.qfrc_applied[da] = np.clip(tau, -limits, limits)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            return dict(held=False, held_art=False, unstable=True)
        if time >= marks[2]:
            track_err = max(track_err, float(np.abs(e).max()))
            grip_sat += int(np.any(np.abs(data.actuator_force) >= .99*force_limit))
        if q_grip is None and time >= marks[2]:
            q_grip = v2._rel_quat(data.xquat[palm], data.xquat[part])
            rel0 = data.xpos[part]-data.xpos[palm]
        if rel0 is not None:
            max_rel = max(max_rel, float(np.linalg.norm((data.xpos[part]-data.xpos[palm]) - rel0)))
        if rel is None and time >= marks[3]:
            rel = data.xpos[part]-data.xpos[palm]
        for j in range(data.ncon):
            c = data.contact[j]
            g1, g2 = c.geom1, c.geom2
            if (g1 in link_geoms and (g2 in part_geoms or g2 == table)) or (g2 in link_geoms and (g1 in part_geoms or g1 == table)):
                link_hits.add(model.geom(g1 if g1 in link_geoms else g2).name)
            if (g1 in palm_geoms and g2 == table) or (g2 in palm_geoms and g1 == table):
                palm_table = True
            if marks[3] <= time < marks[5]:
                pad = pads.get(g1) if g2 in part_geoms else pads.get(g2) if g1 in part_geoms else None
                if pad:
                    mujoco.mj_contactForce(model, data, j, f6)
                    fn = abs(f6[0]); ft = float(np.hypot(f6[1], f6[2]))
                    peak_n[pad] = max(peak_n[pad], fn)
                    if fn > 1e-3:
                        peak_mu = max(peak_mu, ft/fn)
    rise = float(data.xpos[part][2]-model.body_pos[part][2])
    slip = float(np.linalg.norm((data.xpos[part]-data.xpos[palm])-rel)) if rel is not None else float('nan')
    turn = v2._angle(q_grip, v2._rel_quat(data.xquat[palm], data.xquat[part]))
    touching = any((data.contact[j].geom1 in pads and data.contact[j].geom2 in part_geoms) or
                   (data.contact[j].geom2 in pads and data.contact[j].geom1 in part_geoms) for j in range(data.ncon))
    held = bool(rise >= .8*planned_rise and slip < .02)
    return dict(held=held, held_art=bool(held and turn <= TURN_LIMIT_DEG and touching), rise=round(rise, 4),
                planned_rise=round(planned_rise, 4), slip=round(slip, 4), turn_from_grip_deg=round(turn, 1), touching=bool(touching),
                max_tracking_error_rad=round(track_err, 4), torque_saturated_steps=saturated, grip_force_saturated_steps=grip_sat,
                peak_normal_N=dict((k, round(v, 2)) for k, v in peak_n.items()), peak_friction_use=round(peak_mu, 3),
                max_relative_motion_mm=round(1e3*max_rel, 2), unexpected_link_contacts=sorted(link_hits), palm_table_contact=palm_table,
                unstable=False)
