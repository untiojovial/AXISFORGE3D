"""Phase VII articulated pilot: the cell's six-axis arm and jaw gripper in MuJoCo.

v3 (``physics.py``) moves a floating palm along a straight vertical line through a stiff weld. This
module instead builds the project's own robot (links, inertias, revolute joints, the two prismatic
jaws and their collision boxes) as an articulated chain and drives it with the joint values the
planner returned for the grasp:

- settle 0.3 s at ``graspOpen``; close 0.5 s (jaws to ``graspClosed`` minus the v1 squeeze, force
  limited to maxGripForce/2 per jaw, as v2); grip 0.2 s;
- lift 1.0 s: joint-space interpolation (smoothstep) from ``graspClosed`` to the planner's ``lift``;
- hold 0.5 s; shake 0.5 s: the base joint oscillates at 3 Hz so the grasp point moves about +-20 mm
  sideways; rest 0.3 s.

Arm control: computed torque, tau = M_arm (kp e - kd qdot) + bias, clipped at each joint's maxEffort
(120 N m). The part's weight is not in the model the controller uses (it acts through the contacts),
as on a real arm. The arm links have no collision geometry (the planner already proved the arm clear);
the palm housing and the two pads collide. Part, table, contact and friction settings are v2's.

Outcome (v3 criterion adapted to the planner's lift height h, the palm's planned rise):
held_art = part rise >= 0.8 h and slip relative to the palm < 20 mm and rotation from the grip
<= 15 degrees and pads touching and not unstable.
This is a pilot of the articulated level, not a calibrated model of a real arm.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_grasping.phase5 import dynamics as v1
from research_grasping.phase6 import physics as v2

SCHEDULE = v1.SCHEDULE
SQUEEZE = v1.SQUEEZE
TURN_LIMIT_DEG = 15.
OMEGA = 30.  # closed-loop bandwidth of the arm joints, rad/s


def _q(rot):
    return f"{rot['w']} {rot['x']} {rot['y']} {rot['z']}"


def _v(d):
    return f"{d['x']} {d['y']} {d['z']}"


def robot_xml(project, friction_attr, condim=4):
    """Nested MJCF bodies for the project's robot; returns (xml, arm joint names, jaw joint names)."""
    r = project['robot']
    links = {l['id']: l for l in r['links']}
    children = {}
    for j in r['joints']:
        children.setdefault(j['parentLinkId'], []).append(j)
    finger_links = set(r['gripper']['fingerLinkIds'])
    palm_link = r['gripper']['palmLinkId']
    arm, jaws = [], []

    def inertial(link, moving):
        i = link['inertial']
        diag = np.array([i['inertiaDiagonal'][k] for k in 'xyz'], float)
        m = float(i['mass'])
        if moving:
            diag = np.maximum(diag, 1e-6); m = max(m, 1e-3)
        if m <= 0:
            return ''
        return f'<inertial pos="{_v(i["centerOfMass"])}" mass="{m}" diaginertia="{diag[0]} {diag[1]} {diag[2]}"/>'

    def geoms(link):
        out = []
        if link['id'] not in finger_links and link['id'] != palm_link:
            return ''
        for g in link['geometries']:
            if g.get('role', 'both') not in ('collision', 'both') or g['shape']['kind'] != 'box':
                continue
            s = g['shape']['size']
            name = ('pad' + ('L' if link['id'] == r['gripper']['fingerLinkIds'][0] else 'R')) if link['id'] in finger_links \
                else f"palm_{g['name']}"
            out.append(f'<geom name="{name}" type="box" pos="{_v(g["origin"]["translation"])}" quat="{_q(g["origin"]["rotation"])}" '
                       f'size="{s["x"]/2} {s["y"]/2} {s["z"]/2}" condim="{condim}" friction="{friction_attr}" mass="0"/>')
        return ''.join(out)

    def body(joint):
        link = links[joint['childLinkId']]
        name = joint['name'].replace(' ', '_')
        o = joint['origin']
        jx = ''
        if joint['kind'] == 'revolute':
            lim = joint['limits']
            jx = (f'<joint name="{name}" type="hinge" axis="{_v(joint["axis"])}" range="{lim["lower"]} {lim["upper"]}" '
                  f'limited="true" damping="0.5" armature="0.01"/>')
            arm.append(name)
        elif joint['kind'] == 'prismatic':
            lim = joint['limits']
            jx = (f'<joint name="{name}" type="slide" axis="{_v(joint["axis"])}" range="{lim["lower"]} {lim["upper"]}" '
                  f'limited="true" damping="20"/>')
            jaws.append(name)
        inner = ''.join(body(c) for c in children.get(link['id'], []))
        return (f'<body name="{link["name"].replace(" ", "_")}" pos="{_v(o["translation"])}" quat="{_q(o["rotation"])}">'
                f'{inertial(link, True)}{jx}{geoms(link)}{inner}</body>')

    root = links[r['rootLinkId']]
    bp = r['basePose']
    xml = (f'<body name="{root["name"]}" pos="{_v(bp["translation"])}" quat="{_q(bp["rotation"])}">'
           + ''.join(body(c) for c in children.get(root['id'], [])) + '</body>')
    return xml, arm, jaws


def scene(project, pieces, R, t, table_top, mass=.1, friction=.5, com=None, torsion=.005, condim=4, timestep=.001):
    world_pieces = [(p@R.T)+t for p in pieces]
    allv = np.vstack(world_pieces)
    origin = allv.mean(0)
    inertial = np.zeros(3) if com is None else np.asarray(com, float)-origin
    ext = allv.max(0)-allv.min(0)
    inertia = mass/12*np.array([ext[1]**2+ext[2]**2, ext[0]**2+ext[2]**2, ext[0]**2+ext[1]**2])
    fr = f'{friction} {torsion} .0001'
    meshes = ''.join(f'<mesh name="p{i}" vertex="{" ".join(f"{v:.6f}" for v in (p-origin).ravel())}"/>'
                     for i, p in enumerate(world_pieces))
    pgeoms = ''.join(f'<geom name="part{i}" type="mesh" mesh="p{i}" mass="0" condim="{condim}" friction="{fr}"/>'
                     for i in range(len(world_pieces)))
    rob, arm, jaws = robot_xml(project, fr, condim)
    actuators = ''.join(f'<position name="a_{j}" joint="{j}" kp="3000" forcelimited="true" forcerange="-{{force}} {{force}}"/>'
                        for j in jaws)
    xml = f"""<mujoco><compiler angle="radian"/><option timestep="{timestep}" gravity="0 0 -9.81" cone="elliptic" impratio="10"
integrator="implicitfast" solver="Newton"><flag multiccd="enable"/></option>
<default><geom solref=".002 1"/></default>
<asset>{meshes}</asset><worldbody>
<geom name="table" type="plane" pos="0 0 {table_top}" size="2 2 .01" friction="{fr}"/>
<body name="part" pos="{origin[0]} {origin[1]} {origin[2]}"><freejoint/>
<inertial pos="{inertial[0]} {inertial[1]} {inertial[2]}" mass="{mass}"
 diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>{pgeoms}</body>
{rob}</worldbody><actuator>{actuators}</actuator></mujoco>"""
    return xml, arm, jaws


def trial(project, pieces, R, t, table_top, plan, gripper, mass=.1, friction=.5, com=None, torsion=.005):
    """One articulated lift-hold-shake trial of a planner grasp (``plan`` = one entry of plan --json)."""
    import mujoco
    xml, arm, jaws = scene(project, pieces, R, t, table_top, mass=mass, friction=friction, com=com, torsion=torsion)
    xml = xml.replace('{force}', str(gripper['maxGripForce']/2))
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
    pads = {model.geom('padL').id, model.geom('padR').id}
    part_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == part}
    # planned palm rise (kinematics only)
    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos; probe.qpos[qa] = q_closed; mujoco.mj_kinematics(model, probe)
    z_closed = probe.xpos[palm][2]; tcp_r = float(np.linalg.norm(probe.xpos[palm][:2]))
    probe.qpos[qa] = q_lift; mujoco.mj_kinematics(model, probe)
    planned_rise = float(probe.xpos[palm][2] - z_closed)
    jaw_target = np.clip(q_closed[na:] - SQUEEZE, 0, None)
    s = SCHEDULE
    marks = np.cumsum([s['settle'], s['close'], s['grip'], s['lift'], s['hold'], s['shake'], s['rest']])
    dt = model.opt.timestep
    kp, kd = OMEGA**2, 2*OMEGA
    shake_amp = .02/max(tcp_r, .1)
    M = np.zeros((model.nv, model.nv))
    rel = q_grip = None
    track_err, saturated, steps = 0., 0, int(round(marks[-1]/dt))
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
        if q_grip is None and time >= marks[2]:
            q_grip = v2._rel_quat(data.xquat[palm], data.xquat[part])
        if rel is None and time >= marks[3]:
            rel = data.xpos[part]-data.xpos[palm]
    rise = float(data.xpos[part][2]-model.body_pos[part][2])
    slip = float(np.linalg.norm((data.xpos[part]-data.xpos[palm])-rel))
    turn = v2._angle(q_grip, v2._rel_quat(data.xquat[palm], data.xquat[part]))
    touching = any((data.contact[j].geom1 in pads and data.contact[j].geom2 in part_geoms) or
                   (data.contact[j].geom2 in pads and data.contact[j].geom1 in part_geoms) for j in range(data.ncon))
    held = bool(rise >= .8*planned_rise and slip < .02)
    return dict(held=held, held_art=bool(held and turn <= TURN_LIMIT_DEG and touching), rise=round(rise, 4),
                planned_rise=round(planned_rise, 4), slip=round(slip, 4), turn_from_grip_deg=round(turn, 1),
                touching=bool(touching), max_tracking_error_rad=round(track_err, 4),
                torque_saturated_steps=saturated, unstable=False)


def load_project(path):
    return json.loads(Path(path).read_text())['project']
