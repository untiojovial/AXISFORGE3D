"""UR5 isolated gravity/PD diagnostic from the repository URDF."""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

JOINTS = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
INITIAL = np.array([0., -1., 1., 0., 0., 0.])


def load_model(path):
    import mujoco
    root = ET.parse(path).getroot()
    limits = []
    for name in JOINTS:
        joint = next(j for j in root.findall('joint') if j.get('name') == name)
        child = joint.find('child').get('link')
        link = next(l for l in root.findall('link') if l.get('name') == child)
        mass = float(link.find('inertial/mass').get('value'))
        inertia = link.find('inertial/inertia')
        diagonal = np.array([float(inertia.get(k)) for k in ('ixx','iyy','izz')])
        if mass <= 0 or not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
            raise ValueError(f'invalid inertial parameters: {child}')
        limits.append(float(joint.find('limit').get('effort')))
    for link in root.findall('link'):
        for geom in list(link):
            if geom.tag in ('visual', 'collision'):
                link.remove(geom)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    model.opt.gravity[:] = [0., 0., -9.81]
    return model, np.asarray(limits)


def run(path, dt=.001, controlled=True):
    import mujoco
    model, limits = load_model(path)
    model.opt.timestep = dt
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    data = mujoco.MjData(model)
    jids = [model.joint(name).id for name in JOINTS]
    qadr = np.asarray([model.jnt_qposadr[j] for j in jids])
    dadr = np.asarray([model.jnt_dofadr[j] for j in jids])
    data.qpos[qadr] = INITIAL
    mujoco.mj_forward(model, data)
    trace = []
    all_errors = []
    saturated_count = 0
    for i in range(round(2 / dt)):
        t = i * dt
        target = INITIAL.copy()
        target[0] += .1 * min(t, 1.)
        torque = np.clip(100 * (target - data.qpos[qadr]) - 20 * data.qvel[dadr], -limits, limits) if controlled else np.zeros(6)
        data.qfrc_applied[dadr] = torque
        mujoco.mj_step(model, data)
        all_errors.append(np.abs(target - data.qpos[qadr]).copy())
        saturated_count += int(np.count_nonzero(np.abs(torque) >= limits - 1e-10))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ArithmeticError('nonfinite dynamic state')
        if i % round(.01 / dt) == 0:
            trace.append({'time_s':float(data.time),'q_rad':data.qpos[qadr].tolist(),
                          'target_rad':target.tolist(),'velocity_rad_s':data.qvel[dadr].tolist(),'torque_nm':torque.tolist(),
                          'energy_j':data.energy.tolist()})
    errors = np.asarray(all_errors)
    saturation_fraction = saturated_count / (len(all_errors) * len(limits))
    return {'dt_s':dt,'controlled':controlled,'gravity_m_s2':model.opt.gravity.tolist(),
            'effort_limits_nm':limits.tolist(),'max_error_rad':float(errors.max()),
            'final_error_rad':float(errors[-1].max()), 'saturation_fraction':saturation_fraction,
            'tracking_criterion':bool(errors.max() <= .2 and errors[-1].max() <= .1), 'trace':trace}
