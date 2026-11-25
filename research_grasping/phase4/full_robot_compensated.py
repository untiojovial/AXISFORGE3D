"""Exploratory exact-model bias compensation for isolated UR5 control."""
from __future__ import annotations
import time
import numpy as np
from research_grasping.phase4.full_robot_ur5 import JOINTS, INITIAL, load_model


def run(path, dt=.001):
    import mujoco
    start = time.perf_counter()
    model, limits = load_model(path)
    model.opt.timestep = dt
    model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    data = mujoco.MjData(model)
    ids = [model.joint(name).id for name in JOINTS]
    qadr = np.asarray([model.jnt_qposadr[j] for j in ids])
    dadr = np.asarray([model.jnt_dofadr[j] for j in ids])
    data.qpos[qadr] = INITIAL
    mujoco.mj_forward(model, data)
    trace = []
    all_errors = []
    saturated_count = 0
    for i in range(round(2/dt)):
        mujoco.mj_forward(model, data)
        target = INITIAL.copy()
        target[0] += .1 * min(i*dt, 1.)
        bias = data.qfrc_bias[dadr].copy()
        torque = np.clip(bias + 100*(target-data.qpos[qadr]) - 20*data.qvel[dadr], -limits, limits)
        data.qfrc_applied[dadr] = torque
        mujoco.mj_step(model, data)
        all_errors.append(np.abs(target - data.qpos[qadr]).copy())
        saturated_count += int(np.count_nonzero(np.abs(torque) >= limits - 1e-10))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ArithmeticError('nonfinite state')
        if i % round(.01/dt) == 0:
            trace.append({'time_s':float(data.time),'q_rad':data.qpos[qadr].tolist(),
                          'target_rad':target.tolist(),'velocity_rad_s':data.qvel[dadr].tolist(),
                          'torque_nm':torque.tolist(),'bias_nm':bias.tolist(),'energy_j':data.energy.tolist()})
    errors = np.asarray(all_errors)
    saturation_fraction = saturated_count / (len(all_errors) * len(limits))
    return {'dt_s':dt,'controller':'PD_plus_exact_model_bias', 'gravity_m_s2':model.opt.gravity.tolist(),
            'effort_limits_nm':limits.tolist(),'max_error_rad':float(errors.max()),
            'final_error_rad':float(errors[-1].max()),'saturation_fraction':saturation_fraction,
            'tracking_criterion':bool(errors.max()<=.2 and errors[-1].max()<=.1),
            'elapsed_s':time.perf_counter()-start,'trace':trace}
