"""Phase-IV temporal measurements on the existing project-derived proxy scene."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def simulate(model, data, opening, *, close_jaws=True):
    """Run the fixed 1.7 s schedule and retain whole-cycle contact diagnostics."""
    import mujoco
    object_id = model.body('object').id
    palm_id = model.body('palm').id
    jaws = {model.geom(f'jaw{i}_geom').id: i for i in range(2)}
    hand_geoms = set(jaws) | {model.geom('palm_geom').id}
    object_geoms = set(np.flatnonzero(model.geom_bodyid == object_id).tolist())
    support_id = model.geom('support').id
    dt = float(model.opt.timestep)
    steps = round(1.7 / dt)
    initial_warning_counts = data.warning.number.copy()
    previous_time = float(data.time)
    initial_finite = {name: bool(np.isfinite(getattr(data, name)).all()) for name in ('qpos', 'qvel', 'qacc')}
    if np.any(initial_warning_counts) or not all(initial_finite.values()):
        return {'numerical_valid': False, 'sustained': None, 'eligible_sustained': None,
                'numerical_issue': {'step': -1, 'time_s': previous_time,
                    'warning_counts': initial_warning_counts.tolist(),
                    'initial_warning_counts': initial_warning_counts.tolist(),
                    'finite': initial_finite, 'time_reset': False}}
    initial_com = data.xipos[object_id].copy()
    initial_rotation = data.xmat[object_id].reshape(3, 3).copy()
    initial_overlap = 0.
    for contact in data.contact[:data.ncon]:
        a, b = contact.geom
        if ((a in hand_geoms and b in object_geoms | {support_id}) or
                (b in hand_geoms and a in object_geoms | {support_id})):
            initial_overlap = max(initial_overlap, -float(contact.dist))
    minimum_height = np.inf
    maximum_drift = max_penetration = max_object_penetration = max_force = 0.
    baseline = reference = None
    settle = None
    bilateral = support_steps = hold_steps = 0
    max_force_by_stage = [0., 0., 0.]
    for i in range(steps):
        t = i * dt
        data.ctrl[0] = .06 * np.clip((t - .90) / .40, 0, 1)
        data.ctrl[1] = .06 * np.clip((t - .20) / .35, 0, 1)
        data.ctrl[2:] = opening * (1 - np.clip((t - .55) / .35, 0, 1)) if close_jaws else opening
        mujoco.mj_step(model, data)
        warning_counts = data.warning.number.copy()
        finite = {name: bool(np.isfinite(getattr(data, name)).all()) for name in ('qpos', 'qvel', 'qacc')}
        time_reset = float(data.time) <= previous_time
        if np.any(warning_counts > initial_warning_counts) or not all(finite.values()) or time_reset:
            return {'numerical_valid': False, 'sustained': None, 'eligible_sustained': None,
                    'numerical_issue': {'step': i, 'time_s': float(data.time),
                        'warning_counts': warning_counts.tolist(),
                        'initial_warning_counts': initial_warning_counts.tolist(),
                        'finite': finite, 'time_reset': time_reset}}
        previous_time = float(data.time)
        max_force = max(max_force, float(np.max(np.abs(data.actuator_force))))
        stage = 0 if t < .55 else 1 if t < .9 else 2
        max_force_by_stage[stage] = max(max_force_by_stage[stage], float(np.max(np.abs(data.actuator_force))))
        if i + 1 == round(.2 / dt):
            baseline = float(data.xipos[object_id, 2])
            velocity_index = model.body_dofadr[object_id]
            velocity = data.qvel[velocity_index:velocity_index + 6]
            settle = {'translation_m': float(np.linalg.norm(data.xipos[object_id] - initial_com)),
                      'rotation_rad': float(Rotation.from_matrix(initial_rotation.T @ data.xmat[object_id].reshape(3, 3)).magnitude()),
                      'speed_m_s': float(np.linalg.norm(velocity[:3])), 'angular_speed_rad_s': float(np.linalg.norm(velocity[3:]))}
        if i + 1 == round(.9 / dt):
            reference = data.xipos[object_id].copy() - data.xpos[palm_id]
        forces = np.zeros(2)
        touching_support = False
        for index, contact in enumerate(data.contact[:data.ncon]):
            a, b = contact.geom
            max_penetration = max(max_penetration, -float(contact.dist))
            if a in object_geoms or b in object_geoms:
                max_object_penetration = max(max_object_penetration, -float(contact.dist))
                other = b if a in object_geoms else a
                touching_support |= other == support_id
                if other in jaws:
                    wrench = np.empty(6)
                    mujoco.mj_contactForce(model, data, index, wrench)
                    forces[jaws[other]] += max(0., wrench[0])
        if t >= 1.3:
            hold_steps += 1
            minimum_height = min(minimum_height, float(data.xipos[object_id, 2] - baseline))
            maximum_drift = max(maximum_drift, float(np.linalg.norm(data.xipos[object_id] - data.xpos[palm_id] - reference)))
            bilateral += bool(np.all(forces > .01))
            support_steps += touching_support
    stable = (settle['translation_m'] <= .005 and settle['rotation_rad'] <= .087
              and settle['speed_m_s'] <= .02 and settle['angular_speed_rad_s'] <= .2)
    sustained = (minimum_height >= .04 and maximum_drift <= .02 and bilateral / hold_steps >= .9
                 and support_steps == 0 and initial_overlap <= .001)
    return {'numerical_valid': True, 'warning_counts': data.warning.number.tolist(),
            'sustained': bool(sustained), 'settle_stable': bool(stable),
            'eligible_sustained': bool(stable and sustained), 'settle': settle,
            'initial_hand_penetration_m': initial_overlap, 'min_hold_lift_m': minimum_height,
            'max_hold_relative_displacement_m': maximum_drift,
            'loaded_bilateral_fraction': bilateral / hold_steps, 'table_hold_fraction': support_steps / hold_steps,
            'max_penetration_all_m': max_penetration, 'max_object_penetration_all_m': max_object_penetration,
            'max_actuator_force_n': max_force, 'max_actuator_force_by_stage_n': max_force_by_stage,
            'timestep_s': dt, 'solver': int(model.opt.solver)}
