"""Exploratory, deterministic MuJoCo pinch/lift; all lengths in metres."""

from __future__ import annotations

import math


def evaluate(
    offset_x_m: float = 0.0,
    mass_kg: float = 0.05,
    friction: float = 1.0,
    timestep_s: float = 0.001,
    solver: str = "Newton",
) -> dict:
    """Run a prescribed open-close-lift-hold sequence; no object force is applied."""
    import mujoco
    import numpy as np

    if (
        not all(math.isfinite(v) for v in (offset_x_m, mass_kg, friction, timestep_s))
        or mass_kg <= 0
        or friction < 0
        or timestep_s not in (0.0005, 0.001, 0.002)
        or solver not in ("Newton", "CG")
    ):
        raise ValueError("invalid physical parameters")
    xml = f"""<mujoco><option timestep="{timestep_s}" gravity="0 0 -9.81" solver="{solver}" iterations="100" integrator="Euler"/><default><geom condim="3" margin="0" solref="0.02 1" solimp="0.9 0.95 0.001" friction="{friction} 0 0"/></default><worldbody>
    <geom name="table" type="plane" size="1 1 .1"/>
    <body name="object" pos="0 0 .020"><freejoint/><geom name="object_geom" type="box" size=".02 .02 .02" mass="{mass_kg}"/></body>
    <body name="lift" pos="{offset_x_m} 0 .04"><joint name="lift_joint" type="slide" axis="0 0 1"/><inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
    <body name="left" pos="-.035 0 0"><joint name="left_joint" type="slide" axis="1 0 0"/><geom name="left_geom" type="box" size=".005 .02 .03" mass=".1"/></body>
    <body name="right" pos=".035 0 0"><joint name="right_joint" type="slide" axis="-1 0 0"/><geom name="right_geom" type="box" size=".005 .02 .03" mass=".1"/></body></body></worldbody>
    <actuator><position name="lift_act" joint="lift_joint" kp="1000" forcelimited="true" forcerange="-50 50"/><position name="left_act" joint="left_joint" kp="3000" forcelimited="true" forcerange="-10 10"/><position name="right_act" joint="right_joint" kp="3000" forcelimited="true" forcerange="-10 10"/></actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geometry_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("object_geom", "table", "left_geom", "right_geom")
    }
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    lift_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lift")
    initial_position = data.xpos[object_body_id].copy()
    parameters = {
        "offset_x_m": offset_x_m,
        "mass_kg": mass_kg,
        "friction": friction,
        "timestep_s": timestep_s,
        "solver": solver,
        "mujoco_version": mujoco.__version__,
    }
    initial_table_contact = any(
        {c.geom1, c.geom2} == {geometry_ids["object_geom"], geometry_ids["table"]}
        for c in data.contact
    )
    for c in data.contact:
        finger_contact = geometry_ids["object_geom"] in (c.geom1, c.geom2) and (
            geometry_ids["left_geom"] in (c.geom1, c.geom2)
            or geometry_ids["right_geom"] in (c.geom1, c.geom2)
        )
        if c.dist < -0.001 or finger_contact:
            return {
                "success": False,
                "reason": (
                    "initial_penetration"
                    if c.dist < -0.001
                    else "initial_finger_contact"
                ),
                "lift_m": 0.0,
                "quality": 0.0,
                "relative_drift_m": 0.0,
                "table_contact": initial_table_contact,
                "max_penetration_m": max(0.0, -float(c.dist)),
                "trace": [],
                "parameters": parameters,
            }
    trace = []
    hold_heights = []
    hold_drifts = []
    table_contact = False
    hold_relative_start = None
    max_penetration = 0.0
    total = round(4 / timestep_s)
    for i in range(total):
        time_s = i * timestep_s
        closure_target = min(time_s, 1.0) * 0.011
        lift_target = min(max((time_s - 1.0) / 2.0, 0.0), 1.0) * 0.1
        data.ctrl[:] = [lift_target, closure_target, closure_target]
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        height_above_initial = float(data.xpos[object_body_id, 2] - initial_position[2])
        relative_position = data.xpos[object_body_id] - data.xpos[lift_body_id]
        if time_s >= 3:
            if hold_relative_start is None:
                hold_relative_start = relative_position.copy()
            hold_heights.append(height_above_initial)
            hold_drifts.append(
                float(np.linalg.norm(relative_position - hold_relative_start))
            )
            for c in data.contact:
                if {c.geom1, c.geom2} == {
                    geometry_ids["object_geom"],
                    geometry_ids["table"],
                }:
                    table_contact = True
        for c in data.contact:
            max_penetration = max(max_penetration, -float(c.dist))
        if i % max(1, round(0.01 / timestep_s)) == 0:
            trace.append(
                {
                    "time_s": float(data.time),
                    "object_z_m": float(data.xpos[object_body_id, 2]),
                    "relative_xyz_m": relative_position.tolist(),
                    "actuator_force_n": data.actuator_force.tolist(),
                }
            )
    maximum_drift = max(hold_drifts)
    minimum_hold_height = min(hold_heights)
    success = (
        minimum_hold_height >= 0.08 and not table_contact and maximum_drift <= 0.01
    )
    return {
        "success": success,
        "reason": "success" if success else "lift_criteria_not_met",
        "lift_m": minimum_hold_height,
        "quality": float(np.clip(minimum_hold_height / 0.1, 0, 1)),
        "relative_drift_m": maximum_drift,
        "table_contact": table_contact,
        "max_penetration_m": max_penetration,
        "trace": trace,
        "parameters": parameters,
    }
