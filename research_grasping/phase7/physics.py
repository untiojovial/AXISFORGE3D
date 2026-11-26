"""Phase VII dynamics (v3): Phase VI's v2 test with a corrected success criterion.

Phase V/VI measured the part's rotation relative to the palm only between the
end of the lift and the end of the trial, and did not use it: a part could
pivot 66 degrees inside the jaws and still count as held. v3 keeps v2's scene
(planner centre of mass, intended masses, same contacts and schedule) and adds:

- turn_from_grip_deg: rotation of the part relative to the palm from the end
  of the grip phase (pads closed, before lifting) to the end of the trial;
- touching: the pads still touch the part at the end;
- held_v3 = rise >= 80 mm and slip < 20 mm and turn_from_grip <= 15 degrees
  and touching and not unstable.

15 degrees was fixed before any per-method result: it is the rotation that
moves a point 80 mm from the closing axis by 20 mm, so the angular tolerance
matches the linear ones the test already uses (80 mm rise, 20 mm slip).
Thresholds of 5, 10 and 20 degrees are reported as sensitivity only.
"""
from __future__ import annotations

import numpy as np

from research_grasping.phase6 import physics as v2

TURN_LIMIT_DEG = 15.0
RISE_MIN = .08
SLIP_MAX = .02
SCHEDULE = v2.SCHEDULE
PAD_THICKNESS = v2.PAD_THICKNESS
SQUEEZE = v2.SQUEEZE


def trial(pieces, R, t, table_top, candidate, gripper, close=True, **settings):
    """One lift-hold-shake trial with v2 measures plus the v3 rotation and contact telemetry."""
    import mujoco
    xml, open_half = v2.scene(pieces, R, t, table_top, candidate, gripper, **settings)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    part, palm = model.body('part').id, model.body('palm').id
    pads = {model.geom('padL').id, model.geom('padR').id}
    part_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == part}
    start_palm = data.mocap_pos[0].copy()
    target = open_half - PAD_THICKNESS/2 + SQUEEZE
    s = SCHEDULE
    marks = np.cumsum([s['settle'], s['close'], s['grip'], s['lift'], s['hold'], s['shake'], s['rest']])
    dt = model.opt.timestep
    steps = int(round(marks[-1]/dt))
    rel, q_lift, q_grip, depth, max_turn = None, None, None, 0., 0.
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
            return dict(held=False, held_v3=False, unstable=True)
        if q_grip is None and time >= marks[2]:
            q_grip = v2._rel_quat(data.xquat[palm], data.xquat[part])
        if i % every == 0 and time >= marks[1]:
            for j in range(data.ncon):
                con = data.contact[j]
                if con.geom1 in part_geoms or con.geom2 in part_geoms:
                    depth = max(depth, -float(con.dist))
            if q_grip is not None:
                max_turn = max(max_turn, v2._angle(q_grip, v2._rel_quat(data.xquat[palm], data.xquat[part])))
        if rel is None and time >= marks[3]:
            rel = data.xpos[part]-data.xpos[palm]
            q_lift = v2._rel_quat(data.xquat[palm], data.xquat[part])
    rise = float(data.xpos[part][2]-model.body_pos[part][2])
    slip = float(np.linalg.norm((data.xpos[part]-data.xpos[palm])-rel))
    q_end = v2._rel_quat(data.xquat[palm], data.xquat[part])
    turn_after_lift = v2._angle(q_lift, q_end)
    turn_from_grip = v2._angle(q_grip, q_end)
    touching = any((data.contact[j].geom1 in pads and data.contact[j].geom2 in part_geoms) or
                   (data.contact[j].geom2 in pads and data.contact[j].geom1 in part_geoms) for j in range(data.ncon))
    held = bool(rise >= RISE_MIN and slip < SLIP_MAX)
    return dict(held=held, held_v3=bool(held and turn_from_grip <= TURN_LIMIT_DEG and touching),
                rise=round(rise, 4), slip=round(slip, 4), turn_deg=round(turn_after_lift, 1),
                turn_from_grip_deg=round(turn_from_grip, 1), max_turn_deg=round(max_turn, 1),
                touching=bool(touching), max_penetration_mm=round(depth*1e3, 3), unstable=False)


def held_at(record, limit_deg):
    """v3 success for another rotation limit (sensitivity only)."""
    return bool(record.get('held') and record.get('touching') and record.get('turn_from_grip_deg', 1e9) <= limit_deg)
