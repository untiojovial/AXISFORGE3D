"""Analytic and negative controls for the frozen exploratory physics protocol."""

import pytest

mujoco = pytest.importorskip("mujoco")
from research_grasping.simulation.mujoco_pilot import evaluate


def test_freefall_analytic_and_rest():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".001"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".02" mass=".05"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[2] - (1 - 0.5 * 9.81 * 0.1**2)) < 0.003
    for _ in range(3000):
        mujoco.mj_step(model, data)
    assert abs(data.qvel[2]) < 0.01
    assert abs(data.qpos[2] - 0.02) < 0.003


def test_positive_and_negative_controls():
    positive = evaluate()
    assert positive["success"] and positive["lift_m"] >= 0.08
    for params in ({"offset_x_m": 0.15}, {"friction": 0}, {"mass_kg": 5}):
        assert not evaluate(**params)["success"]
    assert not evaluate(offset_x_m=0.015)["success"]  # starts intersecting a finger
    with pytest.raises(ValueError):
        evaluate(mass_kg=-1)


def test_reproducibility():
    assert evaluate() == evaluate()


def test_recorded_sensitivity_runs():
    import json
    from pathlib import Path

    artifact = (
        Path(__file__).resolve().parents[2] / "research/results/mujoco_sensitivity.json"
    )
    recorded = json.loads(artifact.read_text(encoding="utf-8"))
    assert len(recorded) == 6
    for solver in ("Newton", "CG"):
        for timestep in (0.0005, 0.001, 0.002):
            expected = recorded[f"{solver}_{timestep}"]
            actual = evaluate(solver=solver, timestep_s=timestep)
            assert actual["success"] == expected["success"]
            assert actual["lift_m"] == pytest.approx(expected["lift_m"], abs=1e-5)
            assert actual["relative_drift_m"] == pytest.approx(
                expected["relative_drift_m"], abs=1e-5
            )
            assert len(actual["trace"]) == len(expected["trace"])
            assert (
                max(
                    abs(a["object_z_m"] - b["object_z_m"])
                    for a, b in zip(actual["trace"], expected["trace"])
                )
                < 1e-5
            )
