import json
import pytest
from axisforge_bo.archive import TaskArchive, TaskRecord


def task(
    task_id="a", group="train", completed_at="2025-01-01T00:00:00Z", embedding=(0, 0)
):
    return TaskRecord(
        task_id,
        group,
        completed_at,
        {"raw": [1]},
        embedding,
        (
            {
                "x": [0],
                "objective": 2,
                "cost_seconds": 1,
                "metrics": {"epsilon": 0.2},
                "uncertainty": 0.1,
                "constraints": {"g": -1},
            },
        ),
        {"engine": "1"},
        {"run": 42},
        {"length_scale": 1},
    )


def test_selection_warm_start_no_leakage(tmp_path):
    archive = TaskArchive(
        [
            task(),
            task("b", embedding=(1, 0)),
            task("held", "test"),
            task("future", completed_at="2026-01-01T00:00:00Z"),
        ]
    )
    selected = archive.select(
        [0.1, 0],
        decision_at="2026-01-01T00:00:00Z",
        allowed_groups={"train", "test"},
        excluded_groups={"test"},
        max_distance=2,
    )
    assert [x.task_id for x in selected.tasks] == ["a", "b"]
    assert (
        sum(selected.weights) == pytest.approx(1)
        and selected.weights[0] > selected.weights[1]
    )
    warm_start = selected.tasks[0].observations[0]
    assert warm_start["objective"] == 2 and warm_start["constraints"]["g"] == -1
    from axisforge_bo.optimizer import BayesianOptimizer
    from axisforge_bo.space import Box
    from axisforge_bo.models import ExactGP, StationaryKernel

    optimizer = BayesianOptimizer(Box([(-1, 1)]), ExactGP(StationaryKernel()))
    optimizer.add_observation(warm_start["x"], warm_start["objective"])
    assert optimizer.best()["value"] == 2
    selected.tasks[0].context["raw"][0] = 9
    warm_start["x"][0] = 9
    assert archive.records[0].context["raw"] == [1] and archive.records[0].observations[
        0
    ]["x"] == [0]
    path = tmp_path / "archive.json"
    archive.save(path)
    loaded = TaskArchive.load(path)
    assert [x.to_dict() for x in loaded.records] == [
        x.to_dict() for x in archive.records
    ]
    assert loaded.select(
        [0, 0],
        decision_at="2027-01-01T00:00:00Z",
        allowed_groups={"train"},
        max_distance=2,
    ).tasks


def test_fallback_and_negative_transfer():
    archive = TaskArchive([task()])
    kwargs = dict(
        decision_at="2026-01-01T00:00:00Z", allowed_groups={"train"}, max_distance=0.1
    )
    assert archive.select([9, 9], **kwargs).reason == "no_eligible_task_within_support"
    assert TaskArchive().select([0, 0], **kwargs).tasks == ()
    assert (
        archive.select(
            [0, 0], paired_validation_losses=[[2, 1], [3, 1]], **kwargs
        ).reason
        == "negative_transfer_rejected"
    )
    assert (
        archive.select([0, 0], paired_validation_losses=[[1, 2]], **kwargs).reason
        == "selected"
    )
    assert (
        archive.select(
            [0, 0],
            decision_at="2025-01-01T00:00:00Z",
            allowed_groups={"train"},
            max_distance=1,
        ).tasks
        == ()
    )


def test_invalid_archive_and_records(tmp_path):
    a = TaskArchive([task()])
    with pytest.raises(ValueError):
        a.add(task())
    with pytest.raises(ValueError):
        task(completed_at="2025-01-01")
    with pytest.raises(ValueError):
        task(embedding=(float("nan"),))
    with pytest.raises(ValueError):
        a.select(
            [0, 0],
            decision_at="2026-01-01T00:00:00Z",
            allowed_groups="train",
            max_distance=1,
        )
    with pytest.raises(ValueError):
        a.select(
            [0, 0],
            decision_at="2026-01-01T00:00:00Z",
            allowed_groups={"train"},
            max_distance=-1,
        )
    with pytest.raises(ValueError):
        a.select(
            [0, 0],
            decision_at="2026-01-01T00:00:00Z",
            allowed_groups={"train"},
            max_distance=1,
            paired_validation_losses=[],
        )
    path = tmp_path / "bad"
    for content in (
        "{",
        '{"version":9,"tasks":[]}',
        '{"version":1,"tasks":[{}]}',
        '{"version":1,"tasks":[],"x":NaN}',
    ):
        path.write_text(content)
        with pytest.raises((ValueError, TypeError)):
            TaskArchive.load(path)
    with pytest.raises(FileNotFoundError):
        TaskArchive.load(tmp_path / "missing")


def test_input_alias_isolated():
    source = {"raw": [1]}
    obs = [{"x": [1], "cost_seconds": 1}]
    rec = TaskRecord("a", "g", "2025-01-01T00:00:00Z", source, (0,), obs, {}, {}, {})
    archive = TaskArchive([rec])
    source["raw"][0] = 3
    obs[0]["x"][0] = 3
    rec.context["raw"][0] = 4
    assert archive.records[0].context["raw"] == [1] and archive.records[0].observations[
        0
    ]["x"] == [1]


def test_reject_observation_after_completion():
    with pytest.raises(ValueError):
        TaskRecord(
            "x",
            "g",
            "2025-01-01T00:00:00Z",
            {},
            (0,),
            ({"observed_at": "2026-01-01T00:00:00Z"},),
            {},
            {},
            {},
        )
