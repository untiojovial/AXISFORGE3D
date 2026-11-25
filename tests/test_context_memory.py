import pytest
from axisforge_bo.context import ContextMemory


def test_memory_time_group_and_copy():
    m = ContextMemory(1, 1)
    x = [1.0]
    m.append(x, [2], 3, 1, "a")
    x[0] = 99
    m.append([4], [5], 6, 2, "b")
    m.append([7], [8], 9, 3, "c")
    assert m.before(2)[0].decision == (1.0,)
    assert len(m.before(3, excluded_groups={"a"})) == 1
    assert m.before(1) == ()


def test_memory_rejects_invalid_inputs():
    m = ContextMemory(1, 1)
    for t in [None, float("nan"), float("inf"), True, "2"]:
        with pytest.raises(ValueError):
            m.before(t)
    with pytest.raises(ValueError):
        m.append([1, 2], [1], 0, 0, "a")
    with pytest.raises(ValueError):
        m.append([1], [float("nan")], 0, 0, "a")
    with pytest.raises(ValueError):
        m.before(2, excluded_groups="a")


def test_warm_start_allowlist_and_serialization():
    import json

    m = ContextMemory(1, 1)
    m.append([1], [2], 3, 1, "source")
    m.append([2], [2], 4, 1, "target")
    m.append([3], [2], 5, 2, "source")
    restored = ContextMemory.from_dict(json.loads(json.dumps(m.to_dict())))
    assert (
        len(restored.warm_start(2, allowed_groups={"source"}, target_group="target"))
        == 1
    )
    assert (
        restored.warm_start(1, allowed_groups={"source"}, target_group="target") == ()
    )
    with pytest.raises(ValueError):
        restored.warm_start(2, allowed_groups={"target"}, target_group="target")
    with pytest.raises(ValueError):
        ContextMemory.from_dict({"version": 2, "records": []})


def test_adversarial_snapshot_and_invalid_arguments():
    for dims in [(0, 1), (True, 1), (1, -1)]:
        with pytest.raises(ValueError):
            ContextMemory(*dims)
    m = ContextMemory(1, 1)
    for d, c, y, t, g in [
        (None, [1], 1, 1, "a"),
        ([1], [1], 1, 1, ""),
        ([1], [1], float("nan"), 1, "a"),
    ]:
        with pytest.raises(ValueError):
            m.append(d, c, y, t, g)
    with pytest.raises(ValueError):
        m.before(1, excluded_groups={""})
    with pytest.raises(ValueError):
        m.warm_start(1, allowed_groups="source", target_group="target")
    with pytest.raises(ValueError):
        m.warm_start(1, allowed_groups={""}, target_group="target")
    with pytest.raises(ValueError):
        ContextMemory.from_dict(
            {
                "version": 1,
                "decision_dimension": 1,
                "context_dimension": 1,
                "records": [{}],
            }
        )
    m.append([1], [2], 3, 0, "source")
    snapshot = m.to_dict()
    snapshot["records"][0]["decision"][0] = 99
    assert m.before(1)[0].decision == (1.0,)
