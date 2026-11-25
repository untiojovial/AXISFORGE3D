import json
import numpy as np
import pytest
from axisforge_bo.space import Box
from axisforge_bo.models import ExactGP, StationaryKernel
from axisforge_bo.optimizer import BayesianOptimizer, Candidate
from axisforge_bo.evaluation import Evaluation, record


def make(**kw):
    return BayesianOptimizer(
        Box([(-1, 1)]),
        ExactGP(StationaryKernel()),
        seed=7,
        n_initial=1,
        pool_size=16,
        **kw,
    )


def test_pending_checkpoint_resume(tmp_path):
    o = make()
    a = o.ask()
    o.tell(a, a.x[0] ** 2)
    p = o.ask()
    path = tmp_path / "run.json"
    o.save_checkpoint(path)
    restored = BayesianOptimizer.load_checkpoint(path)
    assert restored.pending == p
    o.tell(p, 0.25)
    restored.tell(p, 0.25)
    assert o.ask() == restored.ask()
    assert o.best() == restored.best()


def test_context_and_failures():
    o = make(context_dimension=1, maximize=True)
    a = o.ask([1])
    o.tell(a, 3)
    assert o.best([2]) is None
    b = o.ask([2])
    record(o, b, Evaluation("infeasible", detail="constraint"))
    assert o.best([2]) is None and len(o.records) == 2
    with pytest.raises(ValueError):
        o.ask([float("nan")])
    assert o.pending is None
    c = o.ask([1])
    with pytest.raises(ValueError):
        o.tell(Candidate(c.id, (0.0,), c.context), 1)
    with pytest.raises(ValueError):
        o.tell(c, float("nan"))
    assert o.pending == c
    o.tell(c, 4)
    assert o.best([1])["value"] == 4
    m, v = o.predict([0], [1])
    assert np.isfinite(m) and v >= 0


def test_corrupt_checkpoint(tmp_path):
    o = make()
    p = tmp_path / "x"
    o.save_checkpoint(p)
    state = json.loads(p.read_text())
    state["version"] = 999
    p.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        BayesianOptimizer.load_checkpoint(p)
    p.write_text('{"version": NaN}')
    with pytest.raises(ValueError):
        BayesianOptimizer.load_checkpoint(p)


def test_best_cannot_modify_observations():
    o = make()
    a = o.ask()
    o.tell(a, 1)
    result = o.best()
    result["x"][0] = 42
    result["context"].append(99)
    assert o.best()["x"] == list(a.x) and o.best()["context"] == []


@pytest.mark.parametrize("sampler", ["random", "sobol", "lhs"])
@pytest.mark.parametrize("acquisition", ["ei", "pi", "lcb"])
def test_warm_start_samplers_acquisitions_and_budget(tmp_path, sampler, acquisition):
    o = BayesianOptimizer(
        Box([(-1, 1)]),
        ExactGP(StationaryKernel()),
        seed=4,
        n_initial=1,
        pool_size=8,
        max_evaluations=2,
        sampler=sampler,
        acquisition=acquisition,
    )
    o.add_observation([0], 1)
    a = o.ask()
    o.tell(a, 0)
    with pytest.raises(RuntimeError):
        o.ask()
    path = tmp_path / "state.json"
    o.save_checkpoint(path)
    loaded = BayesianOptimizer.load_checkpoint(path)
    assert loaded.sampler == sampler and loaded.acquisition == acquisition
    with pytest.raises(RuntimeError):
        loaded.ask()


def test_stop_callback_and_history_are_detached():
    o = BayesianOptimizer(
        Box([(-1, 1)]), ExactGP(StationaryKernel()), stop=lambda history: bool(history)
    )
    o.add_observation([0], 1)
    o.records[0]["x"][0] = 99
    assert o.best()["x"] == [0]
    with pytest.raises(RuntimeError):
        o.ask()


def test_mock_evaluator_never_turns_failure_into_objective():
    from axisforge_bo.evaluation import MockEvaluator

    o = make()
    candidate = o.ask()
    record(o, candidate, MockEvaluator(status="timeout").evaluate(candidate))
    assert o.best() is None and o.records[0]["status"] == "timeout"


def test_checkpoint_rejects_budget_corruption_and_duplicate(tmp_path):
    o = make()
    a = o.ask()
    o.tell(a, 1)
    o.ask()
    path = tmp_path / "s"
    o.save_checkpoint(path)
    state = json.loads(path.read_text())
    state["max_evaluations"] = 1
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        BayesianOptimizer.load_checkpoint(path)
    state["max_evaluations"] = 2
    state["pending"][0]["x"] = state["records"][0]["x"]
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        BayesianOptimizer.load_checkpoint(path)


def test_batch_checkpoint_and_out_of_order(tmp_path):
    o = make()
    o.add_observation([0], 1)
    a, b, c = o.ask_batch(3)
    assert len({a.x, b.x, c.x}) == 3
    path = tmp_path / "batch"
    o.save_checkpoint(path)
    r = BayesianOptimizer.load_checkpoint(path)
    assert r.pending_candidates == (a, b, c)
    for obj in (o, r):
        obj.tell(c, 3)
        obj.fail(a, "timeout")
        obj.tell(b, 2)
    assert o.ask() == r.ask()


def test_batch_rolls_back_after_budget_failure():
    o = BayesianOptimizer(
        Box([(-1, 1)]), ExactGP(StationaryKernel()), seed=5, max_evaluations=1
    )
    control = BayesianOptimizer(
        Box([(-1, 1)]), ExactGP(StationaryKernel()), seed=5, max_evaluations=1
    )
    with pytest.raises(RuntimeError):
        o.ask_batch(2)
    assert not o.pending_candidates and o.ask() == control.ask()


def test_mixed_space_checkpoint(tmp_path):
    from axisforge_bo.space import MixedSpace, Real, Integer, Categorical

    space = MixedSpace([Real(0, 1), Integer(1, 2), Categorical(("a", "b"))])
    o = BayesianOptimizer(space, ExactGP(StationaryKernel()), n_initial=1, seed=2)
    a = o.ask()
    o.tell(a, 1)
    b = o.ask()
    assert space.validate(b.x) == b.x
    path = tmp_path / "mixed"
    o.save_checkpoint(path)
    r = BayesianOptimizer.load_checkpoint(path)
    assert r.pending == b
    o.tell(b, 2)
    r.tell(b, 2)
    assert o.ask() == r.ask()


def test_concurrent_reservations_and_checkpoint(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    o = make()
    path = tmp_path / "parallel"
    with ThreadPoolExecutor(max_workers=8) as pool:
        candidates = list(pool.map(lambda _: o.ask(), range(12)))
        list(pool.map(lambda _: o.save_checkpoint(path), range(8)))
    assert len({c.id for c in candidates}) == 12
    assert len(BayesianOptimizer.load_checkpoint(path).pending_candidates) == 12


def test_reject_vector_noise_for_optimizer():
    with pytest.raises(ValueError):
        BayesianOptimizer(
            Box([(0, 1)]), ExactGP(StationaryKernel(), noise_variance=[0.1])
        )


def test_registered_acquisition_roundtrip(tmp_path):
    from axisforge_bo.optimizer import register_acquisition

    register_acquisition("test_prefer_high_mean", lambda mean, var, best: mean, "max")
    o = BayesianOptimizer(
        Box([(-1, 1)]),
        ExactGP(StationaryKernel()),
        n_initial=1,
        acquisition="test_prefer_high_mean",
    )
    o.add_observation([0], 1)
    candidate = o.ask()
    path = tmp_path / "custom"
    o.save_checkpoint(path)
    loaded = BayesianOptimizer.load_checkpoint(path)
    assert loaded.pending == candidate and loaded.acquisition == "test_prefer_high_mean"


def test_batch_constant_liar_refit_and_context():
    o = BayesianOptimizer(
        Box([(-1, 1)]),
        ExactGP(StationaryKernel()),
        context_dimension=1,
        n_initial=1,
        pool_size=16,
    )
    o.add_observation([0], 1, [1])
    a, b = o.ask_batch(2, [1])
    assert a.x != b.x
    # Previously unseen context has no incumbent or lie; it remains valid.
    c = o.ask([2])
    o.tell(b, 2)
    o.tell(c, 3)
    o.tell(a, 4)
    assert o.best([1])["value"] == 1 and o.best([2])["value"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(context_dimension=-1),
        dict(n_initial=0),
        dict(pool_size=0),
        dict(maximize=1),
        dict(max_evaluations=0),
        dict(acquisition="missing"),
        dict(sampler="missing"),
        dict(stop=3),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BayesianOptimizer(Box([(0, 1)]), ExactGP(StationaryKernel()), **kwargs)


def test_registry_and_sampling_validation(tmp_path):
    from axisforge_bo.optimizer import register_acquisition
    from axisforge_bo.space import MixedSpace, Integer

    with pytest.raises(ValueError):
        register_acquisition("ei", lambda *x: 0, "min")
    with pytest.raises(ValueError):
        register_acquisition("bad", 1, "min")
    with pytest.raises(TypeError):
        BayesianOptimizer(None, ExactGP(StationaryKernel()))
    with pytest.raises(ValueError):
        BayesianOptimizer(
            MixedSpace([Integer(0, 1)]), ExactGP(StationaryKernel()), sampler="sobol"
        )
    o = make()
    with pytest.raises(ValueError):
        o.ask_batch(0)
    a = o.ask()
    with pytest.raises(RuntimeError):
        o.add_observation([0], 2)
    with pytest.raises(ValueError):
        o.fail(a, "unknown")
    o.fail(a, "error")
    with pytest.raises(ValueError):
        o.fail(a)
    with pytest.raises(ValueError):
        o.add_observation([0], np.nan)
    o.add_observation([0], 2)
    with pytest.raises(ValueError):
        o.add_observation([0], 2)
    with pytest.raises(TypeError):
        BayesianOptimizer(
            Box([(0, 1)]), ExactGP(StationaryKernel()), stop=lambda _: False
        ).save_checkpoint(tmp_path / "x")


def test_invalid_acquisition_rolls_back_rng():
    from axisforge_bo.optimizer import register_acquisition

    register_acquisition("bad_shape_for_test", lambda m, v, b: [1], "max")
    o = BayesianOptimizer(
        Box([(0, 1)]),
        ExactGP(StationaryKernel()),
        n_initial=1,
        pool_size=8,
        acquisition="bad_shape_for_test",
    )
    o.add_observation([0], 1)
    state = o.rng.bit_generator.state
    with pytest.raises(ValueError):
        o.ask()
    assert o.rng.bit_generator.state == state and not o.pending_candidates


def test_checkpoint_validation_variants(tmp_path):
    o = make()
    a = o.ask()
    o.tell(a, 1)
    o.ask()
    path = tmp_path / "x"
    o.save_checkpoint(path)
    state = json.loads(path.read_text())
    for key, value in [("records", {}), ("pending", 3), ("next_id", 0)]:
        altered = dict(state)
        altered[key] = value
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            BayesianOptimizer.load_checkpoint(path)
    altered = dict(state)
    altered["space"] = {"type": "unknown"}
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):
        BayesianOptimizer.load_checkpoint(path)


def test_evaluation_validation():
    from axisforge_bo.evaluation import Evaluation, MockEvaluator, record

    for args in [("wrong",), ("success", None), ("success", True), ("error", 1)]:
        with pytest.raises(ValueError):
            Evaluation(*args)
    with pytest.raises(ValueError):
        Evaluation("error", detail=4)
    with pytest.raises(ValueError):
        Evaluation("success", 1, cost_seconds=-1)
    with pytest.raises(TypeError):
        MockEvaluator().evaluate(None)
    o = make()
    a = o.ask()
    with pytest.raises(TypeError):
        record(o, a, None)
    record(o, a, MockEvaluator(value=2).evaluate(a))
    assert o.best()["value"] == 2


def test_acquisition_registry_single_source_and_log_ei(tmp_path):
    from axisforge_bo.acquisitions import register_acquisition as register_core
    from axisforge_bo.optimizer import register_acquisition as register_optimizer

    assert register_core is register_optimizer
    register_core("shared_registry_test", lambda mean, variance, best: mean, "min")
    for name in ("shared_registry_test", "log_ei"):
        o = BayesianOptimizer(
            Box([(0, 1)]), ExactGP(StationaryKernel()), n_initial=1, acquisition=name
        )
        o.add_observation([0.5], 1)
        candidate = o.ask()
        path = tmp_path / name
        o.save_checkpoint(path)
        resumed = BayesianOptimizer.load_checkpoint(path)
        assert resumed.pending == candidate and resumed.acquisition == name


def test_reserved_acquisition_names_and_invalid_registration():
    from axisforge_bo.acquisitions import ACQUISITIONS, register_acquisition

    before = dict(ACQUISITIONS)
    for name in ("ei", "pi", "lcb", "log_ei"):
        with pytest.raises(ValueError):
            register_acquisition(name, lambda m, v, b: m, "max")
    with pytest.raises(ValueError):
        register_acquisition("invalid_direction", lambda m, v, b: m, "sideways")
    assert ACQUISITIONS == before


def test_evaluation_metadata_roundtrips_for_success_and_failure(tmp_path):
    o = make()
    a, b = o.ask_batch(2)
    record(
        o,
        a,
        Evaluation(
            "success", 2, units="metres", cost_seconds=0.5, provenance="sha256:abc"
        ),
    )
    record(
        o,
        b,
        Evaluation(
            "timeout", detail="deadline", cost_seconds=1, provenance="sha256:def"
        ),
    )
    path = tmp_path / "metadata"
    o.save_checkpoint(path)
    loaded = BayesianOptimizer.load_checkpoint(path)
    assert loaded.records == o.records
    assert loaded.records[0]["metadata"]["units"] == "metres"
    assert loaded.records[1]["metadata"]["cost_seconds"] == 1
    state = json.loads(path.read_text())
    state["records"][1]["metadata"]["cost_seconds"] = -1
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        BayesianOptimizer.load_checkpoint(path)


def test_invalid_metadata_does_not_consume_reservation():
    o = make()
    candidate = o.ask()
    for metadata in (
        {"extra": 1},
        {"units": 42},
        {"cost_seconds": True},
        {"cost_seconds": -1},
    ):
        with pytest.raises(ValueError):
            o.tell(candidate, 1, metadata=metadata)
        assert o.pending == candidate
