import hashlib
import json
import numpy as np
import pytest
from research_grasping.phase3.transfer_policy import CataloguePolicy, FrozenScale, HistoricalTask, approved_history


def setup():
    scaler = FrozenScale(np.zeros(2), np.ones(2), 'train-hash')
    task = HistoricalTask('train', 1., np.array([.2]), np.array([[0., 1.], [1., 0.]]), np.array([0., 1.]), 4.)
    sources = approved_history([task], {'train':'train', 'target':'validation'}, target_id='target', target_time=2, scaler=scaler)
    return sources


def policy(method):
    manifest = {'train':'train', 'target':'validation'}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    scale = FrozenScale(np.zeros(2), np.ones(2), 'train-hash')
    task = HistoricalTask('train', 1., np.array([.2]), np.array([[0., 1.], [1., 0.]]), np.array([0., 1.]), 4.)
    return CataloguePolicy(['a', 'b', 'c', 'd'], [[0,0],[0,1],[1,0],[1,1]], [0], [task],
                           manifest_splits=manifest, manifest_digest=digest, target_id='target', scaler=scale,
                           target_time=2, method=method, budget=3, seed=5, initial_ids=['a'])


def test_sources_and_temporal_restrictions():
    scale = FrozenScale(np.zeros(2), np.ones(2), 'train-hash')
    task = HistoricalTask('bad', 2, np.array([0]), np.array([[0,0]]), np.array([1]), 0)
    with pytest.raises(ValueError):
        approved_history([task], {'bad':'train', 'target':'test'}, target_id='target', target_time=2, scaler=scale)
    with pytest.raises(ValueError):
        approved_history([task], {'bad':'test', 'target':'test'}, target_id='target', target_time=3, scaler=scale)


@pytest.mark.parametrize('method', ['random','sobol_projected','target','geometry_target','rgpe','equal','adaptive','contextual','multitask_fixed','no_evidence','no_floor','target_local'])
def test_causal_budget_and_reproducibility(method):
    first, second = policy(method), policy(method)
    for t in range(3):
        a, b = first.ask(), second.ask()
        assert a == b and (t != 0 or a == 'a')
        with pytest.raises(RuntimeError): first.ask()
        first.tell(a, status='ok', utility=.2*t, cost=1)
        second.tell(b, status='ok', utility=.2*t, cost=1)
    assert first.records == second.records
    assert len({r['candidate_id'] for r in first.records}) == 3
    with pytest.raises(StopIteration): first.ask()


def test_hidden_future_and_infrastructure_failure():
    a, b = policy('adaptive'), policy('adaptive')
    i = a.ask(); assert i == b.ask()
    a.tell(i, status='infrastructure_failure', cost=2)
    b.tell(i, status='infrastructure_failure', cost=2)
    assert a.ask() == b.ask()  # no unobserved labels passed to either policy
    assert not len(a.model.y)


def test_rejected_tell_preserves_reservation():
    p = policy('adaptive'); candidate = p.ask()
    with pytest.raises(ValueError): p.tell(candidate, status='ok', cost=0, utility=float('nan'))
    p.tell(candidate, status='ok', cost=1, utility=.1)
    assert len(p.records) == 1


def test_cannot_inject_sources_or_false_digest():
    manifest = {'train':'train', 'target':'test'}
    scale = FrozenScale(np.zeros(2), np.ones(2), 'train-hash')
    with pytest.raises(ValueError):
        CataloguePolicy(['a'], [[0,0]], [0], [], manifest_splits=manifest, manifest_digest='0'*64,
                        target_id='target', scaler=scale, target_time=2, method='random', budget=1, seed=0, initial_ids=[])
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    bad = HistoricalTask('target', 1., np.array([0]), np.array([[0,0]]), np.array([1]), 0)
    with pytest.raises(ValueError):
        CataloguePolicy(['a'], [[0,0]], [0], [bad], manifest_splits=manifest, manifest_digest=digest,
                        target_id='target', scaler=scale, target_time=2, method='adaptive', budget=1, seed=0, initial_ids=['a'])


def test_ask_prediction_failure_does_not_consume_initial(monkeypatch):
    p = policy('adaptive')
    original = p.model.predict
    def fail(_):
        raise ArithmeticError('injected')
    monkeypatch.setattr(p.model, 'predict', fail)
    with pytest.raises(ArithmeticError): p.ask()
    assert p.pending is None and p.initial_count == 0
    monkeypatch.setattr(p.model, 'predict', original)
    assert p.ask() == 'a'


@pytest.mark.parametrize('method', ['dinov2_contextual', 'dinov3_contextual', 'visual_fixed'])
def test_visual_methods_reject_unproven_checkpoint(method):
    manifest = {'train':'train', 'target':'validation'}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    scale = FrozenScale(np.zeros(2), np.ones(2), 'train-hash')
    with pytest.raises(ValueError):
        CataloguePolicy(['a'], [[0,0]], [0], [], manifest_splits=manifest, manifest_digest=digest,
                        target_id='target', scaler=scale, target_time=2, method=method, budget=1, seed=0, initial_ids=['a'])
