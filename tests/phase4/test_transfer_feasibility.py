import hashlib
import json
import numpy as np
import pytest
from research_grasping.phase3.transfer_policy import FrozenScale
from research_grasping.phase4.feasibility_policy import FeasibilityPolicy, FeasibilityTask, kernel_probability


def make(mode='adaptive'):
    manifest={'src':'train','target':'validation'}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    scale=FrozenScale(np.zeros(1),np.ones(1),'train')
    task=FeasibilityTask('src',1.,np.array([0.]),np.array([[0.],[1.]]),np.array([0.,1.]),2.)
    return FeasibilityPolicy(['a','b','c'],[[0.],[.5],[1.]],[0.],[task],manifest_splits=manifest,
        manifest_digest=digest,target_id='target',target_time=2.,scaler=scale,budget=3,initial_ids=['a'],mode=mode)


def test_weighted_kernel_and_prequential():
    assert kernel_probability([[0]],np.empty((0,1)),[],length_scale=1)[0] == .5
    assert kernel_probability([[0]],[[0]],[1],length_scale=1)[0] == pytest.approx(2/3)
    p=make(); expected=p.probabilities([[0]])[:,0]; picked=p.ask()
    before=p.weights.copy();p.tell(picked,status='ok',utility=.8,cost=1)
    assert np.allclose(p.evidence,np.log(expected))
    assert np.isclose(p.weights.sum(),1) and p.weights[0]>=.1
    assert not np.array_equal(before,p.weights)
    assert len(p.success_y)==1


def test_no_success_and_infrastructure():
    p=make();first=p.ask();p.tell(first,status='infrastructure_failure',utility=None,cost=1)
    assert not len(p.labels) and not len(p.success_y) and np.all(p.evidence==0)
    second=p.ask();p.tell(second,status='physical_failure',utility=0,cost=1)
    assert p.labels.tolist()==[0] and not len(p.success_y)
    third=p.ask();p.tell(third,status='ok',utility=.7,cost=1)
    with pytest.raises(StopIteration):p.ask()


def test_target_control_and_future_isolation():
    a,b=make('target'),make('target')
    assert a.weights.tolist()==[1.] and b.weights.tolist()==[1.]
    i=a.ask();assert i==b.ask()
    a.tell(i,status='ok',utility=.5,cost=1);b.tell(i,status='ok',utility=.5,cost=1)
    assert a.ask()==b.ask()
    assert a.pending[1]==b.pending[1]


def test_reject_future_and_test_source():
    manifest={'src':'test','target':'validation'}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    scale=FrozenScale(np.zeros(1),np.ones(1),'train')
    task=FeasibilityTask('src',1.,np.array([0.]),np.array([[0.]]),np.array([1.]),1.)
    with pytest.raises(ValueError):
        FeasibilityPolicy(['a'],[[0.]],[0.],[task],manifest_splits=manifest,manifest_digest=digest,
            target_id='target',target_time=2.,scaler=scale,budget=1,initial_ids=['a'])
    manifest['src']='train'
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with pytest.raises(ValueError):
        FeasibilityPolicy(['a'],[[0.]],[0.],[task],manifest_splits=manifest,manifest_digest=digest,
            target_id='target',target_time=1.,scaler=scale,budget=1,initial_ids=['a'])


def test_nested_outer_labels_cannot_change_predictions_or_selection():
    from research_grasping.phase4.transfer_nested import nested
    tasks=[FeasibilityTask(str(i),1.,np.array([i/5]),np.array([[i/5],[i/5+.1]]),
                           np.array([float(i%2),0.]),1.) for i in range(4)]
    first=nested(tasks)
    altered=list(tasks)
    altered[0]=FeasibilityTask('0',1.,np.array([0.]),tasks[0].features,np.array([1.,1.]),1.)
    second=nested(altered)
    a=next(x for x in first['outer_objects'] if x['object_id']=='0')
    b=next(x for x in second['outer_objects'] if x['object_id']=='0')
    assert a['chosen']==b['chosen'] and a['probabilities']==b['probabilities'] and a['prior']==b['prior']
