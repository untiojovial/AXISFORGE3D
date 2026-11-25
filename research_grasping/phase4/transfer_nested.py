"""Nested, object-held-out train diagnostics for archive-prevalence FA policy."""
import itertools
import time
import numpy as np
from research_grasping.phase3.transfer_policy import FrozenScale
from research_grasping.phase3.experiment import fingerprint
from research_grasping.phase4.feasibility_policy_v2b import PrevalenceFeasibilityPolicy

GRID=tuple(itertools.product((.5,1.,2.),(2.,8.)))


def fold_predict(train,held,length_scale,strength):
    if not train or any(t.object_id==held.object_id for t in train):
        raise ValueError('held-out object in training')
    ids=[t.object_id for t in train]
    if len(set(ids))!=len(ids):raise ValueError('duplicate task')
    train_x=np.concatenate([t.features for t in train]); scale=train_x.std(axis=0);scale[scale<1e-8]=1
    scaler=FrozenScale(train_x.mean(axis=0),scale,fingerprint(ids))
    manifest={t.object_id:'train' for t in train};manifest[held.object_id]='train'
    policy=PrevalenceFeasibilityPolicy(['candidate'],np.zeros((1,len(scale))),held.context,train,
        manifest_splits=manifest,manifest_digest=fingerprint(manifest),target_id=held.object_id,
        target_time=max(t.available_at for t in train)+1,scaler=scaler,budget=1,
        initial_ids=['candidate'],mode='adaptive',length_scale=length_scale,prior_strength=strength)
    predicted=policy.predict(scaler.transform(held.features))[0]
    return predicted,policy.prior_mean


def choose(train):
    if len(train)<2:raise ValueError('need at least two training objects')
    scores=[]
    for length,strength in GRID:
        folds=[]
        for held in train:
            others=[t for t in train if t.object_id!=held.object_id]
            p,_=fold_predict(others,held,length,strength)
            folds.append(float(np.mean((p-held.feasible)**2)))
        scores.append(float(np.mean(folds)))
    best=int(np.argmin(scores))
    return GRID[best],scores


def nested(train):
    if len(train)<3:raise ValueError('need at least three objects')
    start=time.perf_counter(); result=[]
    for held in train:
        rest=[t for t in train if t.object_id!=held.object_id]
        (length,strength),scores=choose(rest)
        p,prior=fold_predict(rest,held,length,strength); y=np.asarray(held.feasible,float)
        clip=np.clip(p,1e-12,1-1e-12)
        result.append({'object_id':held.object_id,'chosen':[length,strength],'inner_brier':scores,
            'prior':prior,'n':len(y),'positive':int(y.sum()),'probabilities':p.tolist(),'labels':y.tolist(),
            'brier':float(np.mean((p-y)**2)),'nll':float(np.mean(-(y*np.log(clip)+(1-y)*np.log1p(-clip)))),
            'constant_half_brier':float(np.mean((.5-y)**2)),
            'prevalence_brier':float(np.mean((prior-y)**2))})
    (length,strength),scores=choose(train)
    bins=[]
    for left,right in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        selected=[(p,y) for obj in result for p,y in zip(obj['probabilities'],obj['labels'])
                  if left<=p and (p<right or right==1 and p<=right)]
        bins.append({'lower':float(left),'upper':float(right),'count':len(selected),
                     'mean_probability':float(np.mean([x[0] for x in selected])) if selected else None,
                     'observed_rate':float(np.mean([x[1] for x in selected])) if selected else None})
    return {'grid':[list(v) for v in GRID],'outer_objects':result,'selected_all_train':[length,strength],
            'all_train_inner_brier':scores,'reliability_bins':bins,'elapsed_s':time.perf_counter()-start}
