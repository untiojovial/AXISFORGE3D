"""Causal beta-kernel feasibility mixture and target-only feasible quality GP."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.special import logsumexp
from axisforge_bo.models import ExactGP, StationaryKernel
from research_grasping.contextual_transfer.baselines import _ei


@dataclass(frozen=True)
class FeasibilityTask:
    object_id: str
    available_at: float
    context: np.ndarray
    features: np.ndarray
    feasible: np.ndarray
    cost: float


def kernel_probability(query, train_x, labels, *, length_scale, alpha=1., beta=1.):
    """Kernel weighted beta-like smoother; not an exact beta posterior."""
    x, train, y = np.asarray(query,float), np.asarray(train_x,float), np.asarray(labels,float)
    if (x.ndim != 2 or train.ndim != 2 or x.shape[1] != train.shape[1]
            or y.shape != (len(train),) or not np.isfinite(x).all()
            or not np.isfinite(train).all() or not np.isfinite(y).all()
            or not np.isin(y, [0.,1.]).all() or not np.isfinite([length_scale,alpha,beta]).all()
            or length_scale <= 0 or alpha <= 0 or beta <= 0):
        raise ValueError('invalid probability input')
    weights = np.exp(-.5*np.sum(((x[:,None,:]-train[None,:,:])/length_scale)**2,axis=2))
    return (alpha + weights @ y)/(alpha + beta + weights.sum(axis=1))


class FeasibilityPolicy:
    """Single reservation ask/tell; rejects non-train archive and future sources.

    `mode=target` and `mode=adaptive` differ only in use/weighting of sources;
    quality model and initial candidate IDs are otherwise identical.
    """
    def __init__(self, ids, features, context, history, *, manifest_splits, manifest_digest,
                 target_id, target_time, scaler, budget, initial_ids, mode='adaptive',
                 length_scale=1., noise_variance=.01, eta=.2, floor=.1, tau=1.):
        canonical=json.dumps(manifest_splits,sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(canonical).hexdigest()!=manifest_digest or target_id not in manifest_splits:
            raise ValueError('invalid authorized manifest')
        ids=tuple(ids); x=np.asarray(features,float); context=np.asarray(context,float)
        if (mode not in {'target','equal','no_evidence','adaptive'} or not ids or len(set(ids))!=len(ids)
            or any(not isinstance(i,str) or not i for i in ids) or x.ndim!=2 or len(x)!=len(ids)
            or not x.shape[1] or not np.isfinite(x).all() or context.ndim!=1 or not len(context)
            or not np.isfinite(context).all() or type(budget) is not int or not 0<budget<=len(ids)
            or not initial_ids or len(set(initial_ids))!=len(initial_ids) or len(initial_ids)>budget
            or not set(initial_ids)<=set(ids) or not np.isfinite([target_time,eta,floor,tau,noise_variance,length_scale]).all()
            or eta<=0 or tau<=0 or not 0<=floor<1 or noise_variance<=0 or length_scale<=0):
            raise ValueError('invalid policy configuration')
        sources=[]; seen=set()
        for task in history:
            c=np.asarray(task.context,float); y=np.asarray(task.feasible,float); tx=scaler.transform(task.features)
            if (task.object_id in seen or task.object_id==target_id or manifest_splits.get(task.object_id)!='train'
                or not np.isfinite(task.available_at) or task.available_at>=target_time
                or not np.isfinite(task.cost) or task.cost<0 or c.shape!=context.shape or not np.isfinite(c).all()
                or tx.shape[1]!=x.shape[1] or y.shape!=(len(tx),) or not len(tx) or not np.isin(y,[0.,1.]).all()):
                raise ValueError('invalid or unauthorized source')
            sources.append((task.object_id,c.copy(),tx.copy(),y.copy())); seen.add(task.object_id)
        if mode=='target': sources=[]
        self.ids,self.x,self.sources,self.mode=ids,x.copy(),sources,mode
        self.budget,self.initial,self.used,self.pending=budget,tuple(initial_ids),set(),None
        self.records=[]; self.length_scale=length_scale; self.noise=noise_variance
        self.eta,self.floor=eta,floor
        self.evidence=np.zeros(len(sources)+1)
        self.log_prior=np.array([0.]+[0. if mode=='equal' else -np.linalg.norm(c-context)/tau for _,c,_,_ in sources])
        self.obs_x=np.empty((0,x.shape[1]));self.labels=np.empty(0)
        self.success_x=np.empty((0,x.shape[1]));self.success_y=np.empty(0)
        self.kernel=StationaryKernel(length_scale=length_scale)
        self.quality=ExactGP(self.kernel,noise_variance)

    @property
    def weights(self):
        score=self.log_prior+(self.eta*self.evidence if self.mode=='adaptive' else 0.)
        q=np.exp(score-logsumexp(score)); result=(1-self.floor)*q; result[0]+=self.floor
        return result

    def probabilities(self,x):
        parts=[kernel_probability(x,self.obs_x,self.labels,length_scale=self.length_scale)]
        parts.extend(kernel_probability(x,tx,y,length_scale=self.length_scale) for _,_,tx,y in self.sources)
        return np.array(parts)

    def predict(self,x):
        x=np.asarray(x,float); parts=self.probabilities(x)
        mean,var=self.quality.predict(x)
        return self.weights@parts,mean,var

    def ask(self):
        if self.pending is not None: raise RuntimeError('candidate already reserved')
        if len(self.records)>=self.budget: raise StopIteration('budget exhausted')
        available=[i for i,c in enumerate(self.ids) if c not in self.used]
        probabilities, mean, var=self.predict(self.x[available])
        ei=_ei(mean,var,max(self.success_y)) if len(self.success_y) else None
        acquisition=probabilities*ei if ei is not None else probabilities
        if len(self.records)<len(self.initial):
            idx=self.ids.index(self.initial[len(self.records)]); j=available.index(idx)
        else:
            j=int(np.argmax(acquisition)); idx=available[j]
        prediction={'feasibility':float(probabilities[j]),'quality_mean':float(mean[j]),
                    'quality_variance':float(var[j]),'conditional_ei':float(ei[j]) if ei is not None else None,
                    'acquisition':float(acquisition[j]),'weights':self.weights.tolist()}
        self.pending=(idx,prediction)
        return self.ids[idx]

    def tell(self,candidate_id,*,status,cost,utility=None):
        if self.pending is None or candidate_id!=self.ids[self.pending[0]]:
            raise ValueError('candidate not reserved')
        if (status not in {'ok','physical_failure','infrastructure_failure'} or not np.isfinite(cost) or cost<0
            or (status=='infrastructure_failure' and utility is not None)
            or (status!='infrastructure_failure' and (utility is None or not np.isfinite(utility)))):
            raise ValueError('invalid observation')
        idx,prediction=self.pending; px=self.x[idx:idx+1]
        if status!='infrastructure_failure':
            label=float(status=='ok'); probabilities=self.probabilities(px)[:,0]
            logs=np.log(np.clip(probabilities if label else 1-probabilities,1e-12,1))
            new_obs=np.vstack((self.obs_x,px)); new_labels=np.append(self.labels,label)
            if label:
                new_x=np.vstack((self.success_x,px)); new_y=np.append(self.success_y,float(utility))
                quality=ExactGP(self.kernel,self.noise).fit(new_x,new_y)
                self.success_x,self.success_y,self.quality=new_x,new_y,quality
            self.obs_x,self.labels=new_obs,new_labels; self.evidence+=logs
        self.records.append({'candidate_id':candidate_id,'status':status,'utility':utility,
                             'cost':float(cost),'prediction':prediction})
        self.used.add(candidate_id);self.pending=None
