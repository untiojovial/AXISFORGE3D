"""Exploratory paired hierarchical bootstrap for completed development traces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def trajectory_auc(rows):
    if not rows or any(r['status']=='infrastructure_failure' for r in rows):
        raise ValueError('cannot silently score missing infrastructure observations')
    values=np.array([r['utility'] for r in rows],float)
    if not np.isfinite(values).all():raise ValueError('nonfinite utility')
    return float(np.maximum.accumulate(values).mean())


def compare(a,b,*,seed=2026,draws=10000):
    """Maps (object,seed) to AUC. Resample objects and paired seeds within objects."""
    if set(a)!=set(b) or not a:raise ValueError('unpaired runs')
    objects=sorted({k[0] for k in a});groups=[]
    for obj in objects:
        seeds=sorted(k[1] for k in a if k[0]==obj)
        if not seeds:raise ValueError('missing seeds')
        groups.append(np.array([a[(obj,s)]-b[(obj,s)] for s in seeds]))
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        indices=rng.integers(0,len(groups),len(groups));values=[]
        for i in indices:
            d=groups[i];values.append(float(rng.choice(d,size=len(d),replace=True).mean()))
        samples.append(np.mean(values))
    return {'n_objects':len(objects),'mean_delta':float(np.mean([x.mean() for x in groups])),
            'bootstrap_ci95_percentile':np.quantile(samples,[.025,.975]).tolist(),
            'draws':draws,'seed':seed,
            'by_object':{obj:{'seeds':len(d),'mean_delta':float(d.mean()),'seed_deltas':d.tolist()} for obj,d in zip(objects,groups)}}


def load_method(directory,method):
    result={}
    for path in sorted((directory/'runs').glob('*/*.json')):
        run=json.loads(path.read_text());d=run['definition']
        if d['method']!=method:continue
        if len(run['rows'])!=d['budget'] or run.get('pending') is not None:raise ValueError('incomplete')
        key=(d['object_id'],d['seed'])
        if key in result:raise ValueError('duplicate run')
        result[key]=(trajectory_auc(run['rows']),d)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--v1',type=Path,required=True);p.add_argument('--v2b',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    names=[('v2b_adaptive','v2b_target',a.v2b,a.v2b),('v2b_adaptive','fa_target',a.v2b,a.v1),
           ('v2b_adaptive','adaptive',a.v2b,a.v1),('fa_adaptive','fa_target',a.v1,a.v1)]
    report={}
    for left,right,ld,rd in names:
        l=load_method(ld,left);r=load_method(rd,right)
        if set(l)!=set(r):raise ValueError('unpaired methods')
        for key in l:
            x,y=l[key][1],r[key][1]
            if any(x[k]!=y[k] for k in ('catalogue_digest','manifest_sha256','archive_digest','initial','budget','fidelity')):
                raise ValueError('different protocol or initialization')
        report[left+'_minus_'+right]=compare({k:v[0] for k,v in l.items()},{k:v[0] for k,v in r.items()})
    a.out.mkdir(parents=True,exist_ok=True)
    input_paths=sorted((a.v1/'runs').glob('*/*.json'))+sorted((a.v2b/'runs').glob('*/*.json'))
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    result={'status':'post hoc descriptive; no multiplicity-adjusted inference',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_sha256':hashes,'comparisons':report}
    (a.out/'paired_bootstrap.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(names),figsize=(16,5),sharey=True)
    for ax,(left,right,_,_) in zip(axes,names):
        d=report[left+'_minus_'+right]['by_object'];objects=sorted(d)
        for i,obj in enumerate(objects):
            vals=d[obj]['seed_deltas'];ax.scatter(vals,np.full(len(vals),i),s=14,alpha=.45,color='tab:blue')
            ax.scatter([d[obj]['mean_delta']],[i],marker='D',s=30,color='black')
        ax.axvline(0,color='grey',lw=.8);ax.set_title(left+' − '+right,fontsize=8);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Paired AUC difference')
        ax.set_yticks(range(len(objects)),objects,fontsize=6)
    fig.tight_layout()
    for ext in ('svg','png'):fig.savefig(a.out/('paired_by_object.'+ext),dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
