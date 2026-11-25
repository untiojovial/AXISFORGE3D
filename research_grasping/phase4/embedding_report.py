"""Summarize the frozen development panel with objects as the unit of averaging."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
import numpy as np

from research_grasping.phase3.experiment import digest, load, save


def summarize(path):
    report=load(path)
    if report['status']!='COMPLETED' or report['objects']!=22:
        raise ValueError('a full verified development import is required')
    stats={}
    sets={}
    reference={}
    required_methods={'random','adaptive','contextual','target_local'}
    expected_kinds={'geometry','global','local','global_local','random_control'}
    if set(report['results'])!=expected_kinds: raise ValueError('incomplete representations')
    for kind,block in report['results'].items():
        panel=defaultdict(dict)
        for t in block['trajectories']:
            key=(t['object_id'],t['seed'])
            if key in panel[t['method']]:
                raise ValueError('duplicate trajectory')
            if len(t['selected'])!=6 or len(set(t['selected']))!=6 or not np.isfinite(t['auc']):
                raise ValueError('invalid measured trajectory')
            panel[t['method']][key]=t
        if set(panel)!=required_methods:
            raise ValueError('incomplete baseline roster')
        keys=set(panel['random'])
        valid=set(block['predictive'])
        if (len(valid)!=6 or keys!={(n,seed) for n in valid for seed in (11,29,47)}
                or any(set(rs)!=keys for rs in panel.values())):
            raise ValueError('unpaired trajectories')
        if any(scores[key]['selected'][:2]!=panel['random'][key]['selected'][:2]
               for scores in panel.values() for key in keys):
            raise ValueError('initial queries differ')
        if reference and reference!=panel['random']:raise ValueError('random control changed')
        reference=panel['random']
        sets[kind]=panel
        stats[kind]=dict(auc={m:mean(t['auc'] for t in scores.values()) for m,scores in panel.items()},
                         prediction={m:mean(t[m] for t in block['predictive'].values())
                                     for m in next(iter(block['predictive'].values()))})
    control=sets['geometry']['adaptive']
    for kind,panel in sets.items():
        paired=defaultdict(list)
        for key,t in panel['adaptive'].items():
            paired[key[0]].append(t['auc']-control[key]['auc'])
        stats[kind]['adaptive_minus_geometry_by_object']={n:mean(v) for n,v in paired.items()}
    return dict(source_sha256=digest(path),objects=22,validation_objects=6,queries_per_run=6,
                prior_candidates_per_object=10,real_outcome='native-f12',results=stats)


def plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(report['results'])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    y=np.arange(len(labels))
    for method,symbol in [('random','o'),('target_local','s'),('contextual','x'),('adaptive','+')]:
        vals=[report['results'][k]['auc'][method] for k in labels]
        axes[0].scatter(vals,y,label=method,marker=symbol,alpha=.8)
    axes[0].set_yticks(y,labels);axes[0].set_xlabel('Media AUC por objeto');axes[0].legend(fontsize=8)
    for i,label in enumerate(labels):
        vals=list(report['results'][label]['adaptive_minus_geometry_by_object'].values())
        axes[1].scatter(vals,[i]*len(vals),color='#285582',alpha=.48)
    axes[1].axvline(0,color='0.5',lw=.8);axes[1].set_yticks(y,labels)
    axes[1].set_xlabel('Δ AUC adaptativo respecto a geometría')
    fig.suptitle('OpenCLIP congelado · 6 objetos de validación · 3 semillas pareadas')
    fig.savefig(path,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    result=summarize(a.analysis);save(a.out/'summary.json',result);plot(result,a.out/'openclip_panel.png')
