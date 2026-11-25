"""Descriptive counterfactual screening on a completed phase-IV physical pilot."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import math
import numpy as np
from scipy.stats import pearsonr, spearmanr
from research_grasping.phase3.experiment import digest, load, save


def ratio(n,d):return n/d if d else None


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return {'n':len(x),'pearson':None,'spearman':None,'reason':'insufficient pairs or constant variable'}
    return {'n':len(x),'pearson':float(pearsonr(x,y).statistic),'spearman':float(spearmanr(x,y).statistic),'reason':None}


def confusion(pred,truth):
    pairs=[(p,y) for p,y in zip(pred,truth) if p is not None and y is not None]
    tp=sum(p and y for p,y in pairs);fp=sum(p and not y for p,y in pairs)
    fn=sum(not p and y for p,y in pairs);tn=sum(not p and not y for p,y in pairs)
    return {'n':len(pairs),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn)}


def classify_numerics(record):
    """Keep nominal and each solver outcome distinct, including invalid cases."""
    f3=record.get('f3') or {};nominal=f3.get('nominal') or {};m=nominal.get('metrics') or {}
    valid=m.get('numerical_valid') is True
    eligible=m.get('eligible_sustained') if valid and isinstance(m.get('eligible_sustained'),bool) else None
    raw=m.get('sustained') if valid and isinstance(m.get('sustained'),bool) else None
    labels=[];invalid=0
    for q in f3.get('numerics',[]):
        a=q.get('metrics') or {};label=a.get('eligible_sustained')
        if a.get('numerical_valid') is True and isinstance(label,bool):labels.append(label)
        else:invalid+=1
    return {'eligible':eligible,'raw':raw,'nominal_valid':valid,
            'numeric_changed':any(x!=eligible for x in labels) if labels and eligible is not None else None,
            'numeric_valid_variants':len(labels),'numeric_invalid_variants':invalid,
            'settle_stable':m.get('settle_stable') if valid else None,
            'penetration':m.get('max_object_penetration_all_m') if valid else None,
            'f3_elapsed_s':nominal.get('elapsed_s')}


def group_summary(rows):
    truth=[r['eligible'] for r in rows];m={}
    m['n']=len(rows);m['nominal_completed']=sum(y is not None for y in truth);m['unknown']=sum(y is None for y in truth)
    m['f1_status']=dict(Counter(r['f1_status'] for r in rows));m['f3_status']=dict(Counter(r['f3_status'] for r in rows))
    m['f0_vs_eligible']=confusion([r['f0'] for r in rows],truth)
    m['f1_vs_eligible']=confusion([r['f1'] for r in rows],truth)
    m['f2_vs_eligible']=confusion([r['f2_positive'] for r in rows],truth)
    pairs=[r for r in rows if r['eligible'] is not None and r['f2'] is not None]
    m['f2_vs_eligible_correlation']=correlation([r['f2'] for r in pairs],[r['eligible'] for r in pairs])
    m['strict_penetration_eligible']=sum(r['eligible'] is True and r['penetration'] is not None and r['penetration']<=.001 for r in rows)
    m['raw_sustained']=sum(r['raw'] is True for r in rows);m['eligible_sustained']=sum(r['eligible'] is True for r in rows)
    m['settle_unstable']=sum(r['settle_stable'] is False for r in rows)
    m['penetration_over_1mm']=sum(r['penetration'] is not None and r['penetration']>.001 for r in rows)
    m['nominal_numerically_valid']=sum(r['nominal_valid'] for r in rows)
    m['numeric_valid_variants']=sum(r['numeric_valid_variants'] for r in rows)
    m['numeric_invalid_variants']=sum(r['numeric_invalid_variants'] for r in rows)
    m['numeric_tested']=sum(r['numeric_changed'] is not None for r in rows)
    m['numerical_label_change']=sum(r['numeric_changed'] is True for r in rows)
    m['native_joint_elapsed_s']=sum(r['native_elapsed_s'] for r in rows if r['native_elapsed_s'] is not None)
    m['native_elapsed_observed']=sum(r['native_elapsed_s'] is not None for r in rows)
    m['strategies']={}
    for name in ('always','f1','f1_f2'):
        def select(r):return True if name=='always' else r['f1'] if name=='f1' else r['f1'] and r['f2_positive']
        chosen=[r for r in rows if select(r) is True]; omitted=[r for r in rows if select(r) is False]
        undecided=[r for r in rows if select(r) is None]
        m['strategies'][name]={'selected':len(chosen),'selection_unknown':len(undecided),'f3_known':sum(r['eligible'] is not None for r in chosen),
            'f3_unknown':sum(r['eligible'] is None for r in chosen),'isolated_success_retained':sum(r['eligible'] is True for r in chosen),
            'isolated_success_discarded':sum(r['eligible'] is True for r in omitted),
            'reachable_success_retained':sum(r['eligible'] is True and r['f1'] for r in chosen),
            'reachable_unknown':sum(r['f1'] and r['eligible'] is None for r in chosen),
            'measured_nominal_f3_elapsed_s':sum(r['f3_elapsed_s'] for r in chosen if r['f3_elapsed_s'] is not None),
            'measured_nominal_f3_costs':sum(r['f3_elapsed_s'] is not None for r in chosen)}
    return m


def analyze(result,protocol,catalog_dir):
    expected={(o['object_id'],str(i)) for o in protocol['objects'] for i in o['candidate_ids']}
    if len(expected)!=84 or len(result['rows'])!=len(expected):raise ValueError('incomplete frozen pilot')
    seen=set();rows=[];catalogs={}
    for r in result['rows']:
        key=(r['object_id'],str(r['candidate_id']))
        if key in seen or key not in expected:raise ValueError('duplicate or unregistered candidate')
        seen.add(key);oid,cid=key
        if oid not in catalogs:
            path=catalog_dir/(oid+'.json'); definition=next(o for o in protocol['objects'] if o['object_id']==oid)
            if digest(path)!=definition['catalog_sha256']:raise ValueError('catalog hash mismatch')
            catalogs[oid]=load(path)
        cat=catalogs[oid];i=cat['ids'].index(cid);width=float(cat['features'][i][9])*.1
        f1=r['f1_status'];
        if f1 not in ('feasible','rejected'):raise ValueError('unknown F1 status')
        f2=r.get('f2',{}).get('quality');f2=float(f2) if f2 is not None else None
        if f2 is not None and not math.isfinite(f2):f2=None
        dyn=classify_numerics(r)
        rows.append({'object_id':oid,'candidate_id':cid,'family':r['family'],'f1_status':f1,'f3_status':r['f3_status'],
                     'f0':width+.01<=.08,'f1':f1=='feasible','f2':f2,'f2_positive':f2>0 if f2 is not None else None,
                     'native_elapsed_s':r.get('native_elapsed_s'), **dyn})
    if seen!=expected:raise ValueError('pilot identities mismatch')
    return {'schema':1,'scope':'train only; isolated gripper; exploratory conditional on comparable proxy',
            'rows':rows,'aggregate':group_summary(rows),
            'by_object':{o:group_summary([r for r in rows if r['object_id']==o]) for o in sorted(catalogs)},
            'adaptive_strategy':'not evaluated; no prospectively fixed decision policy'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('result','protocol','catalogs','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();result=load(a.result)
    if result['protocol_sha256']!=digest(a.protocol):raise ValueError('protocol hash mismatch')
    report=analyze(result,load(a.protocol),a.catalogs)
    report.update({'input_sha256':digest(a.result),'protocol_sha256':digest(a.protocol),'script_sha256':digest(__file__)})
    save(a.out,report)

if __name__=='__main__':main()
