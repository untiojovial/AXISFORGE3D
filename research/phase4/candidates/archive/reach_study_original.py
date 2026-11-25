"""Prospective, development-only first-rejection proxy ranking study."""
import argparse
from collections import Counter
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from research_grasping.phase3.experiment import DevelopmentData, digest, fingerprint, load, save
from research_grasping.phase3.pilot_cli import cached_observation
from .catalogue_study import REFERENCE, ROOT

PROTOCOL=ROOT/'research/phase4/candidates/reach_protocol.json'
HISTORY_LOCK=ROOT/'research/phase4/candidates/history_stratified.lock.json'
SOURCE_SELECTION=ROOT/'research/phase4/candidates/selection.json'
SELECTION=ROOT/'research/phase4/candidates/reach_selection.json'
SELECTION_HASH='6833566d2e0ef828a1ad731a2e7aea2922ca23060cd98dfc445df55735b7cdc2'
PROTOCOL_HASH='c445bc275676b5df7257ffd3294a2f6979b1a8b3eef0eecb52cbeb4eae4081df'
DATA=Path('/workspace/axisforge-phase3-data'); WORK=Path('/workspace/axisforge-phase4-data')
BINARY=WORK/'bin/axisforge-phase3'; CACHE=DATA/'run-development/cache'


def features(cat, cid):
    i=cat['ids'].index(cid); f=np.asarray(cat['features'][i],float)
    position=np.asarray(cat['candidates'][cid]['position'],float)
    ext=np.ptp(np.asarray(cat['metadata']['source_bounds_m'],float),axis=0)
    a=f[3:6]; c=f[6:9]
    if not np.isfinite(f).all() or not np.isfinite(position).all() or not np.isfinite(ext).all() or np.linalg.norm(a)==0 or np.linalg.norm(c)==0:
        raise ValueError('invalid prequery features')
    return np.r_[position,a/np.linalg.norm(a),c/np.linalg.norm(c),f[9]*0.1,ext]


def signature(cat):
    m=cat['metadata']; return fingerprint([m['project_sha256'],m['binary_sha256'],m['processed_sha256'],cat['body']])


def guard_setup():
    if digest(PROTOCOL)!=PROTOCOL_HASH: raise ValueError('protocol changed')
    spec=load(PROTOCOL)
    if digest(HISTORY_LOCK)!=spec['training_lock_sha256'] or digest(SOURCE_SELECTION)!=spec['catalogue_selection_sha256']:
        raise ValueError('input seal changed')
    selection=load(SOURCE_SELECTION)
    if digest(BINARY)!=selection['original_binary_sha256']: raise ValueError('native binary changed')
    guard=DevelopmentData(REFERENCE/'manifest.prepilot.json',load(REFERENCE/'lock.json'),DATA)
    return spec,selection,guard


def train_and_select():
    if SELECTION.exists(): raise FileExistsError('selection already frozen')
    spec,source,guard=guard_setup(); locks=load(HISTORY_LOCK); X=[]; y_ok=[]; y_reach=[]; counts=Counter()
    names=sorted(name for name,split in guard.splits.items() if split=='train')
    if set(names)!=set(locks) or len(names)!=16: raise ValueError('incorrect training roster')
    for name in names:
        path=WORK/'history/stratified'/f'{name}.json'
        if digest(path)!=locks[name]: raise ValueError('training data changed')
        history=load(path); cat=guard.catalogue(WORK/'catalogues/diverse128'/f'{name}.json',BINARY)
        if fingerprint(cat)!=source['objects'][name]['catalogues']['diverse128']['catalogue_digest'] or history['selected']!=source['objects'][name]['sampling']['stratified'] or len(history['rows'])!=50:
            raise ValueError('training selection differs')
        if [r['candidate_id'] for r in history['rows']]!=history['selected']: raise ValueError('training order differs')
        for row in history['rows']:
            reason=(row['details'].get('rejection') or {}).get('reason',row['status']); counts[reason]+=1
            if reason in ('too_wide','no_force_closure'): proxy=-1
            elif reason in ('pregrasp_unreachable','pregrasp_collision'): proxy=0
            elif reason in ('grasp_unreachable','lift_unreachable','approach_collision','ok'): proxy=1
            else: raise ValueError('unrecognized native first reason '+reason)
            X.append(features(cat,row['candidate_id'])); y_reach.append(proxy); y_ok.append(int(reason=='ok'))
    X=np.array(X); y_ok=np.array(y_ok); y_reach=np.array(y_reach)
    if len(X)!=800:raise ValueError('not 800 train rows')
    mean=X.mean(axis=0); std=X.std(axis=0); std[std<1e-12]=1.; Z=(X-mean)/std
    def fit(y, mask):
        model=LogisticRegression(C=.01,solver='lbfgs',max_iter=1000,random_state=0)
        model.fit(Z[mask],y[mask]); return model
    mreach=fit(y_reach,y_reach>=0); mok=fit(y_ok,np.ones(len(y_ok),dtype=bool))
    params={'mean':mean.tolist(),'std':std.tolist(),'pregrasp':{'coef':mreach.coef_[0].tolist(),'intercept':float(mreach.intercept_[0])},'ok':{'coef':mok.coef_[0].tolist(),'intercept':float(mok.intercept_[0])}}
    rng=np.random.default_rng(2026); objects={}
    for name in sorted(n for n,s in guard.splits.items() if s=='validation'):
        cat=guard.catalogue(WORK/'catalogues/diverse128'/f'{name}.json',BINARY)
        if fingerprint(cat)!=source['objects'][name]['catalogues']['diverse128']['catalogue_digest']: raise ValueError('catalogue mismatch')
        e=source['objects'][name]; maximum=e['max_opening_m']; sig=signature(cat)
        queried={p.stem for p in (CACHE/sig/'native-f12').glob('*.json')}
        prior=set(e['sampling']['stratified']) | set(e['sampling']['uniform']); excluded_queried=queried|prior
        widths={cid:float(cat['features'][i][9])*.1 for i,cid in enumerate(cat['ids'])}
        too_wide=[cid for cid in cat['ids'] if widths[cid]+.01>maximum]
        unseen=[cid for cid in cat['ids'] if cid not in excluded_queried and cid not in too_wide]
        if unseen:
            z=(np.array([features(cat,cid) for cid in unseen])-mean)/std
            score=np.sqrt(mreach.predict_proba(z)[:,1]*mok.predict_proba(z)[:,1])
            ranked=sorted(zip(unseen,score.tolist()),key=lambda pair:(-pair[1],int(pair[0])))
        else: ranked=[]
        n=min(12,len(ranked)//2); top=[cid for cid,_ in ranked[:n]]
        rest=sorted((cid for cid,_ in ranked[n:]),key=int)
        random_ids=[str(x) for x in rng.choice(rest,size=n,replace=False)] if n else []
        objects[name]={'catalogue_digest':fingerprint(cat),'signature':sig,'excluded_too_wide':too_wide,
          'excluded_previously_queried':sorted((set(cat['ids'])&excluded_queried),key=int),
          'cache_existing_ids':sorted(queried,key=int),'eligible_scores':{cid:v for cid,v in ranked},
          'top':top,'random':random_ids,'per_arm':n}
    frozen={'protocol_sha256':PROTOCOL_HASH,'binary_sha256':digest(BINARY),'source_selection_sha256':digest(SOURCE_SELECTION),
       'history_lock_sha256':digest(HISTORY_LOCK),'training_counts':dict(counts),'training_n':800,
       'pregrasp_train_n':int(np.sum(y_reach>=0)),'pregrasp_positive':int(np.sum(y_reach==1)),
       'ok_positive':int(y_ok.sum()),'model':params,'model_sha256':fingerprint(params),'objects':objects}
    save(SELECTION,frozen); print('selection SHA256',digest(SELECTION))
    print('per-arm sizes',{k:v['per_arm'] for k,v in objects.items()})


def measure():
    spec,source,guard=guard_setup()
    if digest(SELECTION)!=SELECTION_HASH: raise ValueError('selection seal changed')
    selection=load(SELECTION)
    if selection['protocol_sha256']!=PROTOCOL_HASH or selection['binary_sha256']!=digest(BINARY) or fingerprint(selection['model'])!=selection['model_sha256']:
        raise ValueError('frozen selection changed')
    # Check the entire selection before invoking the oracle, using filenames only.
    if set(selection['objects']) != {n for n, split in guard.splits.items() if split=='validation'}:
        raise ValueError('validation roster mismatch')
    for name,entry in selection['objects'].items():
        if guard.splits[name] != 'validation': raise ValueError('non-validation object')
        ids=entry['top']+entry['random']
        if len(set(ids)) != len(ids) or len(entry['top']) != len(entry['random']): raise ValueError('invalid arms')
        if set(ids) & (set(entry['excluded_too_wide']) | set(entry['excluded_previously_queried'])):
            raise ValueError('invalid preselection')
        for base in [CACHE, DATA/'run-smoke/cache', DATA/'pilot-smoke/cli-check/cache']:
            seen={p.stem for p in (base/entry['signature']/'native-f12').glob('*.json')}
            if set(ids)&seen: raise ValueError('selected ID already queried')
    stamp=ROOT/'research/phase4/candidates/reach_prequery.json'
    if stamp.exists(): raise FileExistsError('prequery stamp already exists; no implicit rerun')
    save(stamp,{'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':digest(PROTOCOL),
                'selection_sha256':digest(SELECTION),'binary_sha256':digest(BINARY),
                'validation_objects':sorted(selection['objects']),'oracle_calls_started':False})
    destination=WORK/'reach_results'; destination.mkdir(parents=True,exist_ok=True)
    for name,entry in sorted(selection['objects'].items()):
        cat=guard.catalogue(WORK/'catalogues/diverse128'/f'{name}.json',BINARY)
        if fingerprint(cat)!=entry['catalogue_digest'] or signature(cat)!=entry['signature']: raise ValueError('catalogue changed')
        for arm in ('top','random'):
            path=destination/f'{name}-{arm}.json'
            if path.exists(): raise FileExistsError(path)
            rows=[]
            for cid in entry[arm]:
                reading=cached_observation(CACHE,BINARY,cat['project'],name,entry['signature'],cid,'native-f12')
                rows.append({'candidate_id':cid, **reading})
            save(path,{'object_id':name,'arm':arm,'selection_sha256':digest(SELECTION),'rows':rows})
            print(name,arm,Counter((r['details'].get('rejection') or {}).get('reason',r['status']) for r in rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','measure']);a=p.parse_args()
    (train_and_select if a.stage=='select' else measure)()
