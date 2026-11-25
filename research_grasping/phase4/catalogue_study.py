"""Construct and lock geometry-only native candidate sets, then measure F1/F2."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from research_grasping.phase3.experiment import DevelopmentData,digest,fingerprint,load,save
from research_grasping.phase3.native_catalog import load_catalog
from research_grasping.phase3.prepare_pilot import catalogue
from research_grasping.phase3.pilot_cli import cached_observation

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'research/phase4/candidates/protocol.json'
REFERENCE=ROOT/'research/phase3/pilot'


def plans(ids, split, method, protocol):
    cap=protocol['train_measurements_per_object' if split=='train' else 'validation_measurements_per_object']
    if split not in ('train','validation') or len(set(ids))!=len(ids) or len(ids)<cap:
        raise ValueError('unsupported split or insufficient distinct native IDs')
    if method=='stratified':
        return ids[:cap]
    if method=='uniform':
        return np.random.default_rng(protocol['uniform_seed']).permutation(ids)[:cap].tolist()
    raise ValueError('unknown sampling design')


def geometry_summary(rows, maximum, clearance):
    centers=np.array([[row['candidate']['center'][i] for i in 'xyz'] for row in rows])
    approaches=np.array([[row['candidate']['approach'][i] for i in 'xyz'] for row in rows])
    widths=np.array([row['candidate']['width'] for row in rows])
    n=len(rows)
    # Occupied directional octants and physical extent; no candidate labels.
    return dict(actual=n,opening_compatible=int(np.count_nonzero(widths+2*clearance<=maximum)),
                span_m=np.ptp(centers,axis=0).tolist(),orientation_octants=int(len(np.unique(approaches>=0,axis=0))))


@contextmanager
def writer_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def prepare(data,binary,selector,out):
    with writer_lock(out/'.prepare.lock'):
        return _prepare(data,binary,selector,out)


def _prepare(data, binary, selector, out):
    protocol=load(PROTOCOL)
    guard=DevelopmentData(REFERENCE/'manifest.prepilot.json',load(REFERENCE/'lock.json'),data)
    source=data/'pilot-expanded/catalogs'
    signature = dict(manifest_sha256=guard.manifest_digest, protocol_sha256=digest(PROTOCOL),
                     original_binary_sha256=digest(binary), selector_binary_sha256=digest(selector))
    if (out/'selection.json').exists():
        previous=load(out/'selection.json')
        if any(previous.get(k)!=v for k,v in signature.items()) or set(previous['objects'])!={
            name for name,split in guard.splits.items() if split!='test'}:
            raise ValueError('existing selection belongs to another configuration')
        for name,entry in previous['objects'].items():
            for scheme,key in [('stratified','diverse128'),('uniform','uniform128')]:
                c=guard.catalogue(out/'catalogues'/key/(name+'.json'),binary)
                if (fingerprint(c)!=entry['catalogues'][key]['catalogue_digest']
                        or entry['sampling'][scheme]!=plans(c['ids'],c['split'],scheme,protocol)):
                    raise ValueError('cached native catalogue or selection changed')
        return previous
    objects={}
    for name,split in sorted(guard.splits.items()):
        if split=='test':continue
        base=guard.catalogue(source/(name+'.json'),binary)
        ids=base['ids']
        original=load_catalog(binary,base['project'],base['body'])['records']
        maximum=load(base['project'])['project']['robot']['gripper']['maxOpening']
        start=time.perf_counter()
        completed=subprocess.run([str(selector.resolve()),'research-candidates',base['project'],name,
                         '--diverse-limit',str(max(protocol['nested_catalogue_sizes']))],check=True,
                         capture_output=True,text=True,timeout=180)
        ordered=json.loads(completed.stdout)
        selector_time=time.perf_counter()-start
        reference={int(row['id']):row for row in original}
        if not ordered or any(reference.get(row['id'])!=row for row in ordered):
            raise ValueError('native pre-query descriptor differs between binaries')
        if len({r['id'] for r in ordered})!=len(ordered):
            raise ValueError('duplicate native indices')
        if ids!=catalogue(original,48)[0]:raise ValueError('Phase III native order changed')
        entry=dict(split=split,native_count=len(original),selector_elapsed_s=selector_time,
                   max_opening_m=maximum,jaw_clearance_m=.005,catalogues={},sampling={})
        for size in protocol['nested_catalogue_sizes']:
            for method,rows in [('diverse',ordered[:size]),
                     ('uniform',[original[i] for i in np.linspace(0,len(original)-1,min(size,len(original)),dtype=int)])]:
                key=method+str(size)
                desc=geometry_summary(rows,maximum,.005)
                desc['native_ids_sha256']=fingerprint([r['id'] for r in rows])
                if size==128:
                    chosen,features,candidates=catalogue(rows,len(rows))
                    meta=dict(base['metadata'],selection=key,phase4_protocol_sha256=digest(PROTOCOL),
                              selector_binary_sha256=digest(selector))
                    c=dict(base,ids=chosen,features=features,candidates=candidates,metadata=meta)
                    save(out/'catalogues'/key/(name+'.json'),c)
                    desc['catalogue_digest']=fingerprint(c)
                    scheme='stratified' if method=='diverse' else 'uniform'
                    entry['sampling'][scheme]=plans(chosen,split,scheme,protocol)
                entry['catalogues'][key]=desc
        objects[name]=entry
    lock=dict(schema=1,manifest_sha256=guard.manifest_digest,protocol_sha256=digest(PROTOCOL),
              original_binary_sha256=digest(binary),selector_binary_sha256=digest(selector),objects=objects)
    save(out/'selection.json',lock)
    return lock


def measure(data, binary, out, workers):
    lock=load(out/'selection.json')
    if digest(PROTOCOL)!=lock['protocol_sha256'] or digest(binary)!=lock['original_binary_sha256']:
        raise ValueError('immutable protocol or oracle mismatch')
    guard=DevelopmentData(REFERENCE/'manifest.prepilot.json',load(REFERENCE/'lock.json'),data)
    if set(lock['objects'])!={name for name,split in guard.splits.items() if split!='test'}:
        raise ValueError('development roster mismatch')
    cache=data/'run-development/cache'
    def evaluate(job):
        name,entry,scheme,key=job
        c=guard.catalogue(out/'catalogues'/key/(name+'.json'),binary)
        sig=fingerprint([c['metadata']['project_sha256'],c['metadata']['binary_sha256'],
                         c['metadata']['processed_sha256'],c['body']])
        ids=entry['sampling'][scheme]
        if fingerprint(c)!=entry['catalogues'][key]['catalogue_digest'] or ids!=plans(c['ids'],c['split'],scheme,load(PROTOCOL)):
            raise ValueError('candidate selection differs from lock')
        category='history' if c['split']=='train' else 'validation_panel'
        path=out/category/scheme/(name+'.json')
        with writer_lock(path.with_suffix('.lock')):
            signature=dict(object_id=name,selected=ids,fidelity='native-f12',catalogue_digest=fingerprint(c),
                           selection_sha256=digest(out/'selection.json'))
            if path.exists():
                previous=load(path)
                if (any(previous.get(k)!=v for k,v in signature.items()) or
                        [r['candidate_id'] for r in previous['rows']]!=ids):
                    raise ValueError('immutable results differ')
                return
            begin=time.monotonic()
            rows=[]
            for i in ids:
                reading=cached_observation(cache,binary,c['project'],name,sig,i,'native-f12')
                rows.append(dict(candidate_id=i,**reading))
            save(path,dict(**signature,rows=rows,completed_at=time.time(),wall_s=time.monotonic()-begin))
            print(category,scheme,name,sum(r['status']=='ok' for r in rows),len(rows),flush=True)
    jobs=[(name,entry,scheme,key) for scheme,key in [('stratified','diverse128'),('uniform','uniform128')]
          for name,entry in lock['objects'].items()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(evaluate,jobs))


def report(out):
    result=[]
    for category in ('history','validation_panel'):
        for scheme in ('stratified','uniform'):
            for path in sorted((out/category/scheme).glob('*.json')):
                entry=load(path)
                for budget in [10,25,50]:
                    if budget>len(entry['rows']):continue
                    rows=entry['rows'][:budget]
                    causes=Counter()
                    for r in rows:
                        reject=r['details'].get('rejection')
                        causes[reject['reason'] if reject else r['status']]+=1
                    result.append(dict(category=category,scheme=scheme,object_id=entry['object_id'],budget=budget,
                     feasible=causes['ok'],causes=dict(causes),nominal_seconds=sum(r['cost'] for r in rows),
                     fresh_seconds=sum(r['cost'] for r in rows if not r['details'].get('cached'))))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','measure','report'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--selector',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.data,a.binary,a.selector,a.out)
    elif a.stage=='measure':measure(a.data,a.binary,a.out,a.workers)
    else:save(a.report,report(a.out))
