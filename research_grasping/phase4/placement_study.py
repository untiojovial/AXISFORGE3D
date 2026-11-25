"""Translate the target within the original table and compare matched native poses."""
from copy import deepcopy
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist

from research_grasping.phase3.experiment import DevelopmentData,digest,fingerprint,load,save
from research_grasping.phase3.native_catalog import evaluate_candidate,load_catalog
from research_grasping.phase3.prepare_pilot import vec
from .catalogue_study import REFERENCE,ROOT

PROTOCOL=ROOT/'research/phase4/candidates/placement_protocol.json'


def supported(project,catalog,delta):
    table=next(b for b in project['project']['environment']['bodies'] if b['name']=='table')
    geom=table['geometries'][0]
    for pose in (table['pose'],geom['origin']):
        if not np.allclose([pose['rotation'][k] for k in 'xyzw'],[0,0,0,1]):
            raise ValueError('requires an axis-aligned support')
    if geom['shape']['kind']!='box':raise ValueError('requires a box support')
    center=vec(table['pose']['translation'])+vec(geom['origin']['translation'])
    half=vec(geom['shape']['size'])/2
    m=catalog['metadata']
    low,high=np.asarray(m['source_bounds_m'])-m['source_origin_m']+np.asarray(m['base_placement_world_m'])+delta
    return bool(abs(low[2]-center[2]-half[2])<1e-5 and np.all(low[:2]>=center[:2]-half[:2])
                and np.all(high[:2]<=center[:2]+half[:2]))


def pose_vector(record,delta):
    c=record['candidate']
    return np.r_[vec(c['center'])+delta,vec(c['approach']),vec(c['closing']),c['width']]


def contacts_match(a,b,delta,tol=1e-5):
    def points(r,movement):
        return np.array([np.r_[vec(p['point'])+movement,vec(p['normal'])] for p in r['candidate']['contacts']])
    p,q=points(a,delta),points(b,np.zeros(3))
    if p.shape!=q.shape:return False
    if not len(p):return True
    dist=cdist(p,q,metric='chebyshev')
    from scipy.optimize import linear_sum_assignment
    ri,ci=linear_sum_assignment(dist)
    return bool(np.all(dist[ri,ci]<=tol))


def run(data,binary,work,destination):
    guard=DevelopmentData(REFERENCE/'manifest.prepilot.json',load(REFERENCE/'lock.json'),data)
    spec=load(PROTOCOL);delta=np.array(spec['change_m'])
    names={}
    for name,record in sorted(guard.objects.items()):
        if record['split']=='train':names.setdefault(record['geometry_family'],name)
    locked=load(ROOT/'research/phase4/candidates/selection.json')
    results=dict(protocol_sha256=digest(PROTOCOL),original_binary_sha256=digest(binary),objects=[])
    for family,name in sorted(names.items()):
        cat=guard.catalogue(work/'catalogues/diverse128'/(name+'.json'),binary)
        expected=locked['objects'][name]['catalogues']['diverse128']['catalogue_digest']
        if fingerprint(cat)!=expected:raise ValueError('unexpected source catalogue')
        start=time.monotonic()
        project=load(cat['project'])
        ids=list(map(int,cat['ids'][:10]))
        entry=dict(family=family,object_id=name,ids=ids,measurements=[],table_supported=all(
            supported(project,cat,offset) for offset in (np.zeros(3),delta)))
        results['objects'].append(entry)
        if entry['table_supported']:
            shifted=deepcopy(project)
            body=next(b for b in shifted['project']['environment']['bodies'] if b['name']==name)
            for k,dx in zip('xyz',delta):body['pose']['translation'][k]+=dx
            path=work/'placement'/(name+'.axisforge');save(path,shifted)
            before=load_catalog(binary,cat['project'],name)['records'];after=load_catalog(binary,path,name)['records']
            descriptors=np.array([pose_vector(r,np.zeros(3)) for r in after])
            taken=set()
            for cid in ids:
                a=before[cid]
                possible=np.flatnonzero(np.max(np.abs(descriptors-pose_vector(a,delta)),axis=1)<=1e-5)
                matched=[int(i) for i in possible if i not in taken and contacts_match(a,after[i],delta)]
                if cid in matched:cid2=cid
                elif len(matched)==1:cid2=matched[0]
                else:
                    entry['measurements'].append(dict(id=cid,status='unmatched',alternatives=len(matched)))
                    continue
                taken.add(cid2)
                pair=[]
                for file,i in ((cat['project'],cid),(path,cid2)):
                    env=evaluate_candidate(binary,file,name,i)
                    rec=env['records'][0]
                    pair.append(dict(id=i,reason='ok' if rec['plan'] is not None else rec['rejection']['reason'],
                                     epsilon=rec['quality']['quality'],seconds=env['elapsed_s']))
                entry['measurements'].append(dict(id=cid,status='paired',before=pair[0],after=pair[1],
                        f2_agrees=abs(pair[0]['epsilon']-pair[1]['epsilon'])<=spec['f2_absolute_tolerance']))
        entry['wall_s']=time.monotonic()-start
        save(destination,results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','binary','work','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.data,a.binary,a.work,a.out)
