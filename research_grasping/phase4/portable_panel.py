"""Package previously measured development candidates and model-independent visible pixels."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from research_grasping.phase3.experiment import DevelopmentData, digest, fingerprint, load, save
from research_grasping.phase3.vision_integrity import verify_visual
from research_grasping.representations.geometry import project

ROOT=Path(__file__).resolve().parents[2]


def visible_coordinates(points, depths, k, poses, tolerance=.003):
    views=[]
    for depth,pose in zip(depths,poses,strict=True):
        uv,z=project(points,k,pose)
        h,w=depth.shape
        coordinates=[]
        for xy,axial in zip(uv,z,strict=True):
            u,v=xy
            if not (z.size and axial>0 and -.5<=u<w-.5 and -.5<=v<h-.5):continue
            ix,iy=np.floor(xy+.5).astype(int)
            if np.isfinite(depth[iy,ix]) and abs(depth[iy,ix]-axial)<=tolerance:
                coordinates.append([float(u),float(v)])
        views.append(coordinates)
    return views


def dump_gzip(path,value):
    path.write_bytes(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0))


def prepare(data,binary,out):
    reference=ROOT/'research/phase3/pilot'
    guard=DevelopmentData(reference/'manifest.prepilot.json',load(reference/'lock.json'),data)
    catalogues={}; measurements={}
    original=ROOT/'research/phase3/bo/results/development/observations.json.gz'
    with gzip.open(original,'rt') as f: archive=json.load(f)
    train={t['object_id']:t for t in archive['history']}
    refs={t['definition']['object_id']:t for t in archive['runs']
          if t['definition']['method']=='random' and t['definition']['seed']==11}
    for name,split in sorted(guard.splits.items()):
        if split=='test':continue
        c=guard.catalogue(data/'pilot-expanded/catalogs'/(name+'.json'),binary)
        query=train[name] if split=='train' else refs[name]
        if (query if split=='train' else query['definition'])['catalogue_digest']!=fingerprint(c):
            raise ValueError('trace/catalogue mismatch')
        rows=query['rows']; ids=[r['candidate_id'] for r in rows]
        if len(set(ids))!=10 or any(r['status']=='infrastructure_failure' for r in rows):
            raise ValueError('incomplete reference sample')
        visual=data/'vision-development-v3'/(name+'.npz')
        verify_visual(visual,name)
        shift=np.asarray(c['metadata']['source_origin_m'])-c['metadata']['base_placement_world_m']
        with np.load(visual,allow_pickle=False) as f:
            k,ts,depths=f['K'],f['world_from_camera'],f['depth']
        catalogues[name]=dict(split=split,ids=ids,features=[c['features'][c['ids'].index(i)] for i in ids],
                              dimensions=c['context'][:3],source_shape=list(depths.shape[1:]),
                              visible=[visible_coordinates(np.asarray(c['candidates'][i]['contacts']).reshape(-1,3)+shift,depths,k,ts) for i in ids])
        measurements[name]={r['candidate_id']:{k:r[k] for k in ('status','utility','cost')}|
                            {'epsilon':r['details']['epsilon']} for r in rows}
    out.mkdir(parents=True,exist_ok=True)
    dump_gzip(out/'candidates.json.gz',catalogues)
    dump_gzip(out/'observations.json.gz',measurements)
    save(out/'lock.json',dict(schema=1,manifest_sha256=guard.manifest_digest,source_archive_sha256=digest(original),
           candidates_sha256=digest(out/'candidates.json.gz'),observations_sha256=digest(out/'observations.json.gz'),
           outcome='native-f12',population='16 historical, 6 validation; ten previous random queries per object',
           online_budget=6,initial=2,seeds=[11,29,47],methods=['random','target_local','contextual','adaptive'],
           representations=['geometry','random_control','global','local','global_local']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','binary','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.data,a.binary,a.out)
