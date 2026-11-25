"""Prepare bounded pre-evaluation native catalogues for development objects only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from research_grasping.phase3.native_catalog import load_catalog


REFERENCE_M = .1
PLACEMENT = np.array([.45, 0., .42])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_objects(manifest, object_ids):
    objects = manifest['objects']
    by_id = {o['object_id']: o for o in objects}
    if len(by_id) != len(objects) or len(set(object_ids)) != len(object_ids):
        raise ValueError('duplicate object IDs')
    selected = []
    for object_id in object_ids:
        if object_id not in by_id or by_id[object_id].get('split') not in ('train', 'validation'):
            raise ValueError(f'unknown or non-development object: {object_id}')
        selected.append(by_id[object_id])
    return selected


def mesh_vertices(path):
    # Dataset pipeline exports binary STL; refuse ambiguous or malformed files.
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError('invalid binary STL')
    n = int.from_bytes(raw[80:84], 'little')
    if n < 1 or len(raw) != 84 + 50*n:
        raise ValueError('expected exact binary STL')
    dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, count=n, offset=84)['vertices'].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError('nonfinite mesh')
    return triangles


def mesh_points(triangles, count=256, seed=20261125):
    low = triangles.min(axis=(0,1)); high = triangles.max(axis=(0,1))
    v = triangles[:,1]-triangles[:,0]; w = triangles[:,2]-triangles[:,0]
    area = np.linalg.norm(np.cross(v,w),axis=1)
    if np.any(high <= low) or not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError('degenerate mesh')
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(triangles), size=count, p=area/area.sum())
    u = rng.random(count); q = rng.random(count); root = np.sqrt(u)
    points = ((1-root)[:,None]*triangles[indices,0] + (root*(1-q))[:,None]*triangles[indices,1] + (root*q)[:,None]*triangles[indices,2])
    # project-io/object.rs places the mesh bounding-box base centre at PLACEMENT.
    origin = np.array([(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]])
    return points-origin+PLACEMENT, low, high, origin


def vec(value):
    return np.asarray([value['x'], value['y'], value['z']], float)


def catalogue(records, cap=48):
    if not records or cap < 1:
        raise ValueError('empty native candidates or invalid cap')
    # Uniformly spaced generator-order indices, prior to any quality query.
    indices = np.unique(np.linspace(0,len(records)-1,min(cap,len(records)),dtype=int))
    ids, features, candidates = [], [], {}
    for index in indices:
        rec = records[int(index)]
        if any(rec.get(k) is not None for k in ('plan','rejection','quality','qualityElapsedS','attemptElapsedS')):
            raise ValueError('post-evaluation information in candidate catalogue')
        identifier = str(rec['id'])
        if identifier in candidates:
            raise ValueError('duplicate native ID')
        c = rec['candidate']; pos = vec(c['center'])
        approach, closing = vec(c['approach']), vec(c['closing'])
        contacts = [vec(entry['point']) for entry in c['contacts']]
        normals = [vec(entry['normal']) for entry in c['contacts']]
        # Each row is pre-evaluation, object-base-relative geometry and local normals.
        normals_mean = np.mean(normals,axis=0) if normals else np.zeros(3)
        local_contacts = np.asarray(contacts)-pos if contacts else np.empty((0,3))
        spread = np.sqrt(np.mean(local_contacts**2,axis=0)) if contacts else np.zeros(3)
        feature = np.concatenate(((pos-PLACEMENT)/REFERENCE_M,approach,closing,
                                  [float(c['width'])/REFERENCE_M],normals_mean,spread/REFERENCE_M))
        if not np.isfinite(feature).all() or not np.isfinite(pos).all() or not all(np.isfinite(p).all() for p in contacts):
            raise ValueError('nonfinite candidate')
        ids.append(identifier); features.append(feature.tolist())
        candidates[identifier] = {'position':pos.tolist(),'contacts':[p.tolist() for p in contacts]}
    return ids, features, candidates


def prepare(manifest_path, binary, out, object_ids, cap=48, vision_path=None):
    manifest_path = Path(manifest_path).resolve(); binary = Path(binary).resolve()
    manifest = json.loads(manifest_path.read_text()); objects = selected_objects(manifest,object_ids)
    data_root = manifest_path.parent
    vision = None
    vision_provenance = None
    if vision_path is not None:
        provenance_path = Path(vision_path).resolve().parent / 'provenance.json'
        vision_provenance = json.loads(provenance_path.read_text())
        if vision_provenance.get('manifest_sha256') != digest(manifest_path):
            raise ValueError('vision provenance manifest mismatch')
        with np.load(vision_path, allow_pickle=False) as archive:
            vision_ids = [str(x) for x in archive['ids']]
            vision_splits = [str(x) for x in archive['splits']]
            vectors = np.asarray(archive['embedding_4'], float)
        if len(set(vision_ids)) != len(vision_ids) or vectors.shape != (len(vision_ids), 512) or not np.isfinite(vectors).all():
            raise ValueError('invalid vision features')
        if any(split not in ('train','validation') for split in vision_splits):
            raise ValueError('non-development object in vision archive')
        vision = {i:(split, vector) for i,split,vector in zip(vision_ids,vision_splits,vectors)}
        if any(o['object_id'] not in vision or vision[o['object_id']][0] != o['split'] for o in objects):
            raise ValueError('missing or mismatched vision context')
    out = Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    for obj in objects:
        mesh = (data_root / obj['visual_mesh']).resolve()
        if not mesh.is_relative_to(data_root) or digest(mesh) != obj['processed_sha256']:
            raise ValueError('mesh path or hash mismatch')
        triangles = mesh_vertices(mesh)
        points, low, high, source_origin = mesh_points(triangles)
        project = out / (obj['object_id'] + '.axisforge')
        body = obj['object_id']; start = time.perf_counter()
        commands = [[str(binary),'demo','--no-parts','--out',str(project)],
                    [str(binary),'add-object',str(project),str(mesh),'--at',*[str(x) for x in PLACEMENT],
                     '--units','m','--name',body,'--mass','0.1']]
        logs = []
        for cmd in commands:
            done = subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=120)
            logs.append({'args':cmd,'stdout':done.stdout,'stderr':done.stderr})
        native = load_catalog(binary, project, body, timeout_s=300)
        ids, features, candidates = catalogue(native['records'],cap)
        result = {'object_id':body,'split':obj['split'],'project':str(project),'body':body,
                  'mesh_file':str(mesh),'ids':ids,'features':features,'context':((high-low)/REFERENCE_M).tolist() + (vision[body][1].tolist() if vision is not None else []),
                  'points':points.tolist(),'candidates':candidates,'frame':'AxisForge3D world, metres, z up',
                  'metadata':{'manifest_sha256':digest(manifest_path),'project_sha256':digest(project),
                              'binary_sha256':digest(binary),'processed_sha256':digest(mesh),
                              'source_project_mesh_sha256':digest(mesh), 'source_bounds_m':[low.tolist(),high.tolist()],
                              'source_origin_m':source_origin.tolist(),'base_placement_world_m':PLACEMENT.tolist(),
                              'reference_length_m':REFERENCE_M,'native_candidate_count':len(native['records']),
                              'selection':'uniform generator-order indices before evaluation',
                              'point_sampling':'256 area-weighted triangle samples, seed 20261125',
                              'context':'bounding box dimensions / 0.1 m' + ('; OpenCLIP embedding_4 (512 values)' if vision is not None else '; no foundation model'),
                              'vision_archive_sha256':digest(vision_path) if vision is not None else None,
                              'vision_archive':str(Path(vision_path).resolve()) if vision is not None else None,
                              'vision_provenance_sha256':digest(provenance_path) if vision is not None else None,
                              'vision_provenance':vision_provenance,
                              'features':'base-relative center/0.1m, approach, closing, width/0.1m, mean proposed contact normals, contact RMS spread/0.1m',
                              'mass_kg_assumed':0.1,'friction_assumed':0.5,'scale':'manifest metres, uncalibrated',
                              'elapsed_s':time.perf_counter()-start,'native_elapsed_s':native['elapsed_s'],
                              'commands':logs,'native_command':native['command']}}
        (out / (body + '.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(body, obj['split'], len(ids), len(native['records']), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--binary',required=True)
    p.add_argument('--output',required=True); p.add_argument('--object',action='append',required=True)
    p.add_argument('--max-candidates',type=int,default=48); p.add_argument('--vision-archive')
    a = p.parse_args(); prepare(a.manifest,a.binary,a.output,a.object,a.max_candidates,a.vision_archive)

if __name__ == '__main__':
    main()
