"""Does pre-query similarity predict an independently queried native F2 value?

Uses only the frozen train archive for neighbours. Validation outcomes are read
after predictions. Validation seed 11 from the random policy fixes the target
queries independently of model predictions. Pair counts are descriptive.
"""
import argparse
from pathlib import Path
import platform
import importlib.metadata

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from .experiment import DevelopmentData, digest, fingerprint, load, save
from .vision_openclip import visible_patch, CHECKPOINT_SHA256
from .vision_integrity import verify_visual


def local_descriptor(points, camera, depths, poses, tokens):
    seen = []
    views = 0
    for depth,pose,grid in zip(depths,poses,tokens,strict=True):
        observed = [v for p in points if (v:=visible_patch(p, depth, camera, pose, grid)) is not None]
        if observed:
            views+=1
            # Each token contributes once within a view, even for a dense native patch.
            seen.append(np.unique(observed,axis=0).mean(axis=0))
    if not seen:
        return np.zeros(tokens.shape[-1]), views
    v=np.mean(seen,axis=0)
    return v/max(np.linalg.norm(v),1e-12), views


def descriptors(catalog, ids, path, manifest_hash):
    meta=catalog['metadata']
    shift=np.asarray(meta['source_origin_m'])-meta['base_placement_world_m']
    result=[]
    verify_visual(path, catalog['object_id'])
    with np.load(path,allow_pickle=False) as archive:
        if (str(archive['manifest_sha256'])!=manifest_hash or str(archive['mesh_sha256'])!=meta['processed_sha256']
                or str(archive['checkpoint_sha256'])!=CHECKPOINT_SHA256):
            raise ValueError('incorrect per-object vision archive')
        camera,depths,poses,tokens=[archive[k] for k in ('K','depth','world_from_camera','patches')]
        for cid in ids:
            xyz=np.asarray(catalog['candidates'][cid]['contacts'],float).reshape(-1,3)+shift
            local,views=local_descriptor(xyz,camera,depths,poses,tokens)
            result.append(dict(cid=cid,geometry=catalog['features'][catalog['ids'].index(cid)],
                      global_embedding=catalog['context'][3:],local=local,views=views))
    return result


def distances(source, target):
    def matrix(rows,k):
        return np.asarray([r[k] for r in rows])
    geometry=matrix(source,'geometry')
    sd=geometry.std(axis=0);sd[sd<1e-8]=1
    geo=cdist(matrix(target,'geometry')/sd,geometry/sd)/np.sqrt(geometry.shape[1])
    glob=1-matrix(target,'global_embedding')@matrix(source,'global_embedding').T
    local=1-matrix(target,'local')@matrix(source,'local').T
    # Missing or exactly zero tokens carry neutral cosine similarity.
    return dict(geometry=geo,global_visual=glob,local_visual=local,equal_fusion=(geo+glob+local)/3)


def evaluate(train, target, train_labels, target_labels):
    d=distances(train,target)
    y=np.asarray([r['details']['epsilon'] for r in train_labels])
    f1=np.asarray([r['status']=='ok' for r in train_labels],float)
    observed=np.asarray([r['details']['epsilon'] for r in target_labels])
    feasible=np.asarray([r['status']=='ok' for r in target_labels],float)
    estimates={'train_constant': dict(epsilon_mae=float(np.mean(np.abs(y.mean()-observed))),
        f1_brier=float(np.mean((f1.mean()-feasible)**2)), distance_vs_epsilon_delta=None,
        epsilon_predictions=np.full(len(observed), y.mean()).tolist(),
        f1_probabilities=np.full(len(observed), f1.mean()).tolist())}
    for kind,dist in d.items():
        # All ties at the third-neighbour distance receive equal weight.
        k=min(3,len(y)); cutoff=np.partition(dist,k-1,axis=1)[:,k-1]
        weights=(dist <= cutoff[:,None]+1e-12).astype(float);weights/=weights.sum(axis=1,keepdims=True)
        pred=weights@y;prob=weights@f1
        pair_delta=np.abs(observed[:,None]-y)
        rho=None
        if np.ptp(dist)>0 and np.ptp(pair_delta)>0:
            rho=float(spearmanr(dist.ravel(),pair_delta.ravel()).statistic)
        estimates[kind]=dict(epsilon_mae=float(np.mean(np.abs(pred-observed))),
                            f1_brier=float(np.mean((prob-feasible)**2)),distance_vs_epsilon_delta=rho,
                            epsilon_predictions=pred.tolist(),f1_probabilities=prob.tolist())
    return estimates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','catalogues','protocol','binary','vision','run','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    data=DevelopmentData(a.protocol/'manifest.prepilot.json',load(a.protocol/'lock.json'),a.data)
    source=[]; source_labels=[]; target_catalogs=[]; inputs={}
    for file in sorted(a.catalogues.glob('*.json')):
        c=data.catalogue(file,a.binary);oid=c['object_id']
        log_path = a.run/'history'/(oid+'.json') if c['split']=='train' else a.run/'runs'/oid/'11-random.json'
        inputs[oid] = {'catalogue_sha256':digest(file), 'observations_sha256':digest(log_path),
                       'vision_archive_sha256':digest(a.vision/(oid+'.npz')), 'split':c['split']}
        if c['split']=='train':
            log=load(a.run/'history'/(oid+'.json'))
            if log['catalogue_digest']!=fingerprint(c):raise ValueError('training catalogue mismatch')
            labels=[r for r in log['rows'] if r['status']!='infrastructure_failure']
            source.extend(descriptors(c,[r['candidate_id'] for r in labels],a.vision/(oid+'.npz'),data.manifest_digest))
            source_labels.extend(labels)
        else:target_catalogs.append(c)
    report=dict(train=len(source),train_visible=sum(r['views']>0 for r in source),
                train_feasible=sum(r['status']=='ok' for r in source_labels),objects=[],
                provenance={'selection':'training history excluding infrastructure failures; validation random seed 11',
                    'manifest_sha256':data.manifest_digest, 'binary_sha256':digest(a.binary),
                    'python':platform.python_version(), 'numpy':np.__version__,
                    'scipy':importlib.metadata.version('scipy'),
                    'torch':importlib.metadata.version('torch'),
                    'scikit_learn':importlib.metadata.version('scikit-learn'),
                    'script_sha256':digest(__file__), 'lock_sha256':digest(a.protocol/'lock.json'),
                    'checkpoint_sha256':CHECKPOINT_SHA256, 'inputs':inputs})
    for c in target_catalogs:
        oid=c['object_id']; log=load(a.run/'runs'/oid/'11-random.json')
        if log['definition']['catalogue_digest']!=fingerprint(c):raise ValueError('validation catalogue mismatch')
        rows=log['rows']
        if any(r['status']=='infrastructure_failure' for r in rows):
            raise ValueError('incomplete random reference')
        ids=[r['candidate_id'] for r in rows]
        target=descriptors(c,ids,a.vision/(oid+'.npz'),data.manifest_digest)
        estimates=evaluate(source,target,source_labels,rows)
        report['objects'].append(dict(object_id=oid,ids=ids,visible_views=[r['views'] for r in target],
                                      measured_epsilon=[r['details']['epsilon'] for r in rows],methods=estimates))
    report['object_macro'] = {name: {metric:float(np.mean([obj['methods'][name][metric] for obj in report['objects']]))
        for metric in ('epsilon_mae', 'f1_brier')} for name in report['objects'][0]['methods']}
    save(a.out,report)


if __name__=='__main__':
    main()
