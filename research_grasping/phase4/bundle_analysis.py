"""Continue a verified feature import on the same archived development queries.

All model variants use the same measured finite search space. The policy
receives a target outcome only after choosing that candidate. No robot is run.
"""
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression

from research_grasping.phase3.experiment import PREPILOT_MANIFEST, digest, fingerprint, load, save
from research_grasping.phase3.transfer_policy import CataloguePolicy, FrozenScale, HistoricalTask

PANEL=Path(__file__).resolve().parents[2]/'research/phase4/embedding_experiment'


def patch_features(visible, patches, geometry, source_shape):
    """Pool unique visible token indices per view; equal weight per visible view."""
    gh,gw=geometry['grid_shape']
    h,w=geometry['processed_size']
    affine=np.asarray(geometry['pixel_transform'],float)
    top,left,bottom,right=geometry['content_bounds']
    if (list(source_shape)!=geometry['origin_size'] or patches.ndim!=3 or patches.shape[1]!=gh*gw
            or len(patches)!=len(visible) or min(gh,gw,h,w)<=0 or affine.shape!=(3,3)
            or not np.isfinite(affine).all() or not np.allclose(affine[2],[0,0,1])):
        raise ValueError('invalid feature/geometry correspondence')
    pooled=[]
    for grid,points in zip(patches,visible,strict=True):
        if not points:continue
        uv=np.asarray(points,float)
        xy=uv@affine[:2,:2].T+affine[:2,2]
        x,y=xy.T
        ok=(x>=max(left,-.5))&(x<min(right,w-.5))&(y>=max(top,-.5))&(y<min(bottom,h-.5))
        col=np.floor((x[ok]+.5)*gw/w).astype(int)
        row=np.floor((y[ok]+.5)*gh/h).astype(int)
        if len(col):
            pooled.append(grid[np.unique(row*gw+col)].astype(float).mean(axis=0))
    if not pooled:return np.zeros(patches.shape[-1]),0
    v=np.mean(pooled,axis=0)
    return v/max(np.linalg.norm(v),1e-12),len(pooled)


def compressed(training, blocks):
    train=np.vstack([blocks[k] for k in training])
    center=train.mean(axis=0)
    _,singular,axes=np.linalg.svd(train-center,full_matrices=False)
    keep=axes[:min(4,np.count_nonzero(singular>1e-8))]
    return {k:(v-center)@keep.T for k,v in blocks.items()}


def load_inputs(import_dir,index,panel):
    rows={}
    geometry=index['token_geometry']
    for item in index['objects']:
        name=item['object_id']
        if name in rows or name not in panel or panel[name]['split']!=item['split']:
            raise ValueError('unexpected feature object or split')
        if item['file']!=f'features/{name}.npz':
            raise ValueError('unexpected feature file path')
        path=import_dir/item['file']
        if digest(path)!=item['sha256']:raise ValueError('feature file changed after import')
        with np.load(path,allow_pickle=False) as f:
            globals_,patches,ids=f['global_embeddings'],f['patch_embeddings'],f['view_ids']
        if (not np.array_equal(ids,np.arange(16)) or globals_.ndim!=2 or len(globals_)!=16
                or not np.isfinite(globals_).all() or not np.isfinite(patches).all()):
            raise ValueError('invalid features')
        v=globals_.astype(float).mean(axis=0);v/=max(np.linalg.norm(v),1e-12)
        local=[patch_features(points,patches,geometry,panel[name]['source_shape']) for points in panel[name]['visible']]
        rows[name]=dict(global_vector=v,local=np.stack([p for p,_ in local]),views=[n for _,n in local])
    return rows


def feature_sets(panel, imported):
    train=sorted(name for name,row in panel.items() if row['split']=='train')
    global_pc=compressed(train,{k:r['global_vector'][None] for k,r in imported.items()})
    local_pc=compressed(train,{k:r['local'] for k,r in imported.items()})
    for kind in ('geometry','random_control','global','local','global_local'):
        x={};context={}
        for name,record in panel.items():
            geo=np.array(record['features'])
            ctx=np.array(record['dimensions'])
            pieces=[geo]
            if kind in ('global','global_local'):
                ctx=np.r_[ctx,global_pc[name][0]]
            if kind in ('local','global_local'):
                pieces += [local_pc[name],(np.array(imported[name]['views'])[:,None]>0).astype(float)]
            if kind=='random_control':
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'big'))
                ctx=np.r_[ctx,rng.normal(size=global_pc[name].shape[1])]
                pieces += [rng.normal(size=local_pc[name].shape),np.ones((len(geo),1))]
            x[name]=np.concatenate(pieces,axis=1)
            context[name]=ctx
        yield kind,x,context


def prediction_probe(panel, measured, x, ctx, train, valid):
    def matrix(name):
        return np.c_[x[name],np.tile(ctx[name],(len(x[name]),1))]
    inputs=np.vstack([matrix(name) for name in train])
    labels=[measured[name][cid] for name in train for cid in panel[name]['ids']]
    mean=inputs.mean(0);std=inputs.std(0);std[std<1e-8]=1
    y=np.array([r['epsilon'] for r in labels])
    f=np.array([r['status']=='ok' for r in labels],dtype=int)
    gp=GaussianProcessRegressor(kernel=Matern(length_scale=np.sqrt(inputs.shape[1]),nu=2.5),
                        alpha=.15,optimizer=None,normalize_y=True).fit((inputs-mean)/std,y)
    clf=LogisticRegression(C=.1,random_state=0).fit((inputs-mean)/std,f) if len(set(f))>1 else None
    result={}
    for name in valid:
        z=(matrix(name)-mean)/std
        pred=gp.predict(z);prob=clf.predict_proba(z)[:,1] if clf else np.repeat(f.mean(),len(z))
        target=[measured[name][cid] for cid in panel[name]['ids']]
        epsilon=np.array([r['epsilon'] for r in target]);success=np.array([r['status']=='ok' for r in target])
        result[name]=dict(epsilon_mae=float(np.abs(pred-epsilon).mean()),f1_brier=float(np.mean((prob-success)**2)),
                          constant_epsilon_mae=float(np.abs(y.mean()-epsilon).mean()),
                          constant_f1_brier=float(np.mean((f.mean()-success)**2)))
    return result


def replay(panel,observations,x,ctx,protocol):
    splits={k:v['split'] for k,v in panel.items()}
    train=sorted(k for k,v in splits.items() if v=='train'); valid=sorted(set(panel)-set(train))
    train_x=np.vstack([x[k] for k in train])
    sd=train_x.std(axis=0);sd[sd<1e-8]=1
    scaler=FrozenScale(train_x.mean(axis=0),sd,protocol['observations_sha256'])
    history=[]
    for name in train:
        rows=[observations[name][cid] for cid in panel[name]['ids']]
        history.append(HistoricalTask(name,0.,ctx[name],x[name],np.array([r['utility'] for r in rows]),
                                      sum(r['cost'] for r in rows)))
    runs=[]
    for name in valid:
        ids=panel[name]['ids']
        for seed in protocol['seeds']:
            initial=np.random.default_rng(seed).choice(ids,protocol['initial'],replace=False).tolist()
            for method in protocol['methods']:
                policy=CataloguePolicy(ids,scaler.transform(x[name]),ctx[name],history,manifest_splits=splits,
                     manifest_digest=fingerprint(splits),target_id=name,scaler=scaler,target_time=1.,
                     method=method,budget=protocol['online_budget'],seed=seed,initial_ids=initial)
                best=0.;curve=[];used=[]
                for _ in range(policy.budget):
                    candidate=policy.ask()
                    answer=observations[name][candidate]
                    policy.tell(candidate,status=answer['status'],utility=answer['utility'],cost=answer['cost'])
                    used.append(candidate);best=max(best,answer['utility']);curve.append(best)
                runs.append(dict(object_id=name,seed=seed,method=method,selected=used,auc=float(np.mean(curve))))
    return dict(trajectories=runs,predictive=prediction_probe(panel,observations,x,ctx,train,valid))


def verified_index(root, bundle_path, job_path, expected_checkpoint_sha256, expected_revision, expected_processor_sha256):
    from .bundle_io import verify_bundle
    if bundle_path is None or job_path is None:
        raise ValueError('Original result ZIP and sealed image job are required')
    index = verify_bundle(bundle_path, job_path)
    if index != load(root/'index.json'):
        raise ValueError('installed index differs from verified bundle')
    if index['model_id'] == 'open_clip/ViT-B-32-quickgelu/laion400m_e31':
        expected = ('v0.2-weights',
            'd867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e',
            'c4c7c2276f27ec1b7f9e45f1f76dde5176ca32a245eafeadd607a3c7f4c1660e')
    elif index['model_id'] in ('dinov2','dinov3'):
        import re
        expected = (expected_revision, expected_checkpoint_sha256, expected_processor_sha256)
        if not re.fullmatch(r'[a-f0-9]{40}', expected_revision or '') or not all(expected):
            raise ValueError('DINO requires independent official revision, weight and processor hashes')
    else:
        raise ValueError('unsupported experiment model')
    actual = tuple(index[k] for k in ('model_revision','checkpoint_sha256','processor_sha256'))
    if expected != actual:
        raise ValueError('model identity does not match independent pins')
    return index


def run(import_dir: Path, *, bundle_path=None, job_path=None, expected_checkpoint_sha256=None,
        expected_revision=None, expected_processor_sha256=None, data_root=None, out=None):
    """Run the archive benchmark after verified installation. A subset is a smoke only."""
    root=Path(import_dir)
    index=verified_index(root,bundle_path,job_path,expected_checkpoint_sha256,expected_revision,expected_processor_sha256)
    if digest(PANEL/'lock.json') != '3884a397629cbfd0b3c8f622d5b3cdae98a764e019eb4875025eb729ee3db951':
        raise ValueError('portable panel protocol differs from its independent pin')
    protocol=load(PANEL/'lock.json')
    if index['dataset_manifest_sha256']!=PREPILOT_MANIFEST or protocol['manifest_sha256']!=PREPILOT_MANIFEST:
        raise ValueError('bundle from different dataset')
    records=[]
    for name in ('candidates','observations'):
        path=PANEL/(name+'.json.gz')
        if digest(path)!=protocol[name+'_sha256']:raise ValueError('portable experiment input hash mismatch')
        with gzip.open(path,'rt') as f:records.append(json.load(f))
    panel,measurements=records
    imported=load_inputs(root,index,panel)
    report=dict(model_id=index['model_id'],model_revision=index['model_revision'],
                checkpoint_sha256=index['checkpoint_sha256'],processor_sha256=index['processor_sha256'],
                bundle_sha256=digest(bundle_path),job_sha256=digest(job_path),
                provenance_scope='Artifact identity checked against external pins. External execution requires independent reproduction.',
                bundle_index_sha256=digest(root/'index.json'),panel_lock_sha256=digest(PANEL/'lock.json'),
                implementation_sha256=digest(__file__),objects=len(imported),
                status='PARTIALLY_COMPLETED',missing_objects=sorted(set(panel)-set(imported)))
    if not report['missing_objects']:
        start=time.monotonic()
        report['results']={kind:replay(panel,measurements,x,ctx,protocol) for kind,x,ctx in feature_sets(panel,imported)}
        report.update(status='COMPLETED',wall_seconds=time.monotonic()-start,
                      visible_views={k:v['views'] for k,v in imported.items()})
    path=Path(out) if out else root.parent/(root.name+'-analysis.json')
    if path.suffix!='.json':path=path/'analysis.json'
    save(path,report)
    return report
