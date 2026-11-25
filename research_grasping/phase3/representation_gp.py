"""Train-only multimodal feature compression and a held-out F2 regression comparison."""
import argparse
from pathlib import Path
import platform
import importlib.metadata
import time

import numpy as np

from .contact_similarity import descriptors
from .vision_openclip import CHECKPOINT_SHA256
from .experiment import DevelopmentData, digest, fingerprint, load, save


def design(train, validation):
    """Estimate centring, scales and per-modality PCA directions on train only."""
    xs,xt=[],[]
    for key in ('geometry','global_embedding','local'):
        a=np.array([r[key] for r in train]);b=np.array([r[key] for r in validation])
        mean=a.mean(axis=0)
        a=a-mean;b=b-mean
        if key!='geometry':
            _,sv,vt=np.linalg.svd(a,full_matrices=False)
            basis=vt[:min(4,np.count_nonzero(sv>1e-9))]
            a=a@basis.T;b=b@basis.T
        scale=a.std(axis=0)
        scale[scale<1e-8]=1.
        xs.append(a/scale);xt.append(b/scale)
    return np.concatenate(xs,axis=1),np.concatenate(xt,axis=1)


def main():
    from axisforge_bo.deep_kernel import DeepKernelGP
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF,Matern,WhiteKernel,ConstantKernel
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('data','vision','protocol','catalogues','binary','run','out'):
        p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args();data=DevelopmentData(a.protocol/'manifest.prepilot.json',load(a.protocol/'lock.json'),a.data)
    train=[];valid=[];y=[];targets=[];names=[]
    inputs={}
    for path in sorted(a.catalogues.glob('*.json')):
        c=data.catalogue(path,a.binary);oid=c['object_id']
        training=c['split']=='train'
        log=load(a.run/'history'/(oid+'.json') if training else a.run/'runs'/oid/'11-random.json')
        definition=log if training else log['definition']
        if fingerprint(c)!=definition['catalogue_digest']:raise ValueError('archive mismatch')
        log_path = a.run/'history'/(oid+'.json') if training else a.run/'runs'/oid/'11-random.json'
        inputs[oid] = {'catalogue_sha256': digest(path), 'observations_sha256': digest(log_path),
                       'vision_archive_sha256': digest(a.vision/(oid+'.npz')), 'split': c['split']}
        rows=log['rows']
        if any(r['status']=='infrastructure_failure' for r in rows):raise ValueError('incomplete sample')
        ids=[r['candidate_id'] for r in rows]
        feat=descriptors(c,ids,a.vision/(oid+'.npz'),data.manifest_digest)
        if training:
            train.extend(feat);y.extend(r['details']['epsilon'] for r in rows)
        else:
            valid.extend(feat);targets.extend(r['details']['epsilon'] for r in rows);names.extend([oid]*len(ids))
    x,xt=design(train,valid)
    y=np.array(y);mean=y.mean();scale=y.std()
    if scale<1e-10:raise ValueError('insufficient training label variation')
    z=(y-mean)/scale
    # Baseline estimates its mean and variance exclusively from training labels.
    preds={'train_constant': (np.full(len(targets), mean), np.full(len(targets), max(scale**2, 1e-14)),
                              0., 'training mean and population variance')}
    for key,kernel in [('rbf',RBF(1.,(.05,100.))),('matern52',Matern(1.,(.05,100.),nu=2.5))]:
        start=time.monotonic()
        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(.01,25.))*kernel+WhiteKernel(.05,(1e-5,5.)),
                                    alpha=1e-8,random_state=11).fit(x,z)
        mu,std=gp.predict(xt,return_std=True)
        preds[key]=(mean+scale*mu,(std*scale)**2,time.monotonic()-start,str(gp.kernel_))
    for seed in (11,29,47):
        start=time.monotonic()
        gp=DeepKernelGP(x.shape[1],hidden_dim=8,feature_dim=2,seed=seed).fit(x,z,steps=40)
        mu,var=gp.predict(xt)
        variance=(var+float(gp.log_noise.detach().exp()))*scale**2
        preds['dkl_'+str(seed)]=(mean+mu*scale,variance,time.monotonic()-start,'tanh 8, latent 2, 40 Adam steps')
    # Validation labels enter only evaluation. Absolute error and predictive
    # density are in native wrench-epsilon units, separately for each object.
    targets=np.asarray(targets);names=np.array(names)
    report=dict(train=len(y),target=len(names),input_dimensions=x.shape[1],outcome='native F2 epsilon',models={},
        provenance={'selection':'training history and validation random seed 11; all recorded attempts',
                    'manifest_sha256':data.manifest_digest, 'binary_sha256':digest(a.binary),
                    'python':platform.python_version(), 'numpy':np.__version__,
                    'scipy':importlib.metadata.version('scipy'),
                    'torch':importlib.metadata.version('torch'),
                    'scikit_learn':importlib.metadata.version('scikit-learn'),
                    'checkpoint_sha256':CHECKPOINT_SHA256, 'script_sha256':digest(__file__),
                    'descriptor_script_sha256':digest(Path(__file__).with_name('contact_similarity.py')),
                    'lock_sha256':digest(a.protocol/'lock.json'), 'inputs':inputs})
    for key,(mu,var,elapsed,config) in preds.items():
        grouped={}
        for oid in sorted(set(names)):
            idx=names==oid;residual=mu[idx]-targets[idx];v=np.maximum(var[idx],1e-14)
            grouped[str(oid)]=dict(mae=float(np.abs(residual).mean()),
                                 nlpd=float((.5*(np.log(2*np.pi*v)+residual**2/v)).mean()),
                                 coverage90=float((np.abs(residual)<=1.64485362695*np.sqrt(v)).mean()))
        macro={metric:float(np.mean([values[metric] for values in grouped.values()]))
               for metric in ('mae', 'nlpd', 'coverage90')}
        report['models'][key]=dict(by_object=grouped, object_macro=macro, config=config,elapsed_s=elapsed)
    save(a.out,report)


if __name__=='__main__':
    main()
