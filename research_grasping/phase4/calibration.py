"""Training-object LOO predictive calibration, native F2, development only."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import platform
import time
import warnings
import numpy as np
from scipy.special import ndtri
from research_grasping.phase3.experiment import digest, fingerprint, load, save


def metrics(y, mean, variance):
    y=np.asarray(y,float); mean=np.asarray(mean,float); variance=np.asarray(variance,float)
    if y.ndim!=1 or mean.shape!=y.shape or variance.shape!=y.shape or not len(y) or not all(np.isfinite(x).all() for x in (y,mean,variance)) or np.any(variance<0):
        raise ValueError('invalid observations/predictions')
    variance=np.maximum(variance,1e-14); error=y-mean; radius=ndtri(.95)*np.sqrt(variance)
    return {'n':len(y),'zeros':int(np.sum(y==0)), 'mae':float(np.mean(np.abs(error))),
            'rmse':float(np.sqrt(np.mean(error**2))),
            'nlpd':float(np.mean(.5*(np.log(2*np.pi*variance)+error**2/variance))),
            'coverage90':float(np.mean(np.abs(error)<=radius)),'width90':float(np.mean(2*radius))}


def fold_design(local, context, train, test):
    a=local[train]; b=local[test]
    # Four unweighted global PCs, fitted only on training objects/rows.
    c=context[train];d=context[test];cm=c.mean(axis=0);c=c-cm;d=d-cm
    _,s,vt=np.linalg.svd(c,full_matrices=False)
    basis=vt[:min(4,np.count_nonzero(s>1e-9))]
    a=np.concatenate((a,c@basis.T),axis=1);b=np.concatenate((b,d@basis.T),axis=1)
    center=a.mean(axis=0);scale=a.std(axis=0);scale[scale<1e-8]=1
    return (a-center)/scale,(b-center)/scale


def load_training(manifest_path, catalogs, run):
    manifest=load(manifest_path); records={r['object_id']:r for r in manifest['objects']}
    names=sorted(k for k,r in records.items() if r['split']=='train')
    if len(names)!=16 or len(records)!=len(manifest['objects']): raise ValueError('unexpected train manifest')
    local=[];context=[];target=[];obj=[];ids=[];sources={}
    for oid in names:
        catpath=catalogs/(oid+'.json');logpath=run/'history'/(oid+'.json')
        cat=load(catpath);log=load(logpath);record=records[oid]
        if (cat['split']!='train' or cat['object_id']!=oid or log['object_id']!=oid or
            cat['metadata']['manifest_sha256']!=digest(manifest_path) or
            cat['metadata']['processed_sha256']!=record['processed_sha256'] or
            log['catalogue_digest']!=fingerprint(cat) or log['fidelity']!='native-f12'):
            raise ValueError('history provenance mismatch')
        if len(set(log['selected']))!=len(log['selected']) or [r['candidate_id'] for r in log['rows']]!=log['selected']:
            raise ValueError('history incomplete or duplicate')
        for row in log['rows']:
            if row['status'] not in ('ok','physical_failure') or row['candidate_id'] not in cat['ids']:
                raise ValueError('infrastructure failure or unknown candidate')
            i=cat['ids'].index(row['candidate_id']);y=float(row['details']['epsilon'])
            if not np.isfinite(y): raise ValueError('nonfinite epsilon')
            local.append(cat['features'][i]);context.append(cat['context'][3:]);target.append(y);obj.append(oid);ids.append(row['candidate_id'])
        sources[oid]={'catalogue_sha256':digest(catpath),'history_sha256':digest(logpath),'n':len(log['rows'])}
    return *(np.array(x) for x in (local,context,target,obj,ids)),sources


def run(manifest,catalogs,history):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF,Matern,WhiteKernel,ConstantKernel
    from axisforge_bo.deep_kernel import DeepKernelGP
    import sklearn, scipy, torch
    torch.set_num_threads(1)
    local,context,y,objects,ids,sources=load_training(manifest,catalogs,history)
    report={'schema':1,'outcome':'native F2 epsilon including F1 rejection',
            'manifest_sha256':digest(manifest),'script_sha256':digest(__file__),'sources':sources,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__},'folds':[]}
    for name in sorted(set(objects)):
        tr=objects!=name;te=~tr;x,xt=fold_design(local,context,tr,te)
        avg=y[tr].mean();scale=y[tr].std()
        if scale<1e-10: raise ValueError('degenerate fold training outcome')
        z=(y[tr]-avg)/scale; results={}
        def put(label,mu,var,start,config,warn):
            m=metrics(y[te],mu,var);m.update({'prediction_mean':np.asarray(mu).tolist(),'prediction_variance':np.asarray(var).tolist(),
                                            'elapsed_s':time.monotonic()-start,'config':config,'warnings':warn});results[label]=m
        start=time.monotonic();put('constant',np.full(te.sum(),avg),np.full(te.sum(),scale**2),start,'train mean/variance',[])
        for key,base in [('rbf',RBF(1.,(.05,100.))),('matern52',Matern(1.,(.05,100.),nu=2.5)),('matern52_ard',Matern(np.ones(x.shape[1]),(.05,100.),nu=2.5))]:
            start=time.monotonic()
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(.01,25.))*base+WhiteKernel(.05,(1e-5,5.)),alpha=1e-8,random_state=11).fit(x,z)
                mu,std=gp.predict(xt,return_std=True)
            put(key,avg+scale*mu,(std*scale)**2,start,str(gp.kernel_),[str(w.message) for w in observed])
        for seed in (11,29,47):
            start=time.monotonic()
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                gp=DeepKernelGP(x.shape[1],hidden_dim=8,feature_dim=2,seed=seed).fit(x,z,steps=40)
                mu,var=gp.predict(xt); var=(var+float(gp.log_noise.detach().exp()))*scale**2
            put('dkl_'+str(seed),avg+scale*mu,var,start,'tanh 8, latent 2, 40 steps, seed '+str(seed),[str(w.message) for w in observed])
        report['folds'].append({'object_id':name,'candidate_ids':ids[te].tolist(),'y':y[te].tolist(),'models':results})
        print(name,flush=True)
    report['macro']={key:{metric:float(np.mean([fold['models'][key][metric] for fold in report['folds']]))
                        for metric in ('mae','rmse','nlpd','coverage90','width90')}
                      for key in report['folds'][0]['models']}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','catalogs','history','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();save(a.out,run(a.manifest,a.catalogs,a.history))

if __name__=='__main__': main()
