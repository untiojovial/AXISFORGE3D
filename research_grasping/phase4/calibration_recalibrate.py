"""Nested grouped variance scaling of already-computed outer DKL predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import warnings
import numpy as np
from research_grasping.phase3.experiment import digest, load, save
from research_grasping.phase4.calibration import fold_design, load_training, metrics


def inner_groups(objects, target):
    names=sorted(set(objects)-{target})
    if target not in objects or len(names)!=15:
        raise ValueError('expected one outer and 15 inner objects')
    return [names[i::3] for i in range(3)]


def run(manifest,catalogs,history,outer_path):
    import torch
    from axisforge_bo.deep_kernel import DeepKernelGP
    torch.set_num_threads(1)
    old=load(outer_path)
    if old['manifest_sha256']!=digest(manifest):raise ValueError('manifest mismatch')
    local,context,y,objects,ids,sources=load_training(manifest,catalogs,history)
    if old['sources']!=sources:raise ValueError('history/catalogue mismatch')
    output={'schema':1,'outer_sha256':digest(outer_path),'script_sha256':digest(__file__),'folds':[]}
    for fold in old['folds']:
        start=time.monotonic(); target=fold['object_id']; targets=objects==target
        if fold['candidate_ids']!=ids[targets].tolist() or not np.array_equal(fold['y'],y[targets]):
            raise ValueError('outer predictions do not match input labels')
        groups=inner_groups(objects, target); scores={}; warn=[]
        for held in groups:
            inner_test=np.isin(objects,held);inner_train=(objects!=target)&~inner_test
            x,xt=fold_design(local,context,inner_train,inner_test)
            avg=y[inner_train].mean();scale=y[inner_train].std()
            if scale<1e-10:raise ValueError('degenerate training outcome')
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                gp=DeepKernelGP(x.shape[1],hidden_dim=8,feature_dim=2,seed=11).fit(x,(y[inner_train]-avg)/scale,steps=40)
                pred,var=gp.predict(xt);var=np.maximum((var+float(gp.log_noise.detach().exp()))*scale**2,1e-14)
            warn.extend(str(w.message) for w in ws)
            pred=avg+scale*pred; o=objects[inner_test];obs=y[inner_test]
            for name in held:scores[name]=float(np.mean(((obs[o==name]-pred[o==name])**2)/var[o==name]))
        factor=float(np.clip(np.mean(list(scores.values())),1,100));raw=fold['models']['dkl_11']
        mean=np.array(raw['prediction_mean']);var=np.array(raw['prediction_variance']); adjusted=metrics(y[targets],mean,var*factor)
        output['folds'].append({'object_id':target,'inner_groups':groups,'object_scores':scores,'factor':factor,'warnings':warn,
                                'calibration_elapsed_s':time.monotonic()-start,'raw':{k:raw[k] for k in ('mae','rmse','nlpd','coverage90','width90')},
                                'adjusted':adjusted,'prediction_mean':mean.tolist(),'prediction_variance':(var*factor).tolist()})
        print(target,round(factor,3),flush=True)
    output['macro']={label:{key:float(np.mean([f[label][key] for f in output['folds']])) for key in ('mae','rmse','nlpd','coverage90','width90')} for label in ('raw','adjusted')}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','catalogs','history','outer','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();save(a.out,run(a.manifest,a.catalogs,a.history,a.outer))

if __name__=='__main__':main()
