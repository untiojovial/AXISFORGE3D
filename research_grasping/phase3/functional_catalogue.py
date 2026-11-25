"""Pre-query contact geometry and train-only local visual PCA for development."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .contact_similarity import descriptors
from .experiment import DevelopmentData, fingerprint, load, save
from .vision_integrity import read_lock


def contact_geometry(candidate, feature, reference=.1):
    """Contact mean/std/min/max in closing, transverse, approach frame."""
    f=np.asarray(feature,float); centre=np.asarray(candidate['position'],float); p=np.asarray(candidate['contacts'],float)
    if f.shape != (16,) or centre.shape != (3,) or p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all() or reference<=0: raise ValueError('invalid candidate')
    approach=f[3:6]; na=np.linalg.norm(approach)
    if na<1e-8: raise ValueError('degenerate approach')
    approach=approach/na; closing=f[6:9]-approach*np.dot(f[6:9],approach); nc=np.linalg.norm(closing)
    if nc<1e-8: raise ValueError('degenerate closing axis')
    closing/=nc; transverse=np.cross(approach,closing)
    xyz=(p-centre) @ np.column_stack((closing,transverse,approach))/reference
    return np.concatenate((xyz.mean(0),xyz.std(0),xyz.min(0),xyz.max(0)))


def fit_pca(rows, components=4):
    """Object-balanced covariance of visible candidates from train only."""
    if not rows or any(r['split']!='train' for r in rows): raise ValueError('PCA requires train-only rows')
    selected=[r for r in rows if r['views']>0]
    if len(selected)<components+1: raise ValueError('insufficient visible train candidates')
    counts={k:sum(r['object_id']==k for r in selected) for k in {r['object_id'] for r in selected}}
    if len(counts)!=len({r['object_id'] for r in rows}): raise ValueError('a train object has no visible candidates')
    x=np.array([r['local'] for r in selected]); w=np.array([1/counts[r['object_id']] for r in selected]); w/=w.sum()
    mean=w@x; _,singular,vt=np.linalg.svd((x-mean)*np.sqrt(w[:,None]),full_matrices=False)
    if len(singular)<components or singular[components-1]<1e-10: raise ValueError('insufficient PCA rank')
    vectors=vt[:components].copy()
    for vec in vectors:
        if vec[np.argmax(np.abs(vec))]<0: vec *= -1
    ids=sorted((r['object_id'],str(r['cid'])) for r in selected)
    return mean,vectors,fingerprint(ids)


def main():
    p=argparse.ArgumentParser()
    for name in ('data','catalogues','protocol','binary','vision','out'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); guard=DevelopmentData(a.protocol/'manifest.prepilot.json',load(a.protocol/'lock.json'),a.data)
    lock=read_lock(); catalogs=[]; allrows=[]
    for path in sorted(a.catalogues.glob('*.json')):
        c=guard.catalogue(path,a.binary); oid=c['object_id']
        rows=descriptors(c,c['ids'],a.vision/(oid+'.npz'),guard.manifest_digest)
        for r in rows: r.update(object_id=oid,split=c['split'])
        catalogs.append((path,c,rows)); allrows.extend(rows)
    if len(catalogs)!=22 or sum(c['split']=='train' for _,c,_ in catalogs)!=16 or sum(c['split']=='validation' for _,c,_ in catalogs)!=6: raise ValueError('expected complete development catalogue set')
    input_digest=fingerprint([(c['object_id'],fingerprint(c),lock['objects'][c['object_id']]['vision_semantic_sha256']) for _,c,_ in catalogs])
    train=[r for r in allrows if r['split']=='train']; mean,components,ids_hash=fit_pca(train)
    out=a.out; out.mkdir(parents=True,exist_ok=True)
    pca={'schema':'object-balanced-visible-train-pca-v1','mean':mean.tolist(),'components':components.tolist(),'fit_ids_sha256':ids_hash,'objects':sorted({r['object_id'] for r in train}),'manifest_sha256':guard.manifest_digest,'batch_input_sha256':input_digest,'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'vision_lock_sha256':hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest(),'visible_fit_count':sum(r['views']>0 for r in train)}
    pca_hash=fingerprint(pca); save(out/'pca.json',pca)
    for path,c,rows in catalogs:
        augmented=[]
        for f,cid,row in zip(c['features'],c['ids'],rows,strict=True):
            local=(np.asarray(row['local'])-mean)@components.T if row['views']>0 else np.zeros(len(components))
            augmented.append(list(f)+contact_geometry(c['candidates'][cid],f).tolist()+[float(row['views']>0)]+local.tolist())
        c['features']=augmented
        c['metadata']['functional_features']={'prototype':True,'original_catalogue_sha256':fingerprint(load(path)), 'pca_sha256':pca_hash,'batch_input_sha256':input_digest,'vision_semantic_sha256':lock['objects'][c['object_id']]['vision_semantic_sha256'],'definition':'original 16 + contact mean/std/min/max in closing, transverse, approach frame / 0.1m (12) + visible flag + train-only PCA (4); absent token zero with flag 0','training_labels_used':False}
        save(out/(c['object_id']+'.json'),c)
    print('objects',len(catalogs),'train visible',pca['visible_fit_count'],'fit_ids_sha256',ids_hash,flush=True)

if __name__=='__main__': main()
