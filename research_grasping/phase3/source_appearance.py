"""Recover GSO atlas colours on calibrated RGB-D surface samples and export clouds."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from research_grasping.representations.geometry import backproject
from .experiment import PREPILOT_MANIFEST, digest, load, save
from .vision_openclip import encode_images, load_model, pool_views
from .vision_integrity import verify_visual, verify_texture


def triangles_uv(path):
    vertices, texcoords, faces, texfaces = [], [], [], []
    for line in path.read_text().splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':vertices.append(list(map(float,fields[1:4])))
        elif fields[0]=='vt':texcoords.append(list(map(float,fields[1:3])))
        elif fields[0]=='f':
            if len(fields)!=4:raise ValueError('expected triangulated textured OBJ')
            ix=[f.split('/') for f in fields[1:]]
            faces.append([int(f[0])-1 for f in ix]);texfaces.append([int(f[1])-1 for f in ix])
    if not faces or np.min(faces)<0 or np.min(texfaces)<0:
        raise ValueError('requires positive source OBJ vertex and UV indices')
    return np.asarray(vertices)[faces],np.asarray(texcoords)[texfaces]


def atlas_samples(points, triangles, uv, atlas, tree, tolerance_m=2e-6):
    """Accept UV interpolation only on a triangle containing the metric point."""
    from trimesh.triangles import points_to_barycentric
    rgb=np.zeros((len(points),3),np.uint8)
    valid=np.zeros(len(points),bool)
    for k in (8,32,128):
        missing=np.flatnonzero(~valid)
        if not len(missing):break
        indices=tree.query(points[missing],k=min(k,len(triangles)))[1].reshape(len(missing),-1)
        tris=triangles[indices]
        xyz=np.broadcast_to(points[missing,None,:],indices.shape+(3,))
        with np.errstate(divide='ignore',invalid='ignore'):
            bary=points_to_barycentric(tris.reshape(-1,3,3),xyz.reshape(-1,3)).reshape(xyz.shape)
        err=np.linalg.norm(np.einsum('nki,nkij->nkj',bary,tris)-xyz,axis=2)
        inside=np.all((bary>=-1e-4)&(bary<=1.0001),axis=2)
        err=np.where(inside & np.isfinite(err),err,np.inf)
        column=err.argmin(axis=1)
        ok=np.flatnonzero(err[np.arange(len(missing)),column] <= tolerance_m)
        if not len(ok):continue
        face=indices[ok,column[ok]]
        tex=np.einsum('ni,nij->nj',bary[ok,column[ok]],uv[face]).clip(0,1)
        h,w=atlas.shape[:2]
        rgb[missing[ok]]=atlas[np.rint((1-tex[:,1])*(h-1)).astype(int),np.rint(tex[:,0]*(w-1)).astype(int)]
        valid[missing[ok]]=True
    return rgb,valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('root','vision','checkpoint','out','summary'):
        p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args()
    if digest(a.root/'manifest.json')!=PREPILOT_MANIFEST:raise ValueError('unexpected manifest')
    model,transform=load_model(a.checkpoint)
    a.out.mkdir(parents=True,exist_ok=True)
    report=[]
    for r in load(a.root/'manifest.json')['objects']:
        if r['split']=='test' or not r['texture_availability']:continue
        oid=r['object_id']
        orig=a.root/r['original_mesh'];texture=orig.parent/'texture.png'
        if digest(orig)!=r['original_sha256']:raise ValueError('original mesh mismatch')
        verify_texture(texture,oid)
        verify_visual(a.vision/(oid+'.npz'),oid)
        tris,uv=triangles_uv(orig);tree=cKDTree(tris.mean(axis=1))
        with Image.open(texture) as source:atlas=np.asarray(source.convert('RGB'))
        with np.load(a.vision/(oid+'.npz'),allow_pickle=False) as f:
            if str(f['mesh_sha256'])!=r['processed_sha256']:raise ValueError('RGB-D/mesh mismatch')
            depths,k,poses,vectors=f['depth'],f['K'],f['world_from_camera'],f['vectors']
        images=[];pointsets=[];colours=[];assigned=0
        for depth,pose in zip(depths,poses,strict=True):
            xyz,pixels=backproject(depth,k,pose)
            rgb,good=atlas_samples(xyz,tris,uv,atlas,tree)
            image=np.zeros(depth.shape+(3,),np.uint8)
            image[pixels[:,1],pixels[:,0]]=rgb
            images.append(image);pointsets.append(xyz.astype('f4'));colours.append(rgb);assigned+=int(good.sum())
        embeddings=encode_images(model,transform,images)
        offsets=np.cumsum([0]+[len(x) for x in pointsets])
        np.savez_compressed(a.out/(oid+'.npz'),rgb=np.array(images),points=np.concatenate(pointsets),
                            colours=np.concatenate(colours),view_offsets=offsets,embeddings=embeddings)
        report.append(dict(object_id=oid,split=r['split'],samples=int(offsets[-1]),assigned=assigned,
                           neutral_vs_atlas_cosine=float(pool_views(vectors,16)@pool_views(embeddings,16)),
                           texture_sha256=digest(texture)))
        save(a.summary,dict(manifest_sha256=PREPILOT_MANIFEST,objects=report))
        print(oid,assigned,int(offsets[-1]),flush=True)


if __name__=='__main__':
    main()
