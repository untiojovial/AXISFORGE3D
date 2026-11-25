"""Paired real 224 raster vs resized 96, same development object/camera only."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from research_grasping.phase3.render_cpu import render
from research_grasping.phase3.vision_integrity import verify_visual, semantic_npz_sha256
from research_grasping.phase3.experiment import PREPILOT_MANIFEST
from .frozen_models import load_frozen


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    import torch, trimesh, torchvision, open_clip
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4); manifest=a.root/'manifest.json'; objs=[x for x in json.loads(manifest.read_text())['objects'] if x['split'] in ('train','validation')]
    if sha(manifest)!=PREPILOT_MANIFEST: raise ValueError('manifest lock mismatch')
    model=load_frozen('openclip',a.checkpoint); results=[]
    for obj in objs:
        oid=obj['object_id']; source=a.root/'vision-development-v3'/(oid+'.npz'); verify_visual(source,oid)
        meshfile=(a.root/obj['visual_mesh']).resolve()
        if not meshfile.is_relative_to(a.root.resolve()) or sha(meshfile)!=obj['processed_sha256']:raise ValueError('mesh mismatch')
        mesh=trimesh.load_mesh(meshfile,process=False)
        with np.load(source,allow_pickle=False) as z:
            rgb=z['rgb'][0]; k=z['K']; pose=z['world_from_camera'][0]
        if rgb.shape!=(96,96,3):raise ValueError('unexpected input resolution')
        scale=224/96; affine=np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1]])
        start=time.perf_counter(); frame=render(mesh.vertices,mesh.faces,affine@k,pose,224,224)
        low=model.encode([rgb]); high=model.encode([frame['rgb']]); e=np.concatenate((low['global_embeddings'],high['global_embeddings'])); lp=low['patch_embeddings'][0]; hp=high['patch_embeddings'][0]; patch_cos=np.sum(lp*hp,axis=1)/(np.linalg.norm(lp,axis=1)*np.linalg.norm(hp,axis=1)); elapsed=time.perf_counter()-start
        results.append(dict(object_id=oid,split=obj['split'],observation_semantic_sha256=semantic_npz_sha256(source),mesh_sha256=obj['processed_sha256'],cosine=float(e[0]@e[1]),patch_cosines=patch_cos.tolist(),render_inference_seconds=elapsed));print(oid,round(results[-1]['cosine'],4),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'status':'exploratory; one view per object; no physical outcomes','manifest_sha256':sha(manifest),'checkpoint_sha256':model.provenance['weights_sha256'],'processor':model.provenance['processor'],'software':{'torch':torch.__version__,'torchvision':torchvision.__version__,'open_clip':open_clip.__version__,'trimesh':trimesh.__version__},'study_code_sha256':sha(Path(__file__)),'frozen_code_sha256':sha(Path(__file__).with_name('frozen_models.py')),'renderer_code_sha256':sha(Path(__file__).parents[1]/'phase3/render_cpu.py'),'results':results},indent=2)+'\n')

if __name__=='__main__': main()
