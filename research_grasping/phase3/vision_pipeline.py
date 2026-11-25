"""Exploratory neutral RGB-D and OpenCLIP on train/validation objects only."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from research_grasping.phase3.render_cpu import cameras, render
from research_grasping.phase3.vision_openclip import load_model, encode_images, encode_patches, pool_views, CHECKPOINT_SHA256


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    import trimesh, torch, open_clip, torchvision
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--out',type=Path,required=True); parser.add_argument('--checkpoint',type=Path,required=True); parser.add_argument('--resolution',type=int,default=96); a=parser.parse_args()
    torch.set_num_threads(4)
    manifest=a.root/'manifest.json'; data=json.loads(manifest.read_text()); rows=[x for x in data['objects'] if x['split'] in ('train','validation')]
    a.out.mkdir(parents=True,exist_ok=True)
    model,transform=load_model(a.checkpoint)
    provenance={'manifest_sha256':sha(manifest), 'checkpoint_sha256':CHECKPOINT_SHA256, 'model':'OpenCLIP ViT-B-32-quickgelu/laion400m_e31', 'torch':torch.__version__,'torchvision':torchvision.__version__,'open_clip':open_clip.__version__, 'transform':str(transform),'views':16,'resolution':a.resolution,'render':'neutral gray, two-sided face shading, no texture or shadow','split_policy':'train and validation only', 'renderer_sha256':sha(Path(__file__).with_name('render_cpu.py')), 'vision_sha256':sha(Path(__file__).with_name('vision_openclip.py')), 'pipeline_sha256':sha(Path(__file__))}
    provenance_hash=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    (a.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    records=[]
    for obj in rows:
        meshfile=(a.root/obj['visual_mesh']).resolve()
        if not meshfile.is_relative_to(a.root.resolve()) or sha(meshfile)!=obj['processed_sha256']: raise ValueError('mesh path or SHA mismatch')
        path=a.out/(obj['object_id']+'.npz'); start=time.perf_counter()
        if path.exists():
            with np.load(path,allow_pickle=False) as cached:
                if str(cached['mesh_sha256']) != obj['processed_sha256'] or str(cached['manifest_sha256']) != provenance['manifest_sha256'] or str(cached['checkpoint_sha256']) != CHECKPOINT_SHA256 or int(cached['resolution']) != a.resolution or str(cached['provenance_sha256']) != provenance_hash: raise ValueError('stale cache')
                vectors=cached['vectors']; elapsed=float(cached['elapsed_seconds'])
        else:
            mesh=trimesh.load_mesh(meshfile,process=False)
            k,ts=cameras(mesh.bounds,16,a.resolution,a.resolution); imgs=[]; depths=[]; masks=[]; norms=[]
            for t in ts:
                frame=render(mesh.vertices,mesh.faces,k,t,a.resolution,a.resolution); imgs.append(frame['rgb']); depths.append(frame['depth'].astype(np.float32)); masks.append(frame['mask']); norms.append(frame['normals_world'].astype(np.float32))
            vectors=encode_images(model,transform,imgs); patches=encode_patches(model,transform,imgs); elapsed=time.perf_counter()-start
            # Partial point clouds are losslessly recoverable from depth, K and T.
            np.savez_compressed(path, rgb=np.array(imgs), depth=np.array(depths), masks=np.array(masks), normals_world=np.array(norms), K=k, world_from_camera=np.array(ts), vectors=vectors, patches=patches, provenance_sha256=provenance_hash, mesh_sha256=obj['processed_sha256'], manifest_sha256=provenance['manifest_sha256'], checkpoint_sha256=CHECKPOINT_SHA256, resolution=a.resolution, elapsed_seconds=elapsed)
        records.append((obj['object_id'],obj['split'],vectors,elapsed)); print(obj['object_id'],obj['split'],round(elapsed,2),flush=True)
    matrices={}; embeddings={}
    for n in (1,4,8,16):
        e=np.stack([pool_views(r[2],n) for r in records]); embeddings[str(n)]=e; matrices[str(n)]=e@e.T
    np.savez_compressed(a.out/'summary.npz',ids=np.array([r[0] for r in records]),splits=np.array([r[1] for r in records]),elapsed=np.array([r[3] for r in records]), **{'embedding_'+k:v for k,v in embeddings.items()}, **{'similarity_'+k:v for k,v in matrices.items()})
    print('done',len(records),flush=True)

if __name__=='__main__': main()
