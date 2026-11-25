"""Offline OpenCLIP ViT-B/32 LAION-400M image embeddings on CPU."""
from pathlib import Path
import hashlib
import numpy as np

CHECKPOINT_SHA256 = 'd867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e'


def load_model(checkpoint):
    import torch
    import open_clip
    p=Path(checkpoint)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    if h.hexdigest() != CHECKPOINT_SHA256: raise ValueError('checkpoint SHA-256 mismatch')
    # Local explicit path; no default pretrained model or remote model id.
    model, _, transform = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained=str(p), device='cpu', weights_only=True)
    model.eval()
    return model, transform


def encode_images(model, transform, rgbs, batch_size=16):
    import torch
    from PIL import Image
    if not isinstance(batch_size, int) or batch_size <= 0: raise ValueError('invalid batch size')
    for im in rgbs:
        if not isinstance(im, np.ndarray) or im.ndim != 3 or im.shape[2] != 3 or im.dtype != np.uint8: raise ValueError('images must be HWC RGB uint8')
    vectors=[]
    for i in range(0,len(rgbs),batch_size):
        batch=torch.stack([transform(Image.fromarray(im)) for im in rgbs[i:i+batch_size]])
        with torch.inference_mode(): out=model.encode_image(batch,normalize=True)
        vectors.append(out.cpu().numpy())
    if not vectors: raise ValueError('no images')
    return np.concatenate(vectors)


def pool_views(vectors, n):
    v=np.asarray(vectors,float)
    if v.ndim != 2 or len(v)<n or n not in (1,4,8,16): raise ValueError('invalid views')
    # 16 uniform views, choose every 16/n to yield equivalent orbit for each n.
    if len(v)!=16: raise ValueError('expected 16 ordered views')
    mean=v[::16//n].mean(axis=0); norm=np.linalg.norm(mean)
    if norm<1e-12: raise ValueError('degenerate mean')
    return mean/norm


def encode_patches(model, transform, rgbs, batch_size=8):
    """Last-block patch features (unprojected 768-D), no local-contact claim."""
    import torch
    from PIL import Image
    if not rgbs or batch_size <= 0: raise ValueError('empty images or invalid batch')
    for im in rgbs:
        if not isinstance(im, np.ndarray) or im.ndim != 3 or im.shape[2] != 3 or im.dtype != np.uint8: raise ValueError('images must be HWC RGB uint8')
    chunks=[]
    for i in range(0,len(rgbs),batch_size):
        x=torch.stack([transform(Image.fromarray(im)) for im in rgbs[i:i+batch_size]])
        with torch.inference_mode():
            out=model.visual.forward_intermediates(x,indices=1,normalize_intermediates=True,intermediates_only=True)
        chunks.append(out['image_intermediates'][0].permute(0,2,3,1).cpu().numpy())
    return np.concatenate(chunks)


def visible_patch(point_world, depth, k, world_from_camera, patches, tolerance=0.003):
    """Associate world point to visible rendered pixel and OpenCLIP patch.

    Intended for square images resized to 224 without crop. Nearest-pixel z test;
    tolerance in metres. Returns None for background, occlusion or out of view.
    Patch tokens have global ViT receptive fields and do not establish contact.
    """
    from research_grasping.representations.geometry import project
    depth=np.asarray(depth); patches=np.asarray(patches)
    if depth.ndim != 2 or depth.shape[0] != depth.shape[1] or patches.ndim != 3 or patches.shape[:2] != (7,7) or tolerance < 0:
        raise ValueError('expected square depth and 7x7 patch grid')
    point=np.asarray(point_world,float)
    if point.shape != (3,): raise ValueError('expected 3D point')
    uv,z=project(point[None],k,world_from_camera)
    if z[0] <= 0 or not np.isfinite(uv).all(): return None
    u,v=uv[0]; n=depth.shape[0]
    if u < -.5 or v < -.5 or u >= n-.5 or v >= n-.5: return None
    x,y=int(np.floor(u+.5)),int(np.floor(v+.5))
    if not np.isfinite(depth[y,x]) or abs(depth[y,x]-z[0]) > tolerance: return None
    px=min(6,int(np.floor((u+.5)*7/n))); py=min(6,int(np.floor((v+.5)*7/n)))
    return patches[py,px]
