"""Per-candidate local features at the two contacts: DINOv3 patch tokens and local geometry.

    python tools/research/phase7/local_features.py --store store.json --projects POOL --catalogues cat80/ \
        --names names.txt --checkpoint models/dinov3-timm-vits16 --out-dir local/ --jobs 2

For each part (PHASE7_PROTOCOL.md, I.3): 8 views at 224 px with the scale-normalised Phase III
cameras; DINOv3 ViT-S/16 patch tokens (14 x 14); each contact (centre +- closing * width/2) is
projected into every view, counted visible when the rendered depth at its pixel is within 3 mm of its
own depth, and the patch grid is sampled bilinearly there (each contact is first moved to the
nearest of 100 000 samples of the render mesh: the planner's contacts lie on its own surface model,
a median 6 mm away; the offset is stored); visible views are averaged. A contact seen
in no view gets zeros and a zero visibility count. Local geometry per contact, from 20 000 surface
samples within 10 mm: linearity, planarity and sphericity of the covariance eigenvalues, and the angle
between the local plane normal and the contact normal.
Writes <out-dir>/<part>.npz with dino_a, dino_b (float16, n x 384), vis_a, vis_b, geo_a, geo_b (n x 4).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase5 import catalogue  # noqa: E402

VIEWS, SIZE, PATCH, TOL, RADIUS = 8, 224, 16, .003, .010
STA = {}


def init(a):
    from research_grasping.phase4.frozen_models import load_frozen
    import torch
    torch.set_num_threads(1)
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    STA.update(a=a, store=json.loads(Path(a.store).read_text()),
               encoder=load_frozen('dinov3-timm', a.checkpoint, config_sha256=a.config_sha256,
                                   weights_sha256=a.weights_sha256, model_revision=a.revision), cfg=cfg)


def sample_grid(grid, u, v):
    """Bilinear sample of a (gh, gw, c) patch grid at pixel (u, v)."""
    gh, gw, _ = grid.shape
    x = np.clip(u/PATCH - .5, 0, gw-1); y = np.clip(v/PATCH - .5, 0, gh-1)
    x0, y0 = int(np.floor(x)), int(np.floor(y)); x1, y1 = min(x0+1, gw-1), min(y0+1, gh-1)
    fx, fy = x-x0, y-y0
    return ((1-fx)*(1-fy)*grid[y0, x0] + fx*(1-fy)*grid[y0, x1] + (1-fx)*fy*grid[y1, x0] + fx*fy*grid[y1, x1])


def local_geometry(points, tree, p, n):
    idx = tree.query_ball_point(p, RADIUS)
    if len(idx) < 5:
        return np.zeros(4)
    q = points[idx] - points[idx].mean(0)
    w, vec = np.linalg.eigh(q.T@q/len(q))
    l3, l2, l1 = np.maximum(w, 1e-12)
    normal = vec[:, 0]
    ang = np.degrees(np.arccos(min(1., abs(float(normal@n)/max(np.linalg.norm(n), 1e-12)))))
    return np.array([(l1-l2)/l1, (l2-l3)/l1, l3/l1, ang/90.])


def job(name):
    from scipy.spatial import cKDTree
    import trimesh
    from research_grasping.phase3.render_cpu import cameras, render
    from research_grasping.phase4.visual_prior import part_mesh
    s = STA
    a = s['a']
    out = Path(a.out_dir)/f'{name}.npz'
    if out.exists():
        return name, 'cached'
    start = time.perf_counter()
    vertices, faces = part_mesh(Path(a.projects)/s['store'][name]['project'], name)
    bounds = np.array([vertices.min(0), vertices.max(0)])
    k, transforms = cameras(bounds, VIEWS, SIZE, SIZE)
    views = [render(vertices, faces, k, t, SIZE, SIZE) for t in transforms]
    t_render = time.perf_counter() - start
    enc = s['encoder'].encode([v['rgb'] for v in views])
    gh, gw = enc['token_geometry']['grid_shape']
    grids = np.asarray(enc['patch_embeddings'], np.float32).reshape(len(views), gh, gw, -1)
    t_encode = time.perf_counter() - start - t_render
    cat = catalogue.load(Path(a.catalogues)/f'{name}.npz')
    c = np.asarray(cat['center'], float); kk = np.asarray(cat['closing'], float)
    kk = kk/np.linalg.norm(kk, axis=1, keepdims=True); w = np.asarray(cat['width'], float)[:, None]
    contacts = {'a': c + kk*w/2, 'b': c - kk*w/2}
    normals = {'a': np.asarray(cat['normalA'], float), 'b': np.asarray(cat['normalB'], float)}
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    points, _ = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    tree = cKDTree(points)
    dense, _ = trimesh.sample.sample_surface(mesh, 100000, seed=1)
    dense_tree = cKDTree(dense)
    res = {}
    for side, P in contacts.items():
        # the planner's contacts lie on its own surface model, a few mm off the render mesh:
        # use the nearest visual-surface point, and keep the offset as a feature
        offset, nearest = dense_tree.query(P)
        P = dense[nearest]
        res[f'offset_{side}'] = offset.astype(np.float32)
        feat = np.zeros((len(P), grids.shape[-1]), np.float32); vis = np.zeros(len(P), np.int8)
        for view, grid in zip(views, grids):
            t = view['world_from_camera']; K = view['K']
            cam = (P - t[:3, 3]) @ t[:3, :3]
            z = cam[:, 2]
            uv = (cam @ K.T)[:, :2]/z[:, None]
            for i in range(len(P)):
                u, v = uv[i]
                if not (0 <= u < SIZE and 0 <= v < SIZE and z[i] > 0):
                    continue
                d = view['depth'][int(v), int(u)]
                if np.isfinite(d) and abs(d - z[i]) < TOL:
                    feat[i] += sample_grid(grid, u, v); vis[i] += 1
        feat[vis > 0] /= vis[vis > 0, None]
        res[f'dino_{side}'] = feat.astype(np.float16); res[f'vis_{side}'] = vis
        res[f'geo_{side}'] = np.stack([local_geometry(points, tree, P[i], normals[side][i]) for i in range(len(P))]).astype(np.float32)
    np.savez_compressed(out, **res, seconds=np.array([t_render, t_encode, time.perf_counter()-start]))
    return name, round(time.perf_counter()-start, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('store', 'projects', 'catalogues', 'names', 'checkpoint', 'out-dir'):
        p.add_argument(f'--{k}', required=True)
    for k in ('revision', 'weights-sha256', 'config-sha256', 'config'):
        p.add_argument(f'--{k}')
    p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    names = [s.strip() for s in Path(a.names).read_text().splitlines() if s.strip()]
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=init, initargs=(a,)) as pool:
        for name, s in pool.map(job, names):
            print(name, s, flush=True)


if __name__ == '__main__':
    main()
