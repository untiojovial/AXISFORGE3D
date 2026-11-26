"""GPD classifier as an examination order for AxisForge candidates (ranking-controlled external baseline).

    python tools/research/phase7_extension/gpd_baseline.py score --adapter gpd_score_candidates --gpd-source GPD_DIR \
        --names names.txt --catalogues cat80/ --store store.json --projects POOL --out-dir gpd_scores/ --jobs 1

For each object: two simulated depth cameras (the project's CPU z-buffer renderer, 160 x 160 px, 50 deg
field of view, 0.45 m from the part centre, 45 deg elevation, azimuths 180 +/- 40 deg, i.e. between the
robot and the part) give two point clouds; GPD's own preprocessing and classifier (via the adapter,
tools/research/phase7_extension/gpd_adapter) score every catalogue candidate. The scores become an
examination order with the planner's stroke gate first, exactly like any other prior here
(research_grasping/phase6/fusion.order). This is not GPD's full pipeline: GPD does not generate the
candidates and its hand geometry is not the AxisForge gripper's (see external_baselines_v2.json).
Outputs one JSON per object (scores, timings, cloud sizes), written when complete (restart-safe).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research_grasping.phase3.render_cpu import look_at, render  # noqa: E402
from research_grasping.phase4 import visual_prior as vp  # noqa: E402
from research_grasping.phase5 import catalogue  # noqa: E402

SIZE, FOV_DEG, DIST, ELEV, AZ = 160, 50., .45, 45., (140., 220.)


def write_pcd(path, pts):
    """ASCII PCD with the rgba field GPD's PointXYZRGBA clouds need (constant grey; colour is not used)."""
    with open(path, 'w') as f:
        f.write(f'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgba\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH {len(pts)}\n'
                f'HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA ascii\n')
        for x, y, z in pts:
            f.write(f'{x:.6f} {y:.6f} {z:.6f} 4286611584\n')


def views(v, f):
    centre = (v.min(0) + v.max(0))/2
    fx = SIZE/(2*np.tan(np.radians(FOV_DEG)/2))
    k = np.array([[fx, 0, (SIZE-1)/2], [0, fx, (SIZE-1)/2], [0, 0, 1.]])
    out = []
    for az in AZ:
        d = np.array([np.cos(np.radians(az))*np.cos(np.radians(ELEV)), np.sin(np.radians(az))*np.cos(np.radians(ELEV)), np.sin(np.radians(ELEV))])
        pos = centre + DIST*d
        r = render(v, f, k, look_at(pos, centre), SIZE, SIZE)
        pts = np.asarray(r['points_world'], float)  # already the masked, back-projected pixels
        out.append((pos, pts.reshape(-1, 3)))
    return out


def score_one(task):
    a, n = task
    path = Path(a.out_dir)/f'{n}.json'
    if path.exists():
        return n, 'cached'
    store = json.loads(Path(a.store).read_text())
    cat = catalogue.load(Path(a.catalogues)/f'{n}.npz')
    t0 = time.perf_counter()
    v, f = vp.part_mesh(Path(a.projects)/store[n]['project'], n)
    cams = views(v, f)
    t_render = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for i, (pos, pts) in enumerate(cams):
            files.append(str(Path(tmp)/f'cam{i}.pcd')); write_pcd(files[-1], pts)
        cand = Path(tmp)/'cands.txt'
        np.savetxt(cand, np.column_stack([cat['center'], cat['approach'], cat['closing'], cat['width']]), fmt='%.9f')
        out = Path(tmp)/'scores.txt'
        g = Path(a.gpd_source)
        t1 = time.perf_counter()
        r = subprocess.run([a.adapter, *files, str(cand), str(out), str(g/'models/lenet/15channels/params') + '/',
                            str(g/'cfg/image_geometry_15channels.cfg'), *[f'{x:.6f}' for pos, _ in cams for x in pos]],
                           capture_output=True, text=True, timeout=3600, env={'OMP_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin'})
        t_gpd = time.perf_counter() - t1
        if r.returncode != 0:
            path.write_text(json.dumps(dict(error=r.stderr[-2000:], returncode=r.returncode)))
            return n, 'error'
        scores = np.loadtxt(out, ndmin=1)
    if len(scores) != len(cat['label']):
        path.write_text(json.dumps(dict(error=f'{len(scores)} scores for {len(cat["label"])} candidates')))
        return n, 'error'
    stages = {}
    for line in r.stderr.splitlines():
        if line.startswith('TIMINGS '):
            stages = {k: float(v) for k, v in (kv.split('=') for kv in line.split()[1:])}
    path.write_text(json.dumps(dict(scores=[float(x) for x in scores], render_seconds=round(t_render, 3), gpd_seconds=round(t_gpd, 3),
                                    gpd_stage_seconds=stages,
                                    cloud_points=[int(len(p)) for _, p in cams], stderr_tail=r.stderr[-300:])))
    return n, round(t_gpd, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    for k in ('adapter', 'gpd-source', 'names', 'catalogues', 'store', 'projects', 'out-dir'):
        s.add_argument(f'--{k}', required=True)
    s.add_argument('--jobs', type=int, default=1)
    a = p.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    names = [x.strip() for x in Path(a.names).read_text().splitlines() if x.strip()]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for n, st in pool.map(score_one, [(a, n) for n in names]):
            print(n, st, flush=True)


if __name__ == '__main__':
    main()
