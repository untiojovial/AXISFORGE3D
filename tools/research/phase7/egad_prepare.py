"""Prepare EGAD objects as AxisForge cells. Data preparation only: no planning, physics or models.

    python -I tools/research/phase7/egad_prepare.py select --egad DIR --out selection.json --seed 20261202 --count N
    python -I tools/research/phase7/egad_prepare.py build --egad DIR --selection selection.json \
        --binary target/release/axisforge --out POOL --jobs 4 [--positions START STOP] [--evaluation-set]

Rules (fixed in PHASE7_PROTOCOL.md, part II, before any object is planned):

1. Pool: the EGAD training set (2 281 meshes, BSD-3, Morrison et al. 2020). Meshes are named by grid
   cell (difficulty letter, complexity number) with up to four meshes per cell; one mesh is drawn per
   cell (seeded), so near-identical siblings never enter together. Cells are then drawn at random
   (seeded) up to --count. The 49-object evaluation set is prepared separately as a secondary set.
2. Mesh checks: loads as one triangle mesh; watertight after merging duplicate vertices; positive
   volume. Failures are recorded and replaced by the next drawn cell; never by planning outcome.
3. Scale: the dataset authors' published rule (scripts/prepare_meshes.py) for an 80 mm gripper:
   largest oriented-box side = 80 mm; if the smallest side then exceeds 0.8 x 80 mm, shrink it to that.
   The objects are therefore 3D-printable shapes at gripper scale, not real products.
4. Pose: resting on the most probable stable face (trimesh.poses.compute_stable_poses).
5. Cell: the demo cell without parts, the part placed as every Phase IV/V part
   (add-object --units m --mass 0.1 --at 0.42 0 0.42). The mass is assumed.
6. Collision pieces for MuJoCo: CoACD (threshold 0.1, at most 32 pieces, no preprocessing, 60 tree-search
   iterations with 10 nodes, seed 0), written as model_collision_*.obj. Chosen on excluded cells only (COACD_KW).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

WIDTH = .08
FRAC = .8
PLACEMENT = ['0.42', '0', '0.42']
MASS = '0.1'
CELL = re.compile(r'^([A-Z])(\d+)_(\d+)$')
# Chosen on excluded smoke-test cells (order positions 500-502) before part II: threshold 0.05 without a cap gave
# 100-150 pieces in 200-400 s per object; 32 is the cap of the GSO collision models used in development (median and
# maximum 32 pieces); preprocessing is off because every selected mesh passed the watertight check; the lighter
# tree search (60 iterations, 10 nodes) halves the time (J20: 66 s, hull volume / mesh volume 1.23).
COACD_KW = dict(threshold=.1, max_convex_hull=32, preprocess_mode='off', mcts_iterations=60, mcts_nodes=10, seed=0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    import trimesh
    m = trimesh.load(path, force='mesh', process=True)
    m.merge_vertices()
    return m


def scale_like_egad(m):
    exts = m.bounding_box_oriented.primitive.extents
    s = WIDTH/max(exts)
    m.apply_scale(s)
    exts = m.bounding_box_oriented.primitive.extents
    if min(exts) > WIDTH*FRAC:
        m.apply_scale(WIDTH*FRAC/min(exts))
    return m


def check(m):
    if len(m.faces) == 0:
        return 'no faces'
    if not m.is_watertight:
        return 'not watertight'
    if not m.volume > 0:
        return 'non-positive volume'
    return None


def select(a):
    files = sorted(Path(a.egad).glob('**/egad_train_set/*.obj'))
    cells = {}
    for f in files:
        mt = CELL.match(f.stem)
        if mt:
            cells.setdefault(mt.group(1)+mt.group(2), []).append(f)
    rng = np.random.default_rng(a.seed)
    keys = sorted(cells)
    order = rng.permutation(len(keys))
    chosen, rejected = [], []
    for i in order:
        if len(chosen) >= a.count:
            break
        key = keys[i]
        f = sorted(cells[key])[int(rng.integers(len(cells[key])))]
        reason = check(load(f))
        (rejected if reason else chosen).append(dict(cell=key, file=f.name, sha256=sha256(f), reason=reason))
    evalset = sorted(Path(a.egad).glob('**/egad_eval_set/*.obj'))
    ev, ev_rej = [], []
    for f in evalset:
        reason = check(load(f))
        (ev_rej if reason else ev).append(dict(cell=f.stem, file=f.name, sha256=sha256(f), reason=reason))
    doc = dict(source='EGAD (Morrison, Corke, Leitner 2020), BSD-3; training set 2 281 meshes, evaluation set 49',
               archives='see research/phase7/PHASE7_DATA_MANIFEST.json', seed=a.seed, count=a.count,
               rule=__doc__.split('Rules')[1].split('"""')[0].strip(), cells_in_pool=len(keys),
               primary=chosen, primary_rejected=rejected, evaluation_set=ev, evaluation_rejected=ev_rej)
    Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
    print(len(chosen), 'chosen;', len(rejected), 'rejected;', len(ev), 'evaluation-set objects;', len(ev_rej), 'rejected')


def build_one(task):
    import trimesh
    src, name, out, binary, empty = task
    folder = Path(out)/name
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    m = scale_like_egad(load(src))
    transforms, probs = trimesh.poses.compute_stable_poses(m, n_samples=1, threshold=0.0)
    if len(transforms):
        m.apply_transform(transforms[int(np.argmax(probs))])
    m.apply_translation(-m.bounds[0]*[0, 0, 1])
    model = folder/'model.obj'
    m.export(model)
    cell = folder/'cell.axisforge'
    done = subprocess.run([binary, 'add-object', str(empty), str(model), '--name', name, '--units', 'm', '--mass', MASS,
                           '--at', *PLACEMENT, '--out', str(cell)], capture_output=True, text=True, timeout=900)
    row = dict(name=name, source=Path(src).name, extent_m=[round(float(x), 5) for x in m.extents],
               scale_rule='EGAD prepare_meshes.py, width 0.08 m', stable_pose_probability=float(max(probs)) if len(probs) else None)
    if done.returncode != 0:
        row.update(kept=False, reason='add-object refused: ' + (done.stderr.strip().splitlines() or ['?'])[-1])
        return row
    import coacd
    coacd.set_log_level('error')
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), **COACD_KW)
    coll = folder/'collision'
    coll.mkdir(exist_ok=True)
    for i, (v, f) in enumerate(parts):
        trimesh.Trimesh(v, f).export(coll/f'model_collision_{i}.obj')
    row.update(kept=True, project=f'{name}/cell.axisforge', project_sha256=sha256(cell), model_sha256=sha256(model), coacd=COACD_KW,
               collision_pieces=len(parts), seconds=round(time.perf_counter()-start, 1))
    return row


def build(a):
    sel = json.loads(Path(a.selection).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    empty = out/'empty.axisforge'
    subprocess.run([a.binary, 'demo', '--no-parts', '--out', str(empty)], check=True, capture_output=True)
    root = Path(a.egad)
    find = {p.name: p for p in root.glob('**/egad_*_set/*.obj')}
    lo, hi = a.positions
    tasks = [(find[r['file']], f"egad_{r['cell']}", str(out), a.binary, str(empty)) for r in sel['primary'][lo:hi]]
    if a.evaluation_set:
        tasks += [(find[r['file']], f"egad_eval_{r['cell']}", str(out), a.binary, str(empty)) for r in sel['evaluation_set']]
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for row in pool.map(build_one, tasks):
            rows.append(row)
            print(row['name'], row.get('kept'), row.get('collision_pieces'), row.get('seconds'), flush=True)
    (out/'built.json').write_text(json.dumps(dict(rows=rows, empty_sha256=sha256(empty), positions=[lo, hi],
                                                  evaluation_set=a.evaluation_set), indent=1) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('select'); s.add_argument('--egad', required=True); s.add_argument('--out', required=True)
    s.add_argument('--seed', type=int, default=20261202); s.add_argument('--count', type=int, required=True)
    b = sub.add_parser('build'); b.add_argument('--egad', required=True); b.add_argument('--selection', required=True)
    b.add_argument('--binary', required=True); b.add_argument('--out', required=True); b.add_argument('--jobs', type=int, default=4)
    b.add_argument('--positions', type=int, nargs=2, default=(0, None), metavar=('START', 'STOP'),
                   help='slice of the frozen primary order to build (python slice, STOP exclusive)')
    b.add_argument('--evaluation-set', action='store_true', help='also build the 45 evaluation-set objects')
    a = p.parse_args()
    select(a) if a.cmd == 'select' else build(a)


if __name__ == '__main__':
    main()
