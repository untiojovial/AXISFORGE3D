"""Build a package of new parts from Google Scanned Objects for the grasp prior.

The selection looks only at names and bounding boxes, never at grasp results:

1. The catalogue is the list of models at the pinned MuJoCo adaptation
   (the revision Phase III uses), read from a blob-less clone:
       git init gso && cd gso && git remote add origin https://github.com/kevinzakka/mujoco_scanned_objects.git
       git fetch --depth 1 --filter=blob:none origin REV && git ls-tree --name-only REV models/ > catalogue.txt
2. Every object of the Phase III manifest (train, validation and the sealed
   test split) is removed, so nothing here has been seen and the test split
   stays untouched.
3. The rest is shuffled with a fixed seed and walked in order. Each model.obj
   is downloaded and kept when its box fits the demo gripper somewhere and
   the arm can reach it: smallest side 5-70 mm, largest at most 250 mm,
   height at most 200 mm.
4. A kept part is placed alone on the demo table, as the YCB parts are
   (`add-object --units m --mass 0.1 --at 0.42 0 0.42`). A part the CLI
   refuses is recorded and skipped.

Downloads go into the output folder, which should be new and empty; they are
only read as mesh data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

import numpy as np

REV = '6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c'
BASE = f'https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/{REV}/models/'
RULE = dict(min_side_m=[.005, .070], max_side_m=.25, max_height_m=.20)
PLACEMENT = ['0.42', '0', '0.42']
MASS_KG = '0.1'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, attempts=4):
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == attempts-1:
                raise
            time.sleep(2**attempt)


def obj_extent(data):
    """Bounding-box extents (x, y, z) of an OBJ's vertices, metres."""
    rows = [line.split()[1:4] for line in data.decode('utf-8', 'replace').splitlines() if line.startswith('v ')]
    v = np.asarray(rows, float)
    if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
        raise ValueError('no usable vertices')
    return v.max(0)-v.min(0)


def accept(extent):
    low, high = RULE['min_side_m']
    if not low <= extent.min() <= high:
        return f'smallest side {extent.min()*1000:.0f} mm outside 5-70 mm'
    if extent.max() > RULE['max_side_m']:
        return f'largest side {extent.max()*1000:.0f} mm over 250 mm'
    if extent[2] > RULE['max_height_m']:
        return f'height {extent[2]*1000:.0f} mm over 200 mm'
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalogue', required=True, help='models/<name> lines at the pinned revision')
    p.add_argument('--exclude', required=True, help='the Phase III manifest.lock.json')
    p.add_argument('--also-exclude', action='append', default=[],
                   help='a previous new_parts.json whose kept parts are left out too (repeatable)')
    p.add_argument('--binary', required=True); p.add_argument('--out', required=True)
    p.add_argument('--count', type=int, default=50); p.add_argument('--seed', type=int, default=20261126)
    a = p.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    binary = str(Path(a.binary).resolve())
    catalogue_bytes = Path(a.catalogue).read_bytes()
    names = sorted({line.strip().removeprefix('models/') for line in catalogue_bytes.decode().splitlines() if line.strip()})
    manifest = json.loads(Path(a.exclude).read_text())
    excluded = {o['object_id'] for o in manifest['objects']}
    if manifest.get('source_revision', REV) != REV or not excluded <= set(names):
        raise SystemExit('the exclusion manifest does not match the pinned catalogue')
    previous = []
    for path in a.also_exclude:
        earlier = json.loads(Path(path).read_text())
        if earlier.get('source_revision') != REV or not set(earlier['kept']) <= set(names):
            raise SystemExit(f'{path} is not a selection from the pinned catalogue')
        previous.append(dict(sha256=sha256(Path(path).read_bytes()), seed=earlier['seed'], kept=len(earlier['kept'])))
        excluded |= set(earlier['kept'])
    pool = [n for n in names if n not in excluded]
    order = [pool[i] for i in np.random.default_rng(a.seed).permutation(len(pool))]
    empty = out/'empty.axisforge'
    subprocess.run([binary, 'demo', '--no-parts', '--out', str(empty)], check=True, capture_output=True)
    examined, kept = [], []
    for name in order:
        if len(kept) == a.count:
            break
        if '/' in name or name.startswith('.'):
            continue
        data = fetch(BASE+name+'/model.obj')
        row = dict(name=name, obj_sha256=sha256(data), obj_bytes=len(data))
        try:
            extent = obj_extent(data)
            row['extent_m'] = [round(float(x), 5) for x in extent]
            reason = accept(extent)
        except ValueError as error:
            reason = str(error)
        if reason is None:
            folder = out/name; folder.mkdir(exist_ok=True)
            (folder/'model.obj').write_bytes(data)
            cell = folder/'cell.axisforge'
            done = subprocess.run([binary, 'add-object', str(empty), str(folder/'model.obj'), '--name', name,
                                   '--units', 'm', '--mass', MASS_KG, '--at', *PLACEMENT, '--out', str(cell)],
                                  capture_output=True, text=True, timeout=600)
            if done.returncode == 0:
                row.update(kept=True, project=f'{name}/cell.axisforge', project_sha256=sha256(cell.read_bytes()))
                kept.append(name)
            else:
                reason = 'add-object refused: ' + (done.stderr.strip().splitlines() or ['?'])[-1]
        if reason is not None:
            row.update(kept=False, reason=reason)
        examined.append(row)
        print(f'{len(kept):2d}/{a.count}', name, 'kept' if row['kept'] else row['reason'], flush=True)
    record = dict(format='axisforge-new-parts', version=1, source='Google Scanned Objects, MuJoCo adaptation by '
                  'Kevin Zakka', source_revision=REV, license='CC-BY-4.0', catalogue_sha256=sha256(catalogue_bytes),
                  catalogue_models=len(names), excluded=sorted(excluded), excluded_manifest_sha256=sha256(
                      Path(a.exclude).read_bytes()), also_excluded=previous, seed=a.seed, count=a.count, rule=RULE,
                  placement_m=[float(x) for x in PLACEMENT], mass_kg=float(MASS_KG), kept=kept, examined=examined)
    (out/'new_parts.json').write_text(json.dumps(record, indent=1) + '\n')
    print(f'{len(kept)} parts kept of {len(examined)} examined -> {out/"new_parts.json"}')


if __name__ == '__main__':
    main()
