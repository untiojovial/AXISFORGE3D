"""Build the local research package `axisforge3d-research-reproducibility` (no upload, no release, no DOI).

    python tools/package/build_package.py --out DIR [--commit HEAD]

The package is a snapshot of the repository's tracked files at --commit. Author decisions of 2026-11-26 are
applied:

- **DINOv3.** No DINOv3 embedding or value derived from DINOv3 embeddings is included. Mixed files are cleaned:
  the `dinov3*` arrays are dropped from `.npz` files and the `dinov3*` numeric fields from JSON descriptor
  records. The DINOv3 visualisation and per-object analysis are excluded. Metadata (model id, revision, hashes)
  and aggregate statistics are kept.
- **YCB.** No per-object YCB descriptor, geometry (bounds) or grasp pose is included. Their rows or records are
  removed from mixed files. Files that cannot be separated are excluded. Aggregate statistics, tables and object
  identifiers stay. `tools/verify/ycb_pick.sh`, pinned to pybullet-object-models@57cc400, regenerates the
  objects.
- **Manuscript.** `publication/` is excluded. The manuscript is intended for subscription publication and is
  not licensed under Apache-2.0.
- **Build artefacts.** `*.egg-info/` is excluded.

The script writes:

- `MANIFEST_INCLUDED.json`: path, SHA-256, licence, and whether the file was cleaned (with the original's
  SHA-256);
- `MANIFEST_EXCLUDED.json`: path, original SHA-256, reason, and the regeneration route;
- `PACKAGE_README.md`.

The historical files in the repository and in its history are not modified. Excluding a file here does not
remove its earlier public copies.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
YCB = ('Banana', 'ChipsCan', 'CrackerBox', 'FoamBrick', 'GelatinBox', 'Hammer', 'MasterChefCan', 'MediumClamp', 'MustardBottle',
       'Pear', 'PottedMeatCan', 'PowerDrill', 'Scissors', 'Strawberry', 'TennisBall', 'TomatoSoupCan')
KEEP_DINOV3_META = {'README.md', 'summary.json', 'reproduction.json', 'bundle_index.json'}
REGEN = {
    'dinov3': 'tools/research/phase4/extract_frozen_embeddings.py with the user\'s own licensed DINOv3 weights '
              '(timm/vit_small_patch16_dinov3.lvd1689m, revision and SHA-256 in the provenance of each descriptor file)',
    'ycb': 'tools/verify/ycb_pick.sh (pybullet-object-models@57cc4007607f6bcbb0a5364ccab7636faf7a91e4), then the Phase IV tools '
           'named in research/phase4/grasp_prior/README.md',
    'manuscript': 'not redistributed in the package (rights reserved for subscription publication); the repository keeps the sources',
}


def git(*a, binary=False):
    r = subprocess.run(['git', *a], cwd=ROOT, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def is_ycb(name):
    n = str(name)
    return any(n == y or n.startswith(y + '|') or n.endswith('/' + y) or n == f'Ycb{y}' for y in YCB)


def licence_of(path):
    if path.startswith(('apps/viewer/samples/ur5/', 'crates/project-io/tests/data/ur5/')):
        return 'BSD-3-Clause (example-robot-data; licence file alongside)'
    if path == 'research/phase4/embedding_experiment/development-images.zip':
        return 'CC-BY-4.0 (renders of Google Scanned Objects; attribution in THIRD_PARTY_NOTICES.md)'
    if path.startswith('LICENSES/') or path == 'LICENSE':
        return 'licence text (verbatim)'
    return 'Apache-2.0'


def clean_npz(data):
    z = np.load(io.BytesIO(data), allow_pickle=True)
    keys = list(z.files)
    out, changed, notes = {}, False, []
    groups = {}
    for k in keys:
        base = k[:-len('__names')] if k.endswith('__names') else k
        groups.setdefault(base, []).append(k)
    if 'names' in keys:  # one names array for the whole file
        names = [str(x) for x in z['names']]
        keep = np.array([not is_ycb(n) for n in names])
        for k in keys:
            arr = z[k]
            if k in ('names', 'vectors') and arr.shape[:1] == (len(names),):
                out[k] = arr[keep]
            else:
                out[k] = arr
        if not keep.all():
            changed = True; notes.append(f'removed {int((~keep).sum())} YCB rows')
    else:
        for base, ks in groups.items():
            if base.startswith('dinov3'):
                changed = True; notes.append(f'dropped {base}')
                continue
            nk = f'{base}__names'
            if nk in z.files:
                names = [str(x) for x in z[nk]]
                keep = np.array([not is_ycb(n) for n in names])
                out[nk] = z[nk][keep]
                out[base] = z[base][keep] if z[base].shape[:1] == (len(names),) else z[base]
                if not keep.all():
                    changed = True; notes.append(f'{base}: removed {int((~keep).sum())} YCB rows')
            else:
                for k in ks:
                    out[k] = z[k]
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    return buf.getvalue(), changed, notes


def strip_json(obj, notes):
    """Drop dinov3* fields and YCB-named records/keys, recursively."""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith('dinov3') and isinstance(v, (list, dict)):
                notes.add(f'dropped field {k}'); continue
            if isinstance(k, str) and is_ycb(k):
                notes.add('dropped YCB-keyed entries'); continue
            if isinstance(v, dict) and any(is_ycb(v.get(f, '')) for f in ('name', 'part', 'body', 'object')):
                notes.add('dropped YCB records'); continue
            new[k] = strip_json(v, notes)
        return new
    if isinstance(obj, list):
        out = []
        for v in obj:
            if isinstance(v, dict) and any(is_ycb(v.get(f, '')) for f in ('name', 'part', 'body', 'object')):
                notes.add('dropped YCB records'); continue
            out.append(strip_json(v, notes))
        return out
    return obj


GEOM = re.compile(r'"(low|high|center|approach|closing|descriptors|vectors)"')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True); p.add_argument('--commit', default='HEAD')
    a = p.parse_args()
    commit = git('rev-parse', a.commit).strip()
    out = Path(a.out)/'axisforge3d-research-reproducibility'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = git('ls-tree', '-r', '--name-only', commit).split('\n')
    inc, exc = [], []
    for f in filter(None, files):
        data = git('show', f'{commit}:{f}', binary=True)
        h0 = sha(data)
        base = Path(f).name
        if f.startswith('publication/'):
            exc.append(dict(path=f, sha256=h0, reason='manuscript and submission material: rights reserved for subscription publication',
                            regenerate=REGEN['manuscript'])); continue
        if '.egg-info/' in f:
            exc.append(dict(path=f, sha256=h0, reason='build artefact', regenerate='pip install -e .')); continue
        if ('/dinov3' in f or base.startswith('dinov3')) and base not in KEEP_DINOV3_META:
            exc.append(dict(path=f, sha256=h0, reason='DINOv3 embeddings or values derived from them (author decision: not in the package)',
                            regenerate=REGEN['dinov3'])); continue
        cleaned, notes = data, []
        if f.endswith('.npz'):
            try:
                cleaned, changed, notes = clean_npz(data)
                if not changed:
                    cleaned = data
            except Exception as e:  # noqa: BLE001
                exc.append(dict(path=f, sha256=h0, reason=f'could not be inspected for DINOv3/YCB content ({type(e).__name__})',
                                regenerate='see the producing script named in the phase README')); continue
        elif f.endswith(('.json', '.json.gz')):
            text = gzip.decompress(data).decode() if f.endswith('.gz') else data.decode()
            has_d3 = re.search(r'"dinov3[^"]*"\s*:\s*[\[{]', text) is not None
            has_ycb = any(f'"{y}"' in text for y in YCB) and GEOM.search(text) is not None
            if has_d3 or has_ycb:
                ns = set()
                obj = strip_json(json.loads(text), ns)
                t2 = json.dumps(obj, indent=None if f.endswith('.gz') else 1)
                still = (re.search(r'"dinov3[^"]*"\s*:\s*\[\s*-?\d', t2) is not None) or \
                        (any(f'"{y}"' in t2 for y in YCB) and GEOM.search(t2) is not None and has_ycb)
                if still:
                    exc.append(dict(path=f, sha256=h0, reason='DINOv3 or YCB per-object data could not be separated automatically',
                                    regenerate=REGEN['ycb'] if has_ycb else REGEN['dinov3'])); continue
                cleaned = gzip.compress(t2.encode(), mtime=0) if f.endswith('.gz') else (t2 + '\n').encode()
                notes = sorted(ns)
        dest = out/f
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(cleaned)
        rec = dict(path=f, sha256=sha(cleaned), licence=licence_of(f))
        if cleaned is not data:
            rec.update(cleaned=True, original_sha256=h0, changes=notes,
                       regenerate_removed_parts=REGEN['ycb'] if any('YCB' in n for n in notes) else REGEN['dinov3'])
        inc.append(rec)
    meta = dict(package='axisforge3d-research-reproducibility', source_commit=commit, files_included=len(inc), files_excluded=len(exc),
                rules=__doc__.strip().split('\n\n', 2)[1].replace('\n', ' '))
    (out/'MANIFEST_INCLUDED.json').write_text(json.dumps(dict(meta, files=inc), indent=1) + '\n')
    (out/'MANIFEST_EXCLUDED.json').write_text(json.dumps(dict(meta, files=exc), indent=1) + '\n')
    shutil.copy(ROOT/'research/phase7_extension/package/PACKAGE_README.md', out/'PACKAGE_README.md')
    print(json.dumps(dict(meta, cleaned=sum(1 for r in inc if r.get('cleaned'))), indent=1))


if __name__ == '__main__':
    main()
