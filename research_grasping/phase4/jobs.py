"""Prepare development-only, provenance-bound portable image jobs."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile
import numpy as np
from PIL import Image

RGB_LOCK = Path(__file__).resolve().parents[2] / 'research/phase4/automation/rgb_views.lock.json'
RGB_LOCK_HASH = '810daedd0f3dfb1335d3e1ad7b5ec5dcdc4a2a2733ffb8ba09beabdb1cb35de7'
MANIFEST_HASH = '6c5291288adcd365b873a4c983b30b9f19fb2ed0110a0575112514b866cdf660'


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def prepare(root, observations, output, *, objects=None):
    root, observations, output = Path(root), Path(observations), Path(output)
    source = root / 'manifest.json'
    if sha_file(source) != MANIFEST_HASH:
        raise ValueError('dataset manifest does not match sealed manifest')
    manifest = json.loads(source.read_text())
    allowed = {r['object_id']: r for r in manifest['objects'] if r['split'] in ('train', 'validation')}
    selected = sorted(allowed) if objects is None else sorted(objects)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
        raise ValueError('objects must be unique, development-only IDs')
    from research_grasping.phase3.vision_integrity import read_lock, verify_visual
    vision_lock = read_lock()
    rows, images = [], {}
    for oid in selected:
        if not re.fullmatch(r'[A-Za-z0-9_]+', oid):
            raise ValueError('unsafe object ID')
        p = observations / (oid + '.npz')
        verify_visual(p, oid, vision_lock)
        with np.load(p, allow_pickle=False) as data:
            if str(data['manifest_sha256']) != MANIFEST_HASH or str(data['mesh_sha256']) != allowed[oid]['processed_sha256']:
                raise ValueError('observation provenance mismatch')
            rgb = data['rgb']
            if rgb.dtype != np.uint8 or rgb.shape != (16, 96, 96, 3):
                raise ValueError('expected 16 RGB views at 96x96')
            row = {'object_id': oid, 'split': allowed[oid]['split'], 'observation_sha256': sha_file(p), 'views': []}
            for i, frame in enumerate(rgb):
                stream = io.BytesIO(); Image.fromarray(frame).save(stream, format='PNG')
                name = f'images/{oid}/{i:02d}.png'; blob = stream.getvalue(); images[name] = blob
                row['views'].append({'view_id': i, 'path': name, 'sha256': sha_bytes(blob)})
            rows.append(row)
    spec = {'schema_version': 1, 'dataset_manifest_sha256': MANIFEST_HASH, 'objects': rows}
    spec['job_id'] = sha_bytes(canonical(spec))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=output.parent, prefix='.job-', suffix='.zip'); os.close(fd)
    try:
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('job.json', canonical(spec))
            for name, blob in images.items(): z.writestr(name, blob)
        os.replace(temp, output)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return spec
