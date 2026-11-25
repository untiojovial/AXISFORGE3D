"""Bounded archive validation and atomic installation of development features."""
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile
import numpy as np
from research_grasping.phase4.jobs import MANIFEST_HASH, RGB_LOCK, RGB_LOCK_HASH, canonical, sha_bytes
from PIL import Image

MAX_ARCHIVE = 512 * 1024 * 1024
MAX_MEMBER = 64 * 1024 * 1024
MAX_FILES = 4096
LOCK = Path(__file__).resolve().parents[2] / 'research/phase3/dataset/manifest.lock.json'


def safe_members(z):
    infos = z.infolist()
    if len(infos) > MAX_FILES or sum(x.file_size for x in infos) > MAX_ARCHIVE:
        raise ValueError('archive exceeds resource limits')
    names = set()
    for info in infos:
        name = info.filename; p = PurePosixPath(name)
        if (not name or '\\' in name or p.is_absolute() or '..' in p.parts or '.' in p.parts
                or name in names or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)
                or info.file_size > MAX_MEMBER or (info.file_size and not info.compress_size)
                or (info.compress_size and info.file_size / info.compress_size > 500)):
            raise ValueError('unsafe archive member')
        names.add(name)
    return names


def load_job(path):
    with zipfile.ZipFile(path) as z:
        names = safe_members(z)
        if 'job.json' not in names: raise ValueError('missing job')
        job = json.loads(z.read('job.json'))
        unsigned = dict(job); jid = unsigned.pop('job_id', None)
        if jid != sha_bytes(canonical(unsigned)) or job.get('dataset_manifest_sha256') != MANIFEST_HASH:
            raise ValueError('job hash or manifest mismatch')
        lock = json.loads(LOCK.read_text())
        if sha_bytes(LOCK.read_bytes()) != MANIFEST_HASH: raise ValueError('sealed lock mismatch')
        allowed = {r['object_id']: r['split'] for r in lock['objects'] if r['split'] in ('train', 'validation')}
        if sha_bytes(RGB_LOCK.read_bytes()) != RGB_LOCK_HASH: raise ValueError('RGB seal mismatch')
        rgb_lock = json.loads(RGB_LOCK.read_text())['views']
        ids = set(); expected_names = {'job.json'}
        for obj in job['objects']:
            oid = obj['object_id']
            if oid in ids or allowed.get(oid) != obj['split']: raise ValueError('invalid object or split')
            ids.add(oid)
            views = obj['views']
            if [v['view_id'] for v in views] != list(range(16)): raise ValueError('invalid view order')
            for i, view in enumerate(views):
                name = f'images/{oid}/{i:02d}.png'
                if view['path'] != name or name not in names or sha_bytes(z.read(name)) != view['sha256']:
                    raise ValueError('view mismatch')
                with Image.open(io.BytesIO(z.read(name))) as image:
                    if image.format != 'PNG' or image.mode != 'RGB' or image.size != (96, 96): raise ValueError('invalid image')
                    pixels = np.asarray(image)
                if sha_bytes(np.ascontiguousarray(pixels).tobytes()) != rgb_lock[oid][i]: raise ValueError('RGB view seal mismatch')
                expected_names.add(name)
        if not ids or names != expected_names: raise ValueError('unexpected job members')
    return job


def validate_npz(data, view_count=16, token_geometry=None):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = safe_members(z)
        if not {'global_embeddings.npy', 'view_ids.npy'} <= names or not names <= {'global_embeddings.npy', 'view_ids.npy', 'patch_embeddings.npy'}:
            raise ValueError('invalid NPZ members')
        for info in z.infolist():
            with z.open(info) as member:
                version = np.lib.format.read_magic(member)

                if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(member, max_header_size=10000)
                elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(member, max_header_size=10000)
                else: raise ValueError('unsupported NPY version')
                if dtype.hasobject or len(shape) > 3 or any(type(n) is not int or n < 0 for n in shape): raise ValueError('unsafe array header')
                size = int(dtype.itemsize)
                for n in shape:
                    size *= n
                    if size > MAX_MEMBER: raise ValueError('array too large')
                if size > info.file_size: raise ValueError('truncated array')
    with np.load(io.BytesIO(data), allow_pickle=False) as a:
        v, ids = a['global_embeddings'], a['view_ids']
        if v.ndim != 2 or v.shape[0] != view_count or not 0 < v.shape[1] <= 8192 or v.dtype.kind != 'f' or not np.isfinite(v).all():
            raise ValueError('invalid global embeddings')
        if ids.shape != (view_count,) or ids.dtype.kind not in 'iu' or not np.array_equal(ids, np.arange(view_count)):
            raise ValueError('invalid view IDs')
        if 'patch_embeddings' in a:
            p = a['patch_embeddings']
            if p.ndim != 3 or p.shape[0] != view_count or not (0 < p.shape[1] <= 4096 and 0 < p.shape[2] <= 8192) or p.dtype.kind != 'f' or not np.isfinite(p).all():
                raise ValueError('invalid patch embeddings')
            if token_geometry is not None:
                grid = token_geometry.get('grid_shape')
                if not isinstance(grid, list) or len(grid) != 2 or any(not isinstance(x, int) or x <= 0 for x in grid) or grid[0] * grid[1] != p.shape[1]:
                    raise ValueError('patch grid mismatch')


def verify_bundle(bundle, job_path):
    job = load_job(job_path)
    with zipfile.ZipFile(bundle) as z:
        names = safe_members(z)
        if 'index.json' not in names: raise ValueError('missing index')
        index = json.loads(z.read('index.json'))
        if index.get('schema_version') != 1 or index.get('job_id') != job['job_id'] or index.get('dataset_manifest_sha256') != MANIFEST_HASH:
            raise ValueError('bundle/job mismatch')
        for k in ('model_id', 'model_revision', 'checkpoint_sha256', 'processor_sha256', 'token_geometry'):
            if not index.get(k): raise ValueError('missing model provenance: ' + k)
        config = index.get('config', {})
        for key in ('job_id', 'model_id', 'model_revision', 'checkpoint_sha256', 'processor_sha256'):
            if config.get(key) != index.get(key): raise ValueError('inconsistent provenance: ' + key)
        if sha_bytes(canonical(config)) != index.get('run_id'): raise ValueError('run configuration hash mismatch')
        for key in ('checkpoint_sha256', 'processor_sha256'):
            value = index[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value): raise ValueError('invalid provenance hash')
        expected = {o['object_id']: o['split'] for o in job['objects']}; seen = set(); files = {'index.json'}
        for row in index['objects']:
            oid = row['object_id']; name = f'features/{oid}.npz'
            if oid in seen or expected.get(oid) != row['split'] or row['file'] != name or name not in names:
                raise ValueError('invalid bundle object')
            seen.add(oid); files.add(name); blob = z.read(name)
            if sha_bytes(blob) != row['sha256']: raise ValueError('feature hash mismatch')
            validate_npz(blob, token_geometry=index['token_geometry'])
        if seen != set(expected) or files != names: raise ValueError('incomplete or unexpected bundle files')
    return index


def install_bundle(bundle, job, destination):
    index = verify_bundle(bundle, job)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        # Reuse only the exact verified installation, including the file set.
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError('installed bundle path is not a directory')
        expected = {'index.json'} | {row['file'] for row in index['objects']}
        observed = set()
        for path in destination.rglob('*'):
            if path.is_symlink(): raise ValueError('symlink in installed bundle')
            if path.is_file(): observed.add(path.relative_to(destination).as_posix())
            elif not path.is_dir(): raise ValueError('unexpected installed bundle entry')
        if observed != expected or (destination / 'index.json').read_bytes() != canonical(index):
            raise ValueError('existing installation does not match bundle')
        for row in index['objects']:
            if sha_bytes((destination / row['file']).read_bytes()) != row['sha256']:
                raise ValueError('existing installation is corrupt')
        return index
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.bundle-', dir=destination.parent))
    try:
        with zipfile.ZipFile(bundle) as z:
            for row in index['objects']:
                path = temp / row['file']; path.parent.mkdir(exist_ok=True)
                path.write_bytes(z.read(row['file']))
            (temp / 'index.json').write_bytes(canonical(index))
        os.rename(temp, destination)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    return index
