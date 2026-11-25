"""Validate development visual arrays and source texture against a post-hoc lock."""
import hashlib
import json
from pathlib import Path
import numpy as np

LOCK_PATH = Path(__file__).resolve().parents[2] / 'research/phase3/vision/development_assets.lock.json'
LOCK_SHA256 = '409cb0f61bb9a3069c634a68e5e2d87bce0af6a21ae21a735ed0c7df4ff0df81'


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def semantic_npz_sha256(path):
    """Hash all named arrays, dtype, shape, bytes except runtime measurement."""
    h=hashlib.sha256()
    with np.load(path,allow_pickle=False) as data:
        for key in sorted(data.files):
            if key == 'elapsed_seconds': continue
            arr=np.ascontiguousarray(data[key])
            if arr.dtype.hasobject: raise ValueError('object arrays forbidden')
            header=json.dumps([key,arr.dtype.str,arr.shape],separators=(',',':')).encode()
            h.update(len(header).to_bytes(8,'big')); h.update(header)
            payload=arr.tobytes(); h.update(len(payload).to_bytes(8,'big')); h.update(payload)
    return h.hexdigest()


def read_lock():
    if file_sha256(LOCK_PATH) != LOCK_SHA256: raise ValueError('vision lock hash mismatch')
    return json.loads(LOCK_PATH.read_text())


def verify_visual(path, object_id, lock=None):
    lock=read_lock() if lock is None else lock
    row=lock['objects'].get(object_id)
    if not row or semantic_npz_sha256(path)!=row['vision_semantic_sha256']:
        raise ValueError('vision array lock mismatch')


def verify_texture(path, object_id, lock=None):
    lock=read_lock() if lock is None else lock
    row=lock['objects'].get(object_id)
    if not row or file_sha256(path)!=row['source_texture_sha256']:
        raise ValueError('texture lock mismatch')
