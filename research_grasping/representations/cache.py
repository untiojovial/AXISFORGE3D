"""Content addressed array cache; no pickle deserialization."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def cache_key(metadata, arrays):
    h = hashlib.sha256(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        if a.dtype.hasobject:
            raise ValueError("object arrays are not supported")
        h.update(
            json.dumps([name, a.dtype.str, a.shape], separators=(",", ":")).encode()
        )
        h.update(a.tobytes())
    return h.hexdigest()


class FeatureCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def put(self, key, features):
        self._check(key)
        arrays = {name: np.asarray(value) for name, value in features.items()}
        if any(a.dtype.hasobject for a in arrays.values()):
            raise ValueError("object arrays are not supported")
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, self.directory / (key + ".npz"))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def get(self, key):
        self._check(key)
        path = self.directory / (key + ".npz")
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    @staticmethod
    def _check(key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid cache key")
