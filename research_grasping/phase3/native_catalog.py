"""Finite, unsorted AxisForge3D candidate catalogue and on-demand native checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _invoke(binary, project, body, candidate_id, timeout_s):
    if timeout_s <= 0:
        raise ValueError('timeout_s must be positive')
    args = [str(Path(binary).resolve()), 'research-candidates', str(Path(project).resolve()), str(body)]
    if candidate_id is not None:
        if not isinstance(candidate_id, int) or isinstance(candidate_id, bool) or candidate_id < 0:
            raise ValueError('candidate_id must be a nonnegative integer')
        args += ['--evaluate', str(candidate_id)]
    start = time.perf_counter()
    completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=True)
    result = json.loads(completed.stdout)
    if not isinstance(result, list):
        raise ValueError('unexpected AxisForge3D response')
    return {'records': result, 'elapsed_s': time.perf_counter() - start,
            'project_sha256': _sha256(project), 'binary_sha256': _sha256(binary),
            'body': body, 'command': args}


def load_catalog(binary, project, body, timeout_s=120.0):
    """Generate unsorted candidate poses before F1/F2; no labels are inferred."""
    return _invoke(binary, project, body, None, timeout_s)


def evaluate_candidate(binary, project, body, candidate_id, timeout_s=120.0):
    """Run native F1 and F2 for exactly the named original candidate."""
    envelope = _invoke(binary, project, body, candidate_id, timeout_s)
    if len(envelope['records']) != 1 or envelope['records'][0]['id'] != candidate_id:
        raise ValueError('candidate identity mismatch')
    return envelope


def evaluate_batch(binary, project, body, candidate_ids, timeout_s=120.0):
    """Evaluate explicitly selected IDs, retaining infrastructure errors per ID."""
    results = []
    for candidate_id in candidate_ids:
        try:
            results.append({'id': candidate_id, 'status': 'ok', 'result': evaluate_candidate(
                binary, project, body, candidate_id, timeout_s)})
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            results.append({'id': candidate_id, 'status': 'infrastructure_error',
                            'error': f'{type(exc).__name__}: {exc}'})
    return results
