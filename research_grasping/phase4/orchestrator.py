"""Checkpointed development extraction; no confirmatory execution."""
import io
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import zipfile
import numpy as np
from PIL import Image
from research_grasping.phase4.jobs import canonical, sha_bytes, sha_file, MANIFEST_HASH
from research_grasping.phase4.bundle_io import load_job, validate_npz, verify_bundle


def atomic(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.atomic-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _run_extraction(job_path, work, model, *, model_revision, device='cpu', batch_size=8):
    """Model must be a previously loaded, frozen adapter with provenance."""
    if not model_revision or batch_size < 1: raise ValueError('revision and positive batch size required')
    job = load_job(job_path); work = Path(work); work.mkdir(parents=True, exist_ok=True)
    prov = model.provenance
    checkpoint = prov.get('weights_sha256')
    processor_hash = prov.get('processor_sha256') or sha_bytes(prov['processor'].encode())
    implementation = Path(inspect.getfile(type(model)))
    try: torch_version = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError: torch_version = 'unavailable'
    config = {'adapter_sha256': sha_file(implementation), 'orchestrator_sha256': sha_file(__file__), 'torch_version': torch_version, 'numpy_version': np.__version__, 'schema_version': 1, 'job_id': job['job_id'], 'model_id': prov['model_id'],
              'model_revision': model_revision, 'checkpoint_sha256': checkpoint,
              'processor_sha256': processor_hash, 'model_provenance': prov,
              'batch_size': batch_size, 'device': device, 'precision': 'float32'}
    if not checkpoint: raise ValueError('missing checkpoint hash')
    run_id = sha_bytes(canonical(config)); state_path = work / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['run_id'] != run_id: raise ValueError('incompatible resume configuration')
    else:
        state = {'run_id': run_id, 'config': config, 'status': 'PENDING', 'completed': {}, 'failed': {}, 'hardware': platform.platform()}
    state['status'] = 'RUNNING'; atomic(state_path, canonical(state))
    try:
        with zipfile.ZipFile(job_path) as z:
            for obj in job['objects']:
                oid = obj['object_id']; out = work / 'features' / f'{oid}.npz'
                if oid in state['completed']:
                    entry = state['completed'][oid]
                    if not out.is_file() or sha_file(out) != entry['sha256']:
                        raise ValueError('corrupt cached result: ' + oid)
                    validate_npz(out.read_bytes(), token_geometry=entry['token_geometry']); continue
                start = time.perf_counter(); gs = []; ps = []; geometry = None
                frames = [np.asarray(Image.open(io.BytesIO(z.read(v['path']))).convert('RGB')) for v in obj['views']]
                pos = 0; current = batch_size
                while pos < len(frames):
                    try:
                        result = model.encode(frames[pos:pos + current])
                    except RuntimeError as exc:
                        if 'out of memory' not in str(exc).lower() or current == 1: raise
                        current = max(1, current // 2)
                        try:
                            import torch
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                        except ImportError: pass
                        continue
                    if geometry is not None and geometry != result['token_geometry']: raise ValueError('inconsistent token geometry')
                    geometry = result['token_geometry']; gs.append(result['global_embeddings']); ps.append(result['patch_embeddings']); pos += len(result['global_embeddings'])
                mem = io.BytesIO(); np.savez_compressed(mem, global_embeddings=np.concatenate(gs), patch_embeddings=np.concatenate(ps), view_ids=np.arange(16, dtype=np.int32))
                blob = mem.getvalue(); validate_npz(blob); atomic(out, blob)
                state['completed'][oid] = {'sha256': sha_bytes(blob), 'elapsed_seconds': time.perf_counter() - start, 'token_geometry': geometry}
                atomic(state_path, canonical(state))
        state['status'] = 'COMPLETED'; state['failed'] = {}
    except Exception as exc:
        state['status'] = 'PARTIALLY_COMPLETED' if state['completed'] else 'FAILED'
        state['failed']['last'] = type(exc).__name__
        atomic(state_path, canonical(state)); raise
    atomic(state_path, canonical(state)); return state


def run_extraction(job_path, work, model, *, model_revision, device='cpu', batch_size=8):
    """Serialize writers; an interrupted process releases its operating-system lock."""
    import fcntl
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    with (work / '.writer.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another extraction owns this work directory') from exc
        try:
            return _run_extraction(job_path, work, model, model_revision=model_revision, device=device, batch_size=batch_size)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def export_bundle(job_path, work, output):
    job = load_job(job_path); work = Path(work); state = json.loads((work/'state.json').read_text())
    if state['status'] != 'COMPLETED' or state['config']['job_id'] != job['job_id']:
        raise ValueError('job incomplete or incompatible')
    rows = []; geom = None
    for obj in job['objects']:
        oid = obj['object_id']; entry = state['completed'][oid]; path = work/'features'/f'{oid}.npz'
        if not path.is_file() or sha_file(path) != entry['sha256']: raise ValueError('corrupt cache')
        validate_npz(path.read_bytes())
        if geom is not None and geom != entry['token_geometry']: raise ValueError('inconsistent geometry')
        geom = entry['token_geometry']
        rows.append({'object_id': oid, 'split': obj['split'], 'file': f'features/{oid}.npz', 'sha256': entry['sha256'], 'elapsed_seconds': entry['elapsed_seconds']})
    c = state['config']
    index = {'schema_version':1, 'job_id':job['job_id'], 'dataset_manifest_sha256':MANIFEST_HASH, 'model_id':c['model_id'], 'model_revision':c['model_revision'], 'checkpoint_sha256':c['checkpoint_sha256'], 'processor_sha256':c['processor_sha256'], 'model_provenance':c['model_provenance'], 'token_geometry':geom, 'run_id':state['run_id'], 'hardware':state['hardware'], 'config':c, 'objects':rows}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.bundle-', suffix='.zip', dir=output.parent); os.close(fd)
    try:
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_STORED) as z:
            z.writestr('index.json', canonical(index))
            for row in rows: z.write(work / row['file'], row['file'])
        verify_bundle(tmp, job_path); os.replace(tmp, output)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return index

# The registered DAG states which operations have actual executable integration.
# Pending nodes are intentionally blocked; listing a node does not run it.
DAG = {
    'prepare_job': (),
    'extract_features': ('prepare_job',),
    'verify_bundle': ('extract_features',),
    'development_bo': ('verify_bundle',),
    'candidate_generation': (),
    'historical_evaluation': ('candidate_generation',),
    'physical_validation': ('historical_evaluation',),
    'statistics': ('development_bo',),
    'figures': ('statistics',),
    'audit': ('figures',),
}


def validate_declarative(config):
    """Reject unsupported settings instead of silently changing experiment semantics."""
    required = {'schema_version','mode','model','model_revision','split','device','batch_size','views','precision','frozen','resume','confirmatory'}
    if set(config) != required: raise ValueError('unexpected or missing configuration fields')
    if config['confirmatory'] or config['mode'] == 'confirmatory':
        raise ValueError('BLOCKED: confirmatory protocol and independent approval required')
    if (config['schema_version'] != 1 or config['mode'] != 'development' or config['split'] != 'development'
            or config['model'] not in ('openclip','dinov2','dinov3') or not config['model_revision']
            or config['device'] not in ('cpu','cuda') or type(config['batch_size']) is not int or config['batch_size'] < 1
            or config['views'] != 16 or config['precision'] != 'float32' or config['frozen'] is not True or config['resume'] is not True):
        raise ValueError('unsupported development configuration')
    return config
