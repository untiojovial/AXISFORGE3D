"""Run and resume development-only paired finite-catalogue experiments.

The oracle is a separate process. A policy sees a fixed training archive and one
released observation per reserved candidate. The locked manifest is independent
of input catalogues. All coordinates used for visualization share a named frame.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from .transfer_policy import CataloguePolicy, FrozenScale, HistoricalTask


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as file:
        try:
            json.dump(data, file, separators=(',', ':'), allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
            os.replace(file.name, path)
        finally:
            Path(file.name).unlink(missing_ok=True)


PREPILOT_MANIFEST = '6c5291288adcd365b873a4c983b30b9f19fb2ed0110a0575112514b866cdf660'


class DevelopmentData:
    def __init__(self, manifest, lock, data_root):
        if (lock['phase'] != 'development' or lock['manifest_sha256'] != PREPILOT_MANIFEST
                or digest(manifest) != PREPILOT_MANIFEST):
            raise ValueError('manifest does not match the checked-in development lock')
        self.root = Path(data_root).resolve()
        self.manifest_digest = lock['manifest_sha256']
        self.objects = {}
        self._catalogue_hashes = {}
        self._vision_cache = {}
        groups, meshes = {}, set()
        for record in load(manifest)['objects']:
            name, split, group = record['object_id'], record['split'], record['group']
            if (not name or not group or split not in {'train', 'validation', 'test'}
                    or name in self.objects or record['processed_sha256'] in meshes
                    or groups.setdefault(group, split) != split):
                raise ValueError('duplicate object, split, mesh or cross-split group')
            self.objects[name] = record
            meshes.add(record['processed_sha256'])
        self.splits = {name: row['split'] for name, row in self.objects.items()}
        for split in ('train', 'validation', 'test'):
            if list(self.splits.values()).count(split) != lock[split + '_size']:
                raise ValueError('unexpected split size')

    def catalogue(self, path, binary):
        catalog = load(path)
        record = self.objects[catalog['object_id']]
        if record['split'] == 'test' or catalog['split'] != record['split']:
            raise ValueError('development access denied')
        mesh = (self.root / record['visual_mesh']).resolve()
        if not mesh.is_relative_to(self.root) or mesh != Path(catalog['mesh_file']).resolve():
            raise ValueError('unexpected mesh path')
        if (catalog['metadata']['manifest_sha256'] != self.manifest_digest
                or not digest(mesh) == catalog['metadata']['processed_sha256'] == record['processed_sha256']
                or digest(binary) != catalog['metadata']['binary_sha256']
                or digest(catalog['project']) != catalog['metadata']['project_sha256']):
            raise ValueError('catalogue provenance mismatch')
        metadata = catalog['metadata']
        if metadata.get('vision_archive_sha256'):
            archive = Path(metadata['vision_archive']).resolve()
            provenance = (archive.parent / 'provenance.json').resolve()
            if not archive.is_relative_to(self.root) or not provenance.is_relative_to(self.root):
                raise ValueError('vision files must be under the data root')
            key = (archive, provenance, metadata['vision_archive_sha256'], metadata['vision_provenance_sha256'])
            if key not in self._vision_cache:
                source = load(provenance)
                model_sha = 'd867053b2301634007ed9af230bfb1a217ec634f6c0329f04092133ae5c4b89e'
                weights = self.root/'weights'/'vit_b_32-quickgelu-laion400m_e31-d867053b.pt'
                if (source != metadata['vision_provenance'] or digest(archive) != key[2] or digest(provenance) != key[3]
                        or source['manifest_sha256'] != self.manifest_digest
                        or source['checkpoint_sha256'] != model_sha or digest(weights) != model_sha):
                    raise ValueError('unverified OpenCLIP features')
                with np.load(archive, allow_pickle=False) as features:
                    labels = list(zip(features['ids'].tolist(), features['splits'].tolist(), strict=True))
                    if (len(set(k for k, _ in labels)) != len(labels)
                            or any(self.splits.get(k) != v or v == 'test' for k, v in labels)):
                        raise ValueError('unauthorized entries in development vision archive')
                    self._vision_cache[key] = dict(zip((k for k, _ in labels),features['embedding_4'],strict=True))
            if not np.array_equal(np.asarray(catalog['context'][3:]), self._vision_cache[key][catalog['object_id']]):
                raise ValueError('visual context differs from its object embedding')
        ids = catalog['ids']
        features = np.asarray(catalog['features'], dtype=float)
        if (not ids or set(ids) != set(catalog['candidates']) or len(ids) != len(set(ids))
                or features.ndim != 2 or len(features) != len(ids) or not np.isfinite(features).all()):
            raise ValueError('invalid pre-query candidate catalogue')
        if 'functional_features' in metadata or features.shape[1] == 33:
            lock_path = Path(__file__).resolve().parents[2]/'research/phase3/vision/functional_catalogues.lock.json'
            if digest(lock_path) != '83d62c8c909e7fa3ef9d2702a9da9e55b707540aff88d25331dc353b002d9af9':
                raise ValueError('unknown functional catalogue lock')
            locked = load(lock_path)['files']
            pca = load(Path(path).parent/'pca.json')
            if (features.shape[1] != 33 or fingerprint(catalog) != locked.get(catalog['object_id'])
                    or fingerprint(pca) != locked['pca'] or metadata['functional_features']['pca_sha256'] != locked['pca']
                    or any(self.splits.get(name) != 'train' for name in pca['objects'])):
                raise ValueError('unverified functional features or PCA training population')
        self._catalogue_hashes[catalog['object_id']] = fingerprint(catalog)
        return catalog

    def require_catalogue(self, catalog, split):
        if (self.splits.get(catalog['object_id']) != split
                or self._catalogue_hashes.get(catalog['object_id']) != fingerprint(catalog)):
            raise ValueError('unverified catalogue or split')


def fit_history(data, training, fidelity):
    """Use only measured, previously released training rows at this fidelity."""
    history = []
    for catalog, log in training:
        data.require_catalogue(catalog, 'train')
        if catalog['split'] != 'train' or log['object_id'] != catalog['object_id']:
            raise ValueError('history must come from its own training catalogue')
        if log['catalogue_digest'] != fingerprint(catalog) or log['fidelity'] != fidelity:
            raise ValueError('historical artifact uses different labels or features')
        selected = log['selected']
        if (not selected or len(set(selected)) != len(selected) or not set(selected) <= set(catalog['ids'])
                or [row['candidate_id'] for row in log['rows']] != selected
                or any(row['status'] not in ('ok', 'physical_failure', 'infrastructure_failure')
                       or not np.isfinite(row['cost']) or row['cost'] < 0 for row in log['rows'])):
            raise ValueError('inconsistent historical query log')
        rows = [row for row in log['rows'] if row['status'] in ('ok', 'physical_failure')]
        if not rows:
            raise ValueError('training object has no measurements')
        indices = [catalog['ids'].index(row['candidate_id']) for row in rows]
        history.append(HistoricalTask(catalog['object_id'], log['completed_at'],
                         np.array(catalog['context']), np.asarray(catalog['features'])[indices],
                         np.asarray([row['utility'] for row in rows]), sum(r['cost'] for r in log['rows'])))
    if not history:
        raise ValueError('a frozen train archive is required')
    x = np.concatenate([task.features for task in history])
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.
    scaler = FrozenScale(x.mean(axis=0), scale, fingerprint([log for _, log in training]))
    return history, scaler


def run_policy(data, catalog, archive, *, method, seed, budget, initial_count, fidelity,
               target_time, observe, checkpoint, context_mode='combined'):
    data.require_catalogue(catalog, 'validation')
    history, scaler = fit_history(data, archive, fidelity)
    if catalog['split'] != 'validation' or not 1 <= initial_count <= budget:
        raise ValueError('invalid pilot target or initialization')
    if context_mode not in ('combined','visual','geometry'):
        raise ValueError('unknown context mode')
    context = np.asarray(catalog['context'])
    if context_mode != 'combined' or method == 'visual_fixed':
        if not catalog['metadata'].get('vision_archive_sha256') or len(context) != 515:
            raise ValueError('verified geometry and OpenCLIP context needed')
        if any(not c['metadata'].get('vision_archive_sha256') or len(c['context']) != 515 for c,_ in archive):
            raise ValueError('inconsistent historical context modality')
        if method == 'visual_fixed' and context_mode != 'visual':
            raise ValueError('visual-only baseline requires a visual-only context')
    if context_mode != 'combined':
        block=slice(3,None) if context_mode=='visual' else slice(None,3)
        context=context[block]
        history=[replace(task,context=task.context[block]) for task in history]
    metadata=catalog['metadata'].get('vision_provenance',{})
    provenance=dict(model=metadata.get('model'),checkpoint_sha256=metadata.get('checkpoint_sha256'),
                    preprocessing=metadata.get('transform'))
    ids = catalog['ids']
    initial = np.random.default_rng(seed).choice(ids, initial_count, replace=False).tolist()
    policy = CataloguePolicy(ids, scaler.transform(catalog['features']), context,
                    history, target_time=target_time, manifest_splits=data.splits,
                    manifest_digest=fingerprint(data.splits), target_id=catalog['object_id'],
                    scaler=scaler, method=method, budget=budget, seed=seed, initial_ids=initial,
                    visual_provenance=provenance)
    definition = dict(catalogue_digest=fingerprint(catalog), manifest_sha256=data.manifest_digest,
                   archive_digest=scaler.training_digest, initial=initial, budget=budget, seed=seed,
                   method=method, fidelity=fidelity, object_id=catalog['object_id'], target_time=target_time)
    if context_mode != 'combined':
        definition['context_mode'] = context_mode
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.with_suffix('.lock').open('a') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        state = load(checkpoint) if checkpoint.exists() else dict(definition=definition, rows=[], pending=None)
        if state['definition'] != definition:
            raise ValueError('checkpoint belongs to another immutable run')
        for row in state['rows']:
            if policy.ask() != row['candidate_id']:
                raise ValueError('candidate sequence diverged during replay')
            policy.tell(row['candidate_id'], status=row['status'], cost=row['cost'], utility=row['utility'])
            if policy.records[-1]['prediction'] != row['prediction']:
                raise ValueError('posterior diverged during replay')
        while len(state['rows']) < budget:
            candidate = policy.ask()
            if state['pending'] not in (None, candidate):
                raise ValueError('reserved candidate changed')
            state['pending'] = candidate
            save(checkpoint, state)
            observation = observe(candidate)
            policy.tell(candidate, status=observation['status'], cost=observation['cost'],
                        utility=observation['utility'])
            state['rows'].append(dict(**policy.records[-1], details=observation['details']))
            state['pending'] = None
            save(checkpoint, state)
        return state


def viewer_report(catalogues, runs):
    objects = {c['object_id']: dict(frame=c['frame'], points=c['points'], candidates=c['candidates'], processed_sha256=c['metadata']['processed_sha256'])
               for c in catalogues}
    return dict(schema='axisforge.research.v1', objects=objects,
                runs=[dict(**run['definition'], rows=run['rows']) for run in runs])
