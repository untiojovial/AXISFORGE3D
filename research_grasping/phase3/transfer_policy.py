"""Causal finite-catalogue policies; research development, not physical validation."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from scipy.stats import qmc
from research_grasping.contextual_transfer import AdaptiveTransfer, Source
from research_grasping.contextual_transfer.baselines import RGPE
from axisforge_bo.models import ExactGP
from .policy_baselines import PooledGP


@dataclass(frozen=True)
class FrozenScale:
    """Parameters fitted exclusively from authorized train data."""
    center: np.ndarray
    scale: np.ndarray
    training_digest: str

    def transform(self, values):
        values = np.asarray(values, float)
        center, scale = np.asarray(self.center, float), np.asarray(self.scale, float)
        if (values.ndim != 2 or center.shape != (values.shape[1],)
                or scale.shape != center.shape or not self.training_digest
                or not all(np.isfinite(a).all() for a in (values, center, scale))
                or np.any(scale <= 0)):
            raise ValueError('invalid frozen feature scaling')
        return (values-center)/scale


@dataclass(frozen=True)
class HistoricalTask:
    object_id: str
    available_at: float
    context: np.ndarray
    features: np.ndarray
    utilities: np.ndarray
    cost: float


def approved_history(tasks, manifest_splits, *, target_id, target_time, scaler):
    """Check immutable caller-supplied manifest mapping; provenance must be audited externally."""
    if target_id not in manifest_splits or not np.isfinite(target_time):
        raise ValueError('unknown target or invalid time')
    sources, seen = [], set()
    for task in tasks:
        if (task.object_id in seen or task.object_id == target_id
                or manifest_splits.get(task.object_id) != 'train'
                or not np.isfinite(task.available_at) or task.available_at >= target_time
                or not np.isfinite(task.cost) or task.cost < 0):
            raise ValueError('unauthorized historical task')
        x = scaler.transform(task.features)
        y = np.asarray(task.utilities, float)
        c = np.asarray(task.context, float)
        if (not len(x) or y.shape != (len(x),) or c.ndim != 1 or not len(c)
                or not np.isfinite(y).all() or not np.isfinite(c).all()):
            raise ValueError('invalid historical observations')
        sources.append(Source(task.object_id, c.copy(), x.copy(), y.copy(), task.available_at))
        seen.add(task.object_id)
    return tuple(sources)


class _MultidimensionalRGPE(RGPE):
    def __init__(self, sources, dimension, **kwargs):
        super().__init__(sources, **kwargs)
        self.x = np.empty((0, dimension))

    def observe(self, x, y):
        new_x = np.vstack((self.x, np.asarray(x, float).reshape(1, -1)))
        new_y = np.append(self.y, float(y))
        target = ExactGP(self.kernel, self.noise).fit(new_x, new_y)
        old_x, old_y, old_target, old_weights = self.x, self.y, self.target, self.weights
        try:
            self.x, self.y, self.target = new_x, new_y, target
            weights = self.rank_weights()
        except Exception:
            self.x, self.y, self.target, self.weights = old_x, old_y, old_target, old_weights
            raise
        self.weights = weights


class CataloguePolicy:
    """ask reserves one candidate; tell consumes a measured result or infrastructure failure.

    Public inputs contain features and source labels, never unobserved target labels.
    B1 is Sobol projected onto a finite catalogue, not uniform Sobol sampling.
    """
    METHODS = {'random', 'sobol_projected', 'target', 'target_local', 'geometry_target',
               'contextual', 'dinov2_contextual', 'dinov3_contextual', 'multitask_fixed',
               'rgpe', 'equal', 'visual_fixed', 'no_evidence', 'no_floor', 'adaptive'}

    def __init__(self, ids, features, context, history, *, manifest_splits, manifest_digest,
                 target_id, scaler, target_time, method, budget, seed, initial_ids, length_scale=1., noise_variance=.01,
                 rgpe_samples=256, visual_provenance=None, icm_correlation=.5):
        ids = tuple(ids)
        x = np.asarray(features, float)
        if (method not in self.METHODS or not ids or len(set(ids)) != len(ids)
                or any(not isinstance(i, str) or not i for i in ids)
                or x.ndim != 2 or x.shape[0] != len(ids) or not x.shape[1]
                or not np.isfinite(x).all() or type(budget) is not int
                or not 0 < budget <= len(ids) or len(initial_ids) > budget
                or len(set(initial_ids)) != len(initial_ids) or not set(initial_ids) <= set(ids)):
            raise ValueError('invalid catalogue, method or budget')
        if method not in {'random', 'sobol_projected'} and not initial_ids:
            raise ValueError('BO requires a shared nonempty initial design')
        if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
            raise ValueError('manifest SHA-256 digest required')
        import hashlib, json
        canonical = json.dumps(manifest_splits, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(canonical).hexdigest() != manifest_digest:
            raise ValueError('manifest digest mismatch')
        sources = approved_history(history, manifest_splits, target_id=target_id,
                                   target_time=target_time, scaler=scaler)
        if method in {'dinov2_contextual', 'dinov3_contextual', 'visual_fixed'}:
            expected = {'dinov2_contextual': 'dinov2', 'dinov3_contextual': 'dinov3'}.get(method)
            if (not isinstance(visual_provenance, dict)
                    or not visual_provenance.get('checkpoint_sha256')
                    or len(visual_provenance['checkpoint_sha256']) != 64
                    or not visual_provenance.get('preprocessing')
                    or not visual_provenance.get('model')
                    or (expected and expected not in visual_provenance['model'].lower())):
                raise ValueError('verified pretrained visual provenance required')
        self.manifest_digest = manifest_digest
        self.ids, self.features, self.method = ids, x.copy(), method
        self.budget, self.initial_ids = budget, tuple(initial_ids)
        self.initial_count = 0
        self.used = set()
        self.pending = None
        self.records = []
        self.rng = np.random.default_rng(seed)
        self.sobol = qmc.Sobol(x.shape[1], scramble=True, seed=seed)
        self.minimum, self.span = x.min(axis=0), np.ptp(x, axis=0)
        self.span[self.span == 0] = 1
        if method in {'contextual', 'dinov2_contextual', 'dinov3_contextual', 'multitask_fixed'}:
            self.model = PooledGP(sources, context, dimension=x.shape[1], noise=noise_variance,
                                  length_scale=length_scale, mode='icm' if method == 'multitask_fixed' else 'contextual',
                                  correlation=icm_correlation)
        elif method == 'rgpe':
            self.model = _MultidimensionalRGPE(sources, x.shape[1], noise=noise_variance,
                                              length_scale=length_scale, seed=seed, samples=rgpe_samples)
        elif method in {'random', 'sobol_projected'}:
            self.model = None
        else:
            self.model = AdaptiveTransfer(context, [] if method in {'target', 'target_local', 'geometry_target'} else sources,
                target_time=target_time, dimension=x.shape[1], length_scale=length_scale,
                noise_variance=noise_variance, k=len(sources), floor=0. if method in {'equal', 'no_floor'} else .1,
                contextual=method != 'equal', use_evidence=method in {'adaptive', 'no_floor'})

    def ask(self):
        if self.pending is not None:
            raise RuntimeError('candidate already reserved')
        if len(self.records) >= self.budget:
            raise StopIteration('budget exhausted')
        available = [i for i, name in enumerate(self.ids) if name not in self.used]
        rng = deepcopy(self.rng)
        sobol = deepcopy(self.sobol)
        next_initial_count = self.initial_count
        if self.initial_count < len(self.initial_ids):
            index = self.ids.index(self.initial_ids[self.initial_count])
            next_initial_count += 1
            score = None
        elif self.method == 'random':
            index = int(rng.choice(available))
            score = None
        elif self.method == 'sobol_projected':
            u = sobol.random(1)[0]
            norm = (self.features[available]-self.minimum)/self.span
            index = available[int(np.argmin(np.sum((norm-u)**2, axis=1)))]
            score = None
        elif not len(self.model.y):
            # Infrastructure failures consume budget but do not form a GP incumbent.
            index = available[0]
            score = None
        else:
            scores = self.model.acquisition(self.features[available])
            index = available[int(np.argmax(scores))]
            score = float(np.max(scores))
        prediction = None
        if self.model is not None:
            point = self.features[index:index+1]
            if isinstance(self.model, _MultidimensionalRGPE):
                mean, covariance = self.model.predict_joint(point)
                variance = covariance.diagonal()
            else:
                mean, variance = self.model.predict(point)
            prediction = {'mean': float(mean[0]), 'variance': float(variance[0]),
                          'weights': self.model.weights.tolist() if self.model.weights is not None else None}
        self.rng, self.sobol, self.initial_count = rng, sobol, next_initial_count
        self.pending = (index, score, prediction)
        return self.ids[index]

    def tell(self, candidate_id, *, status, cost, utility=None):
        if self.pending is None or candidate_id != self.ids[self.pending[0]]:
            raise ValueError('candidate is not reserved')
        if status not in {'ok', 'physical_failure', 'infrastructure_failure'} or not np.isfinite(cost) or cost < 0:
            raise ValueError('invalid status or cost')
        valid = status != 'infrastructure_failure'
        if (valid and (utility is None or not np.isfinite(utility))) or (not valid and utility is not None):
            raise ValueError('utility inconsistent with status')
        index, score, prediction = self.pending
        if valid and self.model is not None:
            self.model.observe(self.features[index], utility)
        self.records.append({'candidate_id': candidate_id, 'status': status, 'utility': utility,
                             'cost': float(cost), 'acquisition': score, 'prediction': prediction})
        self.used.add(candidate_id)
        self.pending = None
