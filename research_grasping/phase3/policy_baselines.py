"""Simple pooled contextual and fixed-ICM GP controls, not exact TFM or learned B7."""
import numpy as np
from axisforge_bo.models import ExactGP, StationaryKernel
from research_grasping.contextual_transfer.baselines import _ei


class _ConditionalKernel:
    def __init__(self, length_scale, correlation):
        if not np.isfinite(correlation) or not 0 <= correlation < 1:
            raise ValueError('ICM correlation must be in [0, 1)')
        self.spatial = StationaryKernel(length_scale=length_scale)
        self.correlation = correlation

    def __call__(self, x, y):
        task = self.correlation + (1-self.correlation) * (x[:, -1, None] == y[None, :, -1])
        return self.spatial(x[:, :-1], y[:, :-1]) * task

    def diagonal(self, x):
        return self.spatial.diagonal(x[:, :-1])


class PooledGP:
    """GP pooled over train source tasks and target observations.

    `mode=contextual`: concatenate fixed, caller-scaled context.
    `mode=icm`: fixed PSD task covariance; not a learned multitask model.
    """
    def __init__(self, sources, context, *, dimension, noise, length_scale, mode, correlation=.5):
        if mode not in {'contextual', 'icm'}:
            raise ValueError('invalid pooled GP mode')
        context = np.asarray(context, float)
        if context.ndim != 1 or not len(context) or not np.isfinite(context).all():
            raise ValueError('invalid context')
        self.context, self.dimension, self.mode, self.noise = context, dimension, mode, noise
        self.task_id = len(sources)
        self.kernel = (_ConditionalKernel(length_scale, correlation) if mode == 'icm'
                       else StationaryKernel(length_scale=length_scale))
        rows, values = [], []
        for i, source in enumerate(sources):
            x = np.asarray(source.x, float)
            c = np.asarray(source.context, float)
            if x.ndim != 2 or x.shape[1] != dimension or c.shape != context.shape:
                raise ValueError('incompatible source features or context')
            suffix = np.full((len(x), 1), i) if mode == 'icm' else np.tile(c, (len(x), 1))
            rows.append(np.column_stack((x, suffix)))
            values.append(np.asarray(source.y, float))
        self.x = np.vstack(rows) if rows else np.empty((0, dimension + (1 if mode == 'icm' else len(context))))
        self.all_y = np.concatenate(values) if values else np.empty(0)
        self.y = np.empty(0)
        self.model = ExactGP(self.kernel, noise).fit(self.x, self.all_y)
        self.weights = None

    def _augment(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError('invalid features')
        suffix = (np.full((len(x), 1), self.task_id) if self.mode == 'icm'
                  else np.tile(self.context, (len(x), 1)))
        return np.column_stack((x, suffix))

    def predict(self, x):
        return self.model.predict(self._augment(x))

    def observe(self, x, y):
        y = float(y)
        if not np.isfinite(y):
            raise ValueError('invalid observation')
        xx = self._augment(np.asarray(x, float).reshape(1, -1))
        new_x = np.vstack((self.x, xx))
        new_y = np.append(self.all_y, y)
        model = ExactGP(self.kernel, self.noise).fit(new_x, new_y)
        self.x, self.all_y, self.model = new_x, new_y, model
        self.y = np.append(self.y, y)

    def acquisition(self, x):
        if not len(self.y):
            raise ValueError('EI requires target observations')
        mean, variance = self.predict(x)
        return _ei(mean, variance, max(self.y))
