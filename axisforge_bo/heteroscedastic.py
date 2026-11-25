"""Replicate-based plug-in heteroscedastic regression.

Requires at least two independent Gaussian replicates at every exactly matching
input. This two-stage estimator is not joint Bayesian inference.
"""

import numpy as np
from scipy.special import digamma, polygamma

from .models import ExactGP


class ReplicatedNoiseGP:
    """Learn log noise from within-group sample variances and fit group means.

    ``noise_floor`` is a strictly positive variance in objective-squared units.
    The log-variance GP models bias-corrected log sample variances with known
    log-chi-square observation variances. Its zero-mean prior is centered by the
    weighted average corrected log variance. The mean GP uses the empirical
    variance of each group mean, max(sample variance, floor) / replicate count.
    Noise predictions use E[exp(log variance)] under the log-variance posterior.
    They are plug-in estimates, not a fully joint posterior over mean and noise.
    """

    def __init__(self, mean_kernel, noise_kernel, noise_floor=1e-8, jitter=1e-10):
        self.mean_kernel = mean_kernel
        self.noise_kernel = noise_kernel
        self.noise_floor = float(noise_floor)
        self.jitter = float(jitter)
        if not np.isfinite(self.noise_floor) or self.noise_floor <= 0:
            raise ValueError("noise floor must be finite and positive")
        if not np.isfinite(self.jitter) or self.jitter < 0:
            raise ValueError("jitter must be finite and nonnegative")
        self.mean_model = None
        self.noise_model = None
        self.log_noise_center = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if (
            X.ndim != 2
            or not X.shape[0]
            or not X.shape[1]
            or y.shape != (len(X),)
            or not np.isfinite(X).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("invalid replicate data")
        unique, inverse, counts = np.unique(
            X, axis=0, return_inverse=True, return_counts=True
        )
        if np.any(counts < 2):
            raise ValueError("each distinct input requires at least two replicates")
        means = np.empty(len(unique))
        variances = np.empty(len(unique))
        for i in range(len(unique)):
            group = y[inverse == i]
            with np.errstate(over="ignore", invalid="ignore"):
                means[i] = group.mean()
                variances[i] = np.var(group, ddof=1)
        if not np.isfinite(means).all() or not np.isfinite(variances).all():
            raise ArithmeticError("replicate moments overflowed")
        variances = np.maximum(variances, self.noise_floor)
        df = counts - 1
        correction = digamma(df / 2) + np.log(2) - np.log(df)
        log_y = np.log(variances) - correction
        log_var = polygamma(1, df / 2)
        center = float(np.average(log_y, weights=1 / log_var))
        noise_model = ExactGP(
            self.noise_kernel, noise_variance=log_var, jitter=self.jitter
        ).fit(unique, log_y - center)
        mean_model = ExactGP(
            self.mean_kernel, noise_variance=variances / counts, jitter=self.jitter
        ).fit(unique, means)
        self.group_inputs = unique.copy()
        self.group_counts = counts.copy()
        self.group_means = means.copy()
        self.group_variances = variances.copy()
        self.log_noise_center = center
        self.noise_model = noise_model
        self.mean_model = mean_model
        return self

    def predict(self, X):
        """Return latent mean, latent variance, and individual observation variance.

        The observation variance includes latent predictive variance plus estimated
        individual measurement noise. It is not the variance of a replicate mean.
        """
        if self.mean_model is None:
            raise ValueError("fit first")
        mean, latent_variance = self.mean_model.predict(X)
        log_mean, log_variance = self.noise_model.predict(X)
        with np.errstate(over="ignore", invalid="ignore"):
            noise = np.maximum(
                np.exp(self.log_noise_center + log_mean + 0.5 * log_variance),
                self.noise_floor,
            )
            observed_variance = latent_variance + noise
        if not np.isfinite(observed_variance).all():
            raise ArithmeticError("noise prediction overflowed")
        return mean, latent_variance, observed_variance
