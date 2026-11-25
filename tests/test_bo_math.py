import unittest
import numpy as np
from scipy.stats import norm
from axisforge_bo.models import ExactGP, StationaryKernel
from axisforge_bo.space import Box, MixedSpace, Real, Integer, Categorical
from axisforge_bo.acquisitions import (
    expected_improvement,
    probability_improvement,
    lower_confidence_bound,
)


class MathTests(unittest.TestCase):
    def test_gp_reference(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([1.0, 2.0])
        x = np.array([[0.4]])
        kernel = StationaryKernel(0.8, 1.2)
        gp = ExactGP(kernel, noise_variance=0.1, jitter=1e-10).fit(X, y)
        K = kernel(X, X) + np.eye(2) * (0.1 + 1e-10)
        k = kernel(X, x)
        mean, var = gp.predict(x)
        np.testing.assert_allclose(
            mean, k.T @ np.linalg.solve(K, y), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            var,
            np.diag(kernel(x, x)) - np.diag(k.T @ np.linalg.solve(K, k)),
            rtol=1e-10,
            atol=1e-10,
        )
        ll = (
            -0.5 * y @ np.linalg.solve(K, y)
            - 0.5 * np.linalg.slogdet(K)[1]
            - np.log(2 * np.pi)
        )
        self.assertAlmostEqual(gp.log_marginal_likelihood(), ll, places=10)
        clone = ExactGP.from_config(gp.to_config()).fit(X, y)
        np.testing.assert_allclose(clone.predict(x)[0], mean)

    def test_prior_and_degenerate(self):
        g = ExactGP(StationaryKernel(), 0, 0)
        np.testing.assert_allclose(g.predict([[0], [1]])[1], [1, 1])
        with self.assertRaises(np.linalg.LinAlgError):
            g.fit([[0], [0]], [1, 1])
        g = ExactGP(StationaryKernel(), [0.1, 0.2], 1e-10).fit([[0], [0]], [1, 2])
        self.assertTrue(np.isfinite(g.predict([[0]])[0]).all())
        with self.assertRaises(ValueError):
            g.predict([[float("nan")]])

    def test_acquisitions(self):
        d = 0.5
        sig = 2.0
        z = d / sig
        self.assertAlmostEqual(
            float(expected_improvement([0], [sig**2], d)[0]),
            d * norm.cdf(z) + sig * norm.pdf(z),
        )
        np.testing.assert_allclose(expected_improvement([0, 1], [0, 0], 0.5), [0.5, 0])
        np.testing.assert_allclose(probability_improvement([0, 1], [0, 0], 0.5), [1, 0])
        np.testing.assert_allclose(lower_confidence_bound([1], [4], 4), [-3])

    def test_spaces(self):
        b = Box([(0, 1), (-2, 2)])
        self.assertEqual(b.sample(np.random.default_rng(42), 3).shape, (3, 2))
        with self.assertRaises(ValueError):
            b.validate([2, 0])
        s = MixedSpace([Real(0, 1), Integer(1, 3), Categorical(("a", "b"))])
        for x in s.sample(np.random.default_rng(3), 20):
            s.validate(x)
        with self.assertRaises(ValueError):
            s.validate((0.1, 1.2, "a"))


if __name__ == "__main__":
    unittest.main()


class ExtremeTests(unittest.TestCase):
    def test_overflow_fails(self):
        with self.assertRaises(ArithmeticError):
            expected_improvement([-1e308], [1], 1e308)
        with np.errstate(all="ignore"):
            with self.assertRaises(ArithmeticError):
                StationaryKernel(kind="matern52")([[-1e308]], [[1e308]])
        with self.assertRaises(ValueError):
            Box([(-1e308, 1e308)])
        with self.assertRaises(ArithmeticError):
            lower_confidence_bound([-1e308], [1e308], 1e308)


class ExtensionTests(unittest.TestCase):
    def test_design_and_encoding(self):
        from axisforge_bo.sampling import design

        space = MixedSpace([Real(0, 1), Integer(1, 3), Categorical(("a", "b"))])
        point = (0.3, 2, "b")
        self.assertEqual(space.decode(space.encode(point)), point)
        for method in ("random", "sobol", "latin_hypercube"):
            box = Box([(0, 1), (-1, 1)])
            a = design(box, 8, method, 10)
            np.testing.assert_array_equal(a, design(box, 8, method, 10))
            for x in a:
                box.validate(x)
        with self.assertRaises(ValueError):
            design(box, 3, "sobol", 10)

    def test_kernels_covariance_and_logei(self):
        from axisforge_bo.models import CompositeKernel, LinearKernel, PeriodicKernel
        from axisforge_bo.acquisitions import log_expected_improvement

        x = np.array([[0.0], [0.4], [1.0]])
        for k in [
            LinearKernel(),
            PeriodicKernel(),
            CompositeKernel(StationaryKernel(), PeriodicKernel(), "product"),
        ]:
            K = k(x, x)
            np.testing.assert_allclose(K, K.T, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-10)
        gp = ExactGP(StationaryKernel()).fit(x, [0, 1, 2])
        _, cov = gp.predict_joint(x)
        np.testing.assert_allclose(np.diag(cov), gp.predict(x)[1], atol=1e-10)
        np.testing.assert_allclose(
            np.exp(log_expected_improvement([0, 1], [1, 1], 0)),
            expected_improvement([0, 1], [1, 1], 0),
            rtol=1e-12,
        )
        self.assertTrue(np.isfinite(log_expected_improvement([100], [1], 0)).all())


class IndependentReferenceTests(unittest.TestCase):
    def test_sklearn_posterior(self):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        except ImportError:
            self.skipTest("scikit-learn optional independent reference unavailable")
        x = np.array([[0.0], [0.4], [1.0]])
        y = np.array([0.2, 1.0, -0.4])
        query = np.array([[0.1], [0.75]])
        reference = GaussianProcessRegressor(
            kernel=ConstantKernel(1.44, constant_value_bounds="fixed")
            * RBF(0.8, length_scale_bounds="fixed"),
            alpha=0.05 + 1e-10,
            optimizer=None,
        ).fit(x, y)
        ref_mean, ref_cov = reference.predict(query, return_cov=True)
        model = ExactGP(StationaryKernel(0.8, 1.2), 0.05, 1e-10).fit(x, y)
        mean, cov = model.predict_joint(query)
        np.testing.assert_allclose(mean, ref_mean, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(cov, ref_cov, atol=1e-10, rtol=1e-10)


class ModelStateTests(unittest.TestCase):
    def test_failed_fit_preserves_state_and_diagonal_path(self):
        class CountingKernel(StationaryKernel):
            def __call__(self, x, y):
                if len(x) == 20 and len(y) == 20:
                    raise AssertionError(
                        "marginal prediction constructed full covariance"
                    )
                return super().__call__(x, y)

        model = ExactGP(CountingKernel(), 0, 0).fit([[0.0]], [1.0])
        before = model.predict([[0.3]])
        with self.assertRaises(np.linalg.LinAlgError):
            model.fit([[0.0], [0.0]], [1.0, 1.0])
        np.testing.assert_allclose(before[0], model.predict([[0.3]])[0])
        model.predict(np.arange(20, dtype=float).reshape(-1, 1))


class ThompsonTests(unittest.TestCase):
    def test_reproducible_and_correlated(self):
        from axisforge_bo.acquisitions import thompson_sample

        covariance = np.ones((2, 2))
        a = thompson_sample([0, 0], covariance, np.random.default_rng(7))
        np.testing.assert_allclose(a[0], a[1], atol=1e-12)
        np.testing.assert_array_equal(
            a, thompson_sample([0, 0], covariance, np.random.default_rng(7))
        )
        with self.assertRaises(ValueError):
            thompson_sample([0, 0], [[1, 2], [2, 1]], np.random.default_rng(1))


class LogEIReferenceTests(unittest.TestCase):
    def test_high_precision_negative_tail(self):
        try:
            import mpmath as mp
        except ImportError:
            self.skipTest("mpmath optional")
        from axisforge_bo.acquisitions import log_expected_improvement

        with mp.workdps(100):
            for z in (-20, -10, -3, 0, 2):
                zmp = mp.mpf(z)
                phi = mp.exp(-zmp * zmp / 2) / mp.sqrt(2 * mp.pi)
                cdf = mp.erfc(-zmp / mp.sqrt(2)) / 2
                expected = float(mp.log(phi + zmp * cdf))
                actual = float(log_expected_improvement([-z], [1], 0)[0])
                self.assertAlmostEqual(actual, expected, places=8)


class MarginalLikelihoodExtremeTests(unittest.TestCase):
    def test_overflow_is_explicit(self):
        model = ExactGP(StationaryKernel(), noise_variance=1).fit([[0.0]], [1e308])
        with self.assertRaises(ArithmeticError):
            model.log_marginal_likelihood()


class InvalidAndKernelTests(unittest.TestCase):
    def test_invalid_acquisitions(self):
        from axisforge_bo.acquisitions import (
            log_expected_improvement,
            register_acquisition,
            thompson_sample,
        )

        for fn in (expected_improvement, probability_improvement):
            with self.assertRaises(ValueError):
                fn([0], [-1], 0)
            with self.assertRaises(ValueError):
                fn([0], [1], 0, -1)
        with self.assertRaises(ValueError):
            lower_confidence_bound([0], [1], -1)
        with self.assertRaises(ValueError):
            log_expected_improvement([0], [0], 0)
        with self.assertRaises(ValueError):
            register_acquisition("ei", lambda x: x)
        with self.assertRaises(ValueError):
            thompson_sample([0, 0], [[1, 0], [2, 1]], np.random.default_rng())

    def test_kernels(self):
        from axisforge_bo.models import LinearKernel, PeriodicKernel, CompositeKernel

        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        for kind in ("rbf", "matern12", "matern32", "matern52", "rational_quadratic"):
            k = StationaryKernel([1, 2], 2, kind)
            np.testing.assert_allclose(np.diag(k(x, x)), k.diagonal(x))
            self.assertGreaterEqual(np.linalg.eigvalsh(k(x, x)).min(), -1e-10)
        with self.assertRaises(ValueError):
            StationaryKernel(0)
        with self.assertRaises(ValueError):
            StationaryKernel(kind="unknown")
        with self.assertRaises(ValueError):
            StationaryKernel([1, 2, 3])(x, x)
        for k in (
            LinearKernel(),
            PeriodicKernel(),
            CompositeKernel(LinearKernel(), PeriodicKernel(), "sum"),
        ):
            np.testing.assert_allclose(k.diagonal(x), np.diag(k(x, x)), atol=1e-12)
        with self.assertRaises(ValueError):
            CompositeKernel(LinearKernel(), LinearKernel(), "bad")
        with self.assertRaises(ValueError):
            PeriodicKernel(period=0)
        with self.assertRaises(ValueError):
            LinearKernel(variance=0)

    def test_invalid_models_spaces(self):
        from axisforge_bo.sampling import design

        with self.assertRaises(ValueError):
            ExactGP(StationaryKernel(), -1)
        with self.assertRaises(ValueError):
            ExactGP(StationaryKernel(), jitter=-1)
        gp = ExactGP(StationaryKernel())
        with self.assertRaises(ValueError):
            gp.log_marginal_likelihood()
        with self.assertRaises(ValueError):
            gp.fit([[0]], [float("nan")])
        with self.assertRaises(ValueError):
            ExactGP(StationaryKernel(), [1, 2]).fit([[0]], [0])
        gp.fit(np.empty((0, 1)), np.empty(0))
        np.testing.assert_allclose(gp.predict([[0]])[1], [1])
        with self.assertRaises(ValueError):
            gp.predict([[0, 1]])
        with self.assertRaises(ValueError):
            Box([(1, 1)])
        with self.assertRaises(ValueError):
            Real(1, 0)
        with self.assertRaises(ValueError):
            Integer(2, 1)
        with self.assertRaises(ValueError):
            Categorical(("x", "x"))
        with self.assertRaises(ValueError):
            MixedSpace([])
        mixed = MixedSpace([Integer(0, 2), Categorical(("a", "b"))])
        with self.assertRaises(ValueError):
            mixed.validate([1])
        with self.assertRaises(ValueError):
            mixed.validate([1, "c"])
        with self.assertRaises(ValueError):
            mixed.decode([0.25, 1, 0])
        with self.assertRaises(ValueError):
            mixed.decode([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            design(Box([(0, 1)]), 1, "unknown")


class AdditionalAdversarialTests(unittest.TestCase):
    def test_acquisition_failures_and_registry(self):
        from axisforge_bo.acquisitions import (
            log_expected_improvement,
            register_acquisition,
            ACQUISITIONS,
            thompson_sample,
        )

        with self.assertRaises(ArithmeticError):
            probability_improvement([-1e308], [1], 1e308)
        with self.assertRaises(ArithmeticError):
            log_expected_improvement([-1e308], [1], 1e308)
        with self.assertRaises(ValueError):
            thompson_sample([1], [[float("nan")]], np.random.default_rng())
        register_acquisition("test_temporary", lambda x: x)
        self.assertIn("test_temporary", ACQUISITIONS)
        del ACQUISITIONS["test_temporary"]

    def test_space_edge_cases(self):
        from axisforge_bo.sampling import design

        s = MixedSpace([Integer(2, 2), Categorical(("a", "b"))])
        np.testing.assert_allclose(s.encode((2, "a")), [0, 1, 0])
        self.assertEqual(s.decode([0, 1, 0]), (2, "a"))
        with self.assertRaises(ValueError):
            s.sample(np.random.default_rng(), -1)
        with self.assertRaises(ValueError):
            Box([(0, 1)]).sample(np.random.default_rng(), -1)
        with self.assertRaises(ValueError):
            design(Box([(0, 1)]), -1)
