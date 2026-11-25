import unittest
import numpy as np
from axisforge_bo.space import Box
from axisforge_bo.trust_region import TrustRegion


class TrustRegionTests(unittest.TestCase):
    def test_update_expand_shrink_restart(self):
        t = TrustRegion(
            2,
            length=0.5,
            min_length=0.2,
            max_length=1,
            success_tolerance=2,
            failure_tolerance=2,
        )
        t.update([5, 6])
        t.update([4])
        self.assertEqual(t.length, 1)
        t.update([4])
        t.update([4])
        self.assertEqual(t.length, 0.5)
        t.update([5])
        t.update([5])
        self.assertEqual(t.length, 0.25)
        t.update([5])
        t.update([5])
        self.assertTrue(t.restart_required)
        with self.assertRaises(ValueError):
            t.update([3])
        t.restart(0.5)
        self.assertFalse(t.restart_required)
        self.assertTrue(np.isinf(t.best_value))

    def test_scaled_bounds_and_samples(self):
        b = Box([(0, 10), (-1, 1)])
        t = TrustRegion(2, length=0.4)
        bounds = t.bounds(b, [0, 0], [4, 1])
        np.testing.assert_allclose(bounds, [[0, 4], [-0.2, 0.2]])
        points = t.sample(b, [0, 0], 50, np.random.default_rng(2), [4, 1])
        self.assertTrue(np.all(points >= bounds[:, 0]))
        self.assertTrue(np.all(points <= bounds[:, 1]))
        with self.assertRaises(ValueError):
            t.bounds(b, [0, 0], [0, 1])
        with self.assertRaises(ValueError):
            t.update([np.nan])


class TrustRegionInvalidTests(unittest.TestCase):
    def test_invalid_settings(self):
        with self.assertRaises(ValueError):
            TrustRegion(0)
        t = TrustRegion(2)
        with self.assertRaises(ValueError):
            t.restart(-1)
        with self.assertRaises(ValueError):
            t.bounds(Box([(0, 1)]), [0])
        with self.assertRaises(ValueError):
            t.sample(Box([(0, 1)] * 2), [0, 0], -1, np.random.default_rng())
        t.restart_required = True
        with self.assertRaises(ValueError):
            t.bounds(Box([(0, 1)] * 2), [0, 0])
