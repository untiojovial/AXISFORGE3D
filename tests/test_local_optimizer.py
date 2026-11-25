import unittest
import numpy as np
from axisforge_bo.space import Box
from axisforge_bo.local_optimizer import LocalBayesianOptimizer


class LocalOptimizerTests(unittest.TestCase):
    def test_loop_and_contract(self):
        bounds = Box([(-2, 2)] * 4)

        def run():
            opt = LocalBayesianOptimizer(bounds, seed=8, pool_size=12)
            with self.assertRaises(RuntimeError):
                opt.best()
            for i in range(9):
                x = opt.ask()
                bounds.validate(x)
                with self.assertRaises(RuntimeError):
                    opt.ask()
                with self.assertRaises(ValueError):
                    opt.tell(x, float("nan"))
                opt.tell(x, float(x @ x))
            return opt

        a, b = run(), run()
        np.testing.assert_array_equal(
            [x for x, y in a.observations], [x for x, y in b.observations]
        )
        self.assertEqual(a.best()[1], min(y for x, y in a.observations))

    def test_restart_and_reservation(self):
        opt = LocalBayesianOptimizer(Box([(0, 1)] * 2), seed=1, pool_size=4)
        x = opt.ask()
        with self.assertRaises(ValueError):
            opt.tell([x[0], 1 - x[1]], 1)
        opt.tell(x, 1)
        opt.region.restart_required = True
        x = opt.ask()
        opt.tell(x, 2)
        self.assertEqual(len(opt.observations), 2)
        self.assertEqual(len(opt._region_observations), 1)
