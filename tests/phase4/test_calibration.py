import unittest
import numpy as np
from research_grasping.phase4.calibration import fold_design, metrics
from research_grasping.phase4.calibration_recalibrate import inner_groups


class CalibrationTests(unittest.TestCase):
    def test_known_metrics_and_noise_guard(self):
        m=metrics([0,1],[0,0],[1,1])
        self.assertEqual(m['mae'],.5)
        self.assertAlmostEqual(m['rmse'],2**-.5)
        self.assertEqual(m['coverage90'],1)
        self.assertGreater(m['width90'],3)
        with self.assertRaises(ValueError):metrics([0],[0],[-1])
        with self.assertRaises(ValueError):metrics([0],[float('nan')],[1])

    def test_nested_groups_exclude_outer(self):
        names=[str(i) for i in range(16)]
        groups=inner_groups(names, "0")
        self.assertEqual([len(g) for g in groups],[5,5,5])
        self.assertEqual(set(sum(groups,[])),set(names)-{"0"})
        with self.assertRaises(ValueError):inner_groups(names,"absent")

    def test_heldout_changes_do_not_change_training_transform(self):
        rng=np.random.default_rng(91); local=rng.normal(size=(12,3));context=rng.normal(size=(12,6))
        tr=np.arange(12)<10;te=~tr
        before,_=fold_design(local,context,tr,te)
        local[te]=1e6;context[te]=-1e6
        after,_=fold_design(local,context,tr,te)
        np.testing.assert_allclose(before,after)


if __name__=='__main__':unittest.main()
