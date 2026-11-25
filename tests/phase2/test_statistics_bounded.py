import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bounded",
    Path(__file__).parents[2] / "research_grasping/statistics/simulate_bounded.py",
)
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)


class BoundedTest(unittest.TestCase):
    def test_seed_and_bounds(self):
        first = bounded.run(3, 42)
        self.assertEqual(first, bounded.run(3, 42))
        self.assertNotEqual(first, bounded.run(3, 43))
        for row in first["rows"]:
            self.assertGreaterEqual(row["mu"] - 1 / 6, -1)
            self.assertLessEqual(row["mu"] + 5 / 6, 1)
            self.assertGreater(
                row["hoeffding_two_sided_radius"], row["hoeffding_one_sided_radius"]
            )
            self.assertEqual(row["nominal_evaluations"], 1000 * row["n"])


if __name__ == "__main__":
    unittest.main()
