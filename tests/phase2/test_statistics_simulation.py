import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "simulate_design",
    Path(__file__).parents[2] / "research_grasping/statistics/simulate_design.py",
)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


class SimulationTest(unittest.TestCase):
    def test_wilson(self):
        lo, hi = sim.wilson(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertAlmostEqual((lo + hi) / 2, 0.5)
        self.assertAlmostEqual(sim.wilson(0, 100)[0], 0)

    def test_reproducibility_and_cost(self):
        first = sim.run(2, 37)
        self.assertEqual(first, sim.run(2, 37))
        self.assertNotEqual(first, sim.run(2, 38))
        for row in first["rows"]:
            self.assertLessEqual(row["usable"], row["attempted"])
            self.assertEqual(
                row["nominal_evaluations"],
                100 * row["histories"] * row["objects_per_history"] * row["seeds"],
            )


if __name__ == "__main__":
    unittest.main()
