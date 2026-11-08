import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delta_pick.scheduler import OptimizedScheduler
from delta_pick.object_pool import TrackedObject


class FakePool:
    speed = 0.1
    belt_top = 0.0
    conv_x = 0.0
    conv_y = 0.0
    exit_x = 0.8
    def __init__(self):
        self.items = [
            TrackedObject(0, "a", 0, 0.30, 0.0, 0.02, 0.02, 0.04, 0.0),
            TrackedObject(1, "b", 1, 0.55, 0.0, 0.02, 0.02, 0.04, 0.0),
        ]
    def active_objects(self):
        return self.items


class SchedulerTest(unittest.TestCase):
    def test_returns_feasible_candidate(self):
        runtime = {
            "scheduler": {"max_pick_radius_m": 0.7, "safety_time_s": 0.1, "prediction_horizon_s": 0.1},
            "controller": {"max_cartesian_speed_m_s": 1.0, "pick_clearance_m": 0.005}
        }
        scheduler = OptimizedScheduler(FakePool(), runtime)
        result = scheduler.choose(np.array([0.0, 0.0, 0.2]), 0.0)
        self.assertIsNotNone(result)
        self.assertIn(result.obj.name, {"a", "b"})
        self.assertGreater(result.time_to_exit, result.travel_time)


if __name__ == "__main__":
    unittest.main()
