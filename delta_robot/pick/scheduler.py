from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .object_pool import ObjectPool, TrackedObject


@dataclass
class Candidate:
    obj: TrackedObject
    intercept: np.ndarray
    travel_time: float
    time_to_exit: float
    score: float


class OptimizedScheduler:
    """Select a feasible object using travel-time and deadline-aware scoring."""

    def __init__(self, pool: ObjectPool, runtime: dict[str, Any]):
        self.pool = pool
        self.cfg = runtime["scheduler"]
        self.controller_cfg = runtime["controller"]

    def choose(self, ee_position: np.ndarray, sim_time: float) -> Candidate | None:
        speed = max(float(self.controller_cfg["max_cartesian_speed_m_s"]), 1e-3)
        belt_speed = max(abs(self.pool.speed), 1e-6)
        max_radius = float(self.cfg["max_pick_radius_m"])
        safety = float(self.cfg["safety_time_s"])
        best: Candidate | None = None
        for obj in self.pool.active_objects():
            # Predict the moving object's position at the estimated arrival time.
            current = np.array([obj.x, obj.y, self.pool.belt_top + obj.height + float(self.controller_cfg["pick_clearance_m"])])
            travel = float(np.linalg.norm(current - ee_position)) / speed
            horizon = max(float(self.cfg["prediction_horizon_s"]), travel)
            intercept_x = obj.x + self.pool.speed * horizon
            intercept = np.array([intercept_x, obj.y, current[2]])
            if intercept_x >= self.pool.exit_x:
                continue
            radial = float(np.linalg.norm(intercept[:2] - np.array([self.pool.conv_x, self.pool.conv_y])))
            if radial > max_radius:
                continue
            time_to_exit = (self.pool.exit_x - obj.x) / belt_speed
            # The robot must have enough time to approach, descend and close.
            required = travel + 0.25 + safety
            if required >= time_to_exit:
                continue
            # Prefer short cycles, but pull urgent objects forward in the queue.
            slack = max(0.0, time_to_exit - required)
            score = travel + 0.18 * slack + 0.04 * abs(obj.y - self.pool.conv_y)
            candidate = Candidate(obj, intercept, travel, time_to_exit, score)
            if best is None or candidate.score < best.score:
                best = candidate
        return best
