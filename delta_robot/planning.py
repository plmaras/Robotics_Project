"""Trajectory generation and obstacle avoidance planning for 3-DOF Delta robot.

Includes:
  - Quintic polynomial trajectory generation with C^2 continuity (position, velocity, acceleration).
  - Rapidly-exploring Random Tree (RRT) 3D path planner for obstacle avoidance.
  - Multi-segment trajectory generator for following planned waypoints.
"""

from typing import List, Tuple, Optional
import numpy as np

from delta_robot.params import DeltaParams
from delta_robot.kinematics import inverse_kinematics


class QuinticTrajectory:
    """Quintic polynomial trajectory segment in R^3.

    Satisfies boundary conditions:
      p(0) = p0,   v(0) = v0,   a(0) = a0
      p(T) = pf,   v(T) = vf,   a(T) = af
    """

    def __init__(
        self,
        p0: np.ndarray,
        pf: np.ndarray,
        v0: Optional[np.ndarray] = None,
        vf: Optional[np.ndarray] = None,
        a0: Optional[np.ndarray] = None,
        af: Optional[np.ndarray] = None,
        duration: float = 1.0,
    ):
        self.p0 = np.asarray(p0, dtype=np.float64).copy()
        self.pf = np.asarray(pf, dtype=np.float64).copy()
        self.v0 = np.zeros(3, dtype=np.float64) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
        self.vf = np.zeros(3, dtype=np.float64) if vf is None else np.asarray(vf, dtype=np.float64).copy()
        self.a0 = np.zeros(3, dtype=np.float64) if a0 is None else np.asarray(a0, dtype=np.float64).copy()
        self.af = np.zeros(3, dtype=np.float64) if af is None else np.asarray(af, dtype=np.float64).copy()

        self.duration = max(1e-4, float(duration))
        T = self.duration

        # Coefficient matrices for p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        self.c0 = self.p0
        self.c1 = self.v0
        self.c2 = 0.5 * self.a0

        M = np.array([
            [T**3,       T**4,        T**5],
            [3.0 * T**2, 4.0 * T**3,  5.0 * T**4],
            [6.0 * T,    12.0 * T**2, 20.0 * T**3]
        ], dtype=np.float64)

        b = np.vstack([
            self.pf - self.p0 - self.v0 * T - 0.5 * self.a0 * (T**2),
            self.vf - self.v0 - self.a0 * T,
            self.af - self.a0
        ])  # Shape: (3, 3)

        coeffs_345 = np.linalg.solve(M, b)  # Shape: (3, 3)
        self.c3 = coeffs_345[0]
        self.c4 = coeffs_345[1]
        self.c5 = coeffs_345[2]

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample position, velocity, and acceleration at time t."""
        tc = np.clip(t, 0.0, self.duration)
        pos = self.c0 + self.c1 * tc + self.c2 * (tc**2) + self.c3 * (tc**3) + self.c4 * (tc**4) + self.c5 * (tc**5)
        vel = self.c1 + 2.0 * self.c2 * tc + 3.0 * self.c3 * (tc**2) + 4.0 * self.c4 * (tc**3) + 5.0 * self.c5 * (tc**4)
        acc = 2.0 * self.c2 + 6.0 * self.c3 * tc + 12.0 * self.c4 * (tc**2) + 20.0 * self.c5 * (tc**3)

        # Extrapolate constant velocity past duration if sampled beyond T
        if t > self.duration:
            dt = t - self.duration
            pos = pos + self.vf * dt
            vel = self.vf
            acc = np.zeros(3, dtype=np.float64)

        return pos, vel, acc


class MultiSegmentTrajectory:
    """Concatenation of multiple quintic trajectory segments."""

    def __init__(self, segments: List[QuinticTrajectory]):
        self.segments = segments
        self.durations = [seg.duration for seg in segments]
        self.total_duration = sum(self.durations)
        self.cumulative_times = np.cumsum([0.0] + self.durations)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample combined trajectory at time t."""
        if t <= 0.0:
            return self.segments[0].sample(0.0)
        if t >= self.total_duration:
            return self.segments[-1].sample(self.segments[-1].duration + (t - self.total_duration))

        # Find active segment
        idx = np.searchsorted(self.cumulative_times, t, side='right') - 1
        idx = int(np.clip(idx, 0, len(self.segments) - 1))
        local_t = t - self.cumulative_times[idx]
        return self.segments[idx].sample(local_t)


class SphereObstacle:
    """Spherical obstacle in 3D workspace."""

    def __init__(self, center: np.ndarray, radius: float, margin: float = 0.03):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)
        self.margin = float(margin)
        self.safe_radius = self.radius + self.margin

    def is_point_colliding(self, point: np.ndarray) -> bool:
        """Check if 3D point penetrates the obstacle safety volume."""
        return float(np.linalg.norm(np.asarray(point) - self.center)) <= self.safe_radius

    def is_segment_colliding(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if straight line segment between p1 and p2 penetrates the obstacle."""
        v = p2 - p1
        v_len_sq = np.dot(v, v)
        if v_len_sq < 1e-12:
            return self.is_point_colliding(p1)

        t_proj = np.clip(np.dot(self.center - p1, v) / v_len_sq, 0.0, 1.0)
        closest_point = p1 + t_proj * v
        return float(np.linalg.norm(closest_point - self.center)) <= self.safe_radius


class RRTPlanner3D:
    """Basic Rapidly-exploring Random Tree (RRT) path planner for 3D workspace.

    Avoids obstacles and respects robot inverse kinematics workspace boundaries.
    """

    def __init__(
        self,
        params: DeltaParams,
        obstacles: List[SphereObstacle],
        step_size: float = 0.04,
        max_iterations: int = 600,
        goal_bias: float = 0.20,
        bounds_low: np.ndarray = np.array([-0.25, -0.25, -0.72]),
        bounds_high: np.ndarray = np.array([0.25, 0.25, -0.48]),
        seed: Optional[int] = 42,
    ):
        self.params = params
        self.obstacles = obstacles
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.goal_bias = float(goal_bias)
        self.bounds_low = np.asarray(bounds_low, dtype=np.float64)
        self.bounds_high = np.asarray(bounds_high, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def is_state_valid(self, pos: np.ndarray) -> bool:
        """Check whether position is collision-free and kinematically reachable."""
        # Workspace bounds check
        if np.any(pos < self.bounds_low) or np.any(pos > self.bounds_high):
            return False

        # Obstacle collision check
        for obs in self.obstacles:
            if obs.is_point_colliding(pos):
                return False

        # Kinematic reachability check
        _, reachable, _ = inverse_kinematics(pos, self.params, margin=np.radians(2.0))
        return reachable

    def is_segment_valid(self, p1: np.ndarray, p2: np.ndarray, num_checks: int = 10) -> bool:
        """Check whether straight segment between p1 and p2 is collision-free and reachable."""
        for obs in self.obstacles:
            if obs.is_segment_colliding(p1, p2):
                return False

        alphas = np.linspace(0.0, 1.0, num_checks)
        for alpha in alphas:
            pt = (1.0 - alpha) * p1 + alpha * p2
            _, reachable, _ = inverse_kinematics(pt, self.params, margin=np.radians(1.5))
            if not reachable:
                return False
        return True

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """Plan a collision-free path from start to goal."""
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        if not self.is_state_valid(start) or not self.is_state_valid(goal):
            return None

        # Quick check: direct connection
        if self.is_segment_valid(start, goal):
            return [start, goal]

        nodes = [start]
        parents = [-1]

        for _ in range(self.max_iterations):
            if self.rng.random() < self.goal_bias:
                q_rand = goal.copy()
            else:
                q_rand = self.rng.uniform(self.bounds_low, self.bounds_high)

            # Nearest neighbor
            distances = np.linalg.norm(nodes - q_rand, axis=1)
            nearest_idx = int(np.argmin(distances))
            q_near = nodes[nearest_idx]

            # Steer towards q_rand
            diff = q_rand - q_near
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue

            step = min(self.step_size, dist)
            q_new = q_near + (diff / dist) * step

            # Validation
            if self.is_segment_valid(q_near, q_new, num_checks=6):
                nodes.append(q_new)
                parents.append(nearest_idx)

                # Check connection to goal
                if np.linalg.norm(q_new - goal) <= self.step_size:
                    if self.is_segment_valid(q_new, goal, num_checks=6):
                        nodes.append(goal)
                        parents.append(len(nodes) - 2)

                        # Reconstruct raw path
                        raw_path = []
                        curr = len(nodes) - 1
                        while curr != -1:
                            raw_path.append(nodes[curr])
                            curr = parents[curr]
                        raw_path.reverse()
                        return self._shortcut_path(raw_path)

        return None

    def _shortcut_path(self, path: List[np.ndarray], max_trials: int = 50) -> List[np.ndarray]:
        """Greedy shortcutting / pruning to remove unnecessary zig-zags."""
        if len(path) <= 2:
            return path

        short_path = list(path)
        for _ in range(max_trials):
            if len(short_path) <= 2:
                break
            i = self.rng.integers(0, len(short_path) - 2)
            j = self.rng.integers(i + 2, len(short_path))
            if self.is_segment_valid(short_path[i], short_path[j]):
                # Remove intermediate waypoints between i and j
                short_path = short_path[: i + 1] + short_path[j:]

        return short_path


def build_obstacle_avoidance_trajectory(
    waypoints: List[np.ndarray],
    v_final: np.ndarray,
    total_time: float,
) -> MultiSegmentTrajectory:
    """Fit a smooth multi-segment quintic trajectory through RRT waypoints.

    Allocates segment durations proportionally to euclidean distances,
    ensuring continuous position and velocity.
    """
    num_segs = len(waypoints) - 1
    distances = [float(np.linalg.norm(waypoints[i + 1] - waypoints[i])) for i in range(num_segs)]
    total_dist = max(1e-4, sum(distances))

    # Allocate time proportional to path length
    durations = [max(0.10, total_time * (d / total_dist)) for d in distances]
    scale_factor = total_time / sum(durations)
    durations = [d * scale_factor for d in durations]

    # Compute interior waypoint velocities via finite differences
    velocities = [np.zeros(3, dtype=np.float64)]  # Start from rest
    for i in range(1, len(waypoints) - 1):
        v_dir = waypoints[i + 1] - waypoints[i - 1]
        dt = durations[i - 1] + durations[i]
        velocities.append(v_dir / dt)
    velocities.append(np.asarray(v_final, dtype=np.float64))  # Match final target velocity

    segments = []
    for i in range(num_segs):
        seg = QuinticTrajectory(
            p0=waypoints[i],
            pf=waypoints[i + 1],
            v0=velocities[i],
            vf=velocities[i + 1],
            duration=durations[i],
        )
        segments.append(seg)

    return MultiSegmentTrajectory(segments)
