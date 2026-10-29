"""Target state estimation with Alpha-Beta filter and interception prediction.

Simulates a moving object on a kinematically prescribed conveyor belt with sensor noise,
filters position readings to estimate 3D velocity and position, and predicts
the optimal interception point inside the Delta robot's workspace.
"""

from typing import Tuple, Optional
import numpy as np

from delta_robot.params import DeltaParams
from delta_robot.kinematics import inverse_kinematics


class ConveyorBelt:
    """Kinematically prescribed conveyor belt motion."""

    def __init__(
        self,
        velocity: float = 0.20,
        start_x: float = -0.40,
        y_pos: float = 0.0,
        z_pos: float = -0.65,
    ):
        self.velocity = float(velocity)
        self.start_x = float(start_x)
        self.y_pos = float(y_pos)
        self.z_pos = float(z_pos)

    def true_position(self, t: float) -> np.ndarray:
        """True 3D position of the target object at simulation time t [s]."""
        x = self.start_x + self.velocity * t
        return np.array([x, self.y_pos, self.z_pos], dtype=np.float64)

    def true_velocity(self) -> np.ndarray:
        """True 3D velocity vector of the target object [m/s]."""
        return np.array([self.velocity, 0.0, 0.0], dtype=np.float64)


class NoisySensor:
    """Sensor that corrupts true target position with Gaussian noise."""

    def __init__(self, std_dev: float = 0.002, seed: Optional[int] = 42):
        self.std_dev = float(std_dev)
        self.rng = np.random.default_rng(seed)

    def measure(self, true_pos: np.ndarray) -> np.ndarray:
        """Return noisy 3D position measurement."""
        noise = self.rng.normal(0.0, self.std_dev, size=3)
        return np.asarray(true_pos, dtype=np.float64) + noise


class AlphaBetaFilter3D:
    """3D Alpha-Beta tracking filter for position and velocity estimation.

    Equations:
      Predict:
        p_pred = p_est + v_est * dt
        v_pred = v_est
      Residual:
        res = z_meas - p_pred
      Update:
        p_est = p_pred + alpha * res
        v_est = v_pred + (beta / dt) * res
    """

    def __init__(self, alpha: float = 0.60, beta: float = 0.25, dt: float = 0.01):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dt = float(dt)
        self.p_est: Optional[np.ndarray] = None
        self.v_est: Optional[np.ndarray] = None
        self.initialized = False

    def reset(self, initial_pos: Optional[np.ndarray] = None, initial_vel: Optional[np.ndarray] = None):
        """Reset the filter state."""
        if initial_pos is not None:
            self.p_est = np.asarray(initial_pos, dtype=np.float64).copy()
            self.v_est = (
                np.asarray(initial_vel, dtype=np.float64).copy()
                if initial_vel is not None
                else np.zeros(3, dtype=np.float64)
            )
            self.initialized = True
        else:
            self.p_est = None
            self.v_est = None
            self.initialized = False

    def update(self, measurement: np.ndarray, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with new noisy measurement."""
        meas = np.asarray(measurement, dtype=np.float64)
        step_dt = dt if dt is not None else self.dt

        if not self.initialized:
            self.p_est = meas.copy()
            self.v_est = np.zeros(3, dtype=np.float64)
            self.initialized = True
            return self.p_est, self.v_est

        # Prediction step
        p_pred = self.p_est + self.v_est * step_dt
        v_pred = self.v_est

        # Innovation / residual
        residual = meas - p_pred

        # Correction step
        self.p_est = p_pred + self.alpha * residual
        self.v_est = v_pred + (self.beta / step_dt) * residual

        return self.p_est, self.v_est


def predict_interception(
    robot_current_pos: np.ndarray,
    target_est_pos: np.ndarray,
    target_est_vel: np.ndarray,
    params: DeltaParams,
    t_start: float = 0.0,
    v_avg_robot: float = 0.80,
    min_travel_time: float = 0.40,
    max_search_time: float = 3.00,
    time_step: float = 0.02,
    constrain_plane_z: Optional[float] = None,
    constrain_plane_y: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], float]:
    """Predict interception location and time window inside robot workspace.

    Scans candidate future times T >= min_travel_time to find the earliest time
    where:
      1. Target position p(T) is within the robot's kinematic reachable envelope.
      2. The robot can physically travel to p(T) within time T.

    Returns:
      (intercept_pos, intercept_time): 3D intercept point and absolute time t_intercept.
    """
    robot_current_pos = np.asarray(robot_current_pos, dtype=np.float64)
    target_est_pos = np.asarray(target_est_pos, dtype=np.float64).copy()
    target_est_vel = np.asarray(target_est_vel, dtype=np.float64).copy()

    if constrain_plane_y is not None:
        target_est_pos[1] = constrain_plane_y
        target_est_vel[1] = 0.0
    if constrain_plane_z is not None:
        target_est_pos[2] = constrain_plane_z
        target_est_vel[2] = 0.0

    t_candidates = np.arange(min_travel_time, max_search_time, time_step)

    for T in t_candidates:
        future_target_pos = target_est_pos + target_est_vel * T

        # Check reachability via Inverse Kinematics
        _, reachable, _ = inverse_kinematics(future_target_pos, params, margin=np.radians(2.0))
        if not reachable:
            continue

        # Check feasibility: travel distance vs time
        dist = np.linalg.norm(future_target_pos - robot_current_pos)
        required_time = dist / v_avg_robot

        if T >= required_time:
            return future_target_pos, t_start + T

    # Fallback to earliest reachable point if search exhausted
    for T in t_candidates:
        future_target_pos = target_est_pos + target_est_vel * T
        _, reachable, _ = inverse_kinematics(future_target_pos, params)
        if reachable:
            return future_target_pos, t_start + T

    return None, 0.0
