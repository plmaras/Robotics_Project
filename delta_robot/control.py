"""Controllers and gain tuning for the 3-DOF Delta robot.

Implements:
  1. Standard Proportional-Derivative (PD) Controller
  2. Computed-Torque Controller (CTC) with dynamic feedforward
  3. Formally justified gain tuning procedure based on second-order error dynamics.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

from delta_robot.params import DeltaParams
from delta_robot.dynamics import mass_matrix, coriolis_vector, gravity_vector


@dataclass
class ControllerGains:
    """Feedback gains for joint space controllers."""
    Kp: np.ndarray  # Proportional gains (3,)
    Kd: np.ndarray  # Derivative gains (3,)
    omega_n: float  # Undamped natural frequency [rad/s]
    zeta: float     # Damping ratio


def tune_gains_critical_damping(
    omega_n: float = 25.0,
    zeta: float = 1.0,
    nominal_inertia: float = 0.18,
) -> Tuple[ControllerGains, ControllerGains]:
    """Gain Tuning Procedure Justification:

    For Computed-Torque Control (CTC), the nonlinear dynamics are cancelled by feedforward:
        tau = M(theta) * a* + C(theta, thetadot)*thetadot + G(theta)
    where:
        a* = thetaddot_d + Kv * e_dot + Kp * e
        e = theta_d - theta,  e_dot = thetadot_d - thetadot

    Substituting tau into the robot equations of motion M * thetaddot + C + G = tau yields:
        M(theta) * (thetaddot - a*) = 0
    Since M(theta) is strictly positive definite, the closed-loop error dynamics decouple into:
        e_ddot + Kv * e_dot + Kp * e = 0

    Comparing to the canonical 2nd-order characteristic polynomial s^2 + 2*zeta*omega_n*s + omega_n^2 = 0:
        Kp_ctc = omega_n^2 * I_3
        Kv_ctc = 2 * zeta * omega_n * I_3

    Justification of parameter choices:
      - zeta = 1.0 (Critical Damping): Completely eliminates overshoot (e(t) <= 0),
        preventing the end-effector from bouncing or colliding with the conveyor.
      - omega_n = 25 rad/s: Yields a settling time t_s = 4 / (zeta * omega_n) = 0.16 s,
        which is well within the 0.5 - 1.0 s conveyor intercept window and within actuator limits.

    For uncompensated PD control, the controller operates in torque units without inertia normalization:
        tau = Kp_pd * e + Kd_pd * e_dot + G(theta)
    The equivalent linear gains scale with the average effective joint inertia bar_M:
        Kp_pd = bar_M * omega_n^2
        Kd_pd = bar_M * (2 * zeta * omega_n)
    """
    # CTC operates in acceleration units [rad/s^2 per error]
    kp_ctc_val = omega_n**2
    kv_ctc_val = 2.0 * zeta * omega_n
    ctc_gains = ControllerGains(
        Kp=np.full(3, kp_ctc_val, dtype=np.float64),
        Kd=np.full(3, kv_ctc_val, dtype=np.float64),
        omega_n=omega_n,
        zeta=zeta,
    )

    # PD operates in torque units [Nm per error]
    kp_pd_val = nominal_inertia * kp_ctc_val
    kd_pd_val = nominal_inertia * kv_ctc_val
    pd_gains = ControllerGains(
        Kp=np.full(3, kp_pd_val, dtype=np.float64),
        Kd=np.full(3, kd_pd_val, dtype=np.float64),
        omega_n=omega_n,
        zeta=zeta,
    )

    return ctc_gains, pd_gains


class PDController:
    """Standard Joint-Space Proportional-Derivative (PD) Controller.

    tau = Kp * (theta_d - theta) + Kd * (thetadot_d - thetadot) + G(theta)
    """

    def __init__(self, params: DeltaParams, gains: ControllerGains, with_gravity_comp: bool = True):
        self.params = params
        self.Kp = np.diag(gains.Kp)
        self.Kd = np.diag(gains.Kd)
        self.with_gravity_comp = with_gravity_comp

    def compute_torque(
        self,
        theta_d: np.ndarray,
        thetadot_d: np.ndarray,
        theta: np.ndarray,
        thetadot: np.ndarray,
        payload: float = 0.0,
    ) -> np.ndarray:
        """Compute motor joint torques [Nm]."""
        e = theta_d - theta
        e_dot = thetadot_d - thetadot

        tau_feedback = (self.Kp @ e) + (self.Kd @ e_dot)

        if self.with_gravity_comp:
            G = gravity_vector(theta, self.params, payload=payload)
            return tau_feedback + G
        return tau_feedback


class ComputedTorqueController:
    """Computed-Torque Controller (CTC) with dynamic inverse dynamics feedforward.

    a* = thetaddot_d + Kv * (thetadot_d - thetadot) + Kp * (theta_d - theta)
    tau = M(theta) * a* + C(theta, thetadot) * thetadot + G(theta)
    """

    def __init__(self, params: DeltaParams, gains: ControllerGains):
        self.params = params
        self.Kp = np.diag(gains.Kp)
        self.Kv = np.diag(gains.Kd)

    def compute_torque(
        self,
        theta_d: np.ndarray,
        thetadot_d: np.ndarray,
        thetaddot_d: np.ndarray,
        theta: np.ndarray,
        thetadot: np.ndarray,
        payload: float = 0.0,
    ) -> np.ndarray:
        """Compute motor joint torques with dynamic linearization [Nm]."""
        e = theta_d - theta
        e_dot = thetadot_d - thetadot

        # Desired virtual joint acceleration
        a_star = thetaddot_d + (self.Kv @ e_dot) + (self.Kp @ e)

        # Dynamic model terms
        M = mass_matrix(theta, self.params, payload=payload)
        C_term = coriolis_vector(theta, thetadot, self.params, payload=payload)
        G = gravity_vector(theta, self.params, payload=payload)

        # Inverse dynamics feedforward + feedback
        tau = (M @ a_star) + C_term + G
        return tau
