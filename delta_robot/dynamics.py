"""Rigid-body and Lumped-Mass Dynamics for 3-DOF Delta Robot.

Provides analytical closed-form:
  - Mass matrix M(theta)
  - Coriolis and centrifugal torques C(theta, thetadot) * thetadot
  - Gravitational torques G(theta)
  - Inverse Dynamics: tau = M * thetaddot + C * thetadot + G
  - Forward Dynamics: thetaddot = M^-1 * (tau - C * thetadot - G)
"""

from typing import Tuple, Dict, Any
import numpy as np

from delta_robot.params import DeltaParams
from delta_robot.kinematics import forward_kinematics, jacobian


def mass_matrix(
    theta: np.ndarray,
    params: DeltaParams,
    pos: np.ndarray = None,
    payload: float = 0.0,
) -> np.ndarray:
    """Analytical joint-space inertia / mass matrix M(theta) in R^{3x3}.

    M(theta) = I_arm * I_3 + J^T * (m_eff_total * I_3) * J
    """
    theta = np.asarray(theta, dtype=np.float64)
    if pos is None:
        pos = forward_kinematics(theta, params)
        if pos is None:
            raise ValueError(f"Cannot compute forward kinematics for theta={theta}")

    J = jacobian(theta, pos, params)

    # Lumped lower rod mass: half on elbow, half on effector
    m_elbow = params.lower_arm_mass / 2.0
    # Arm rotational inertia about the motor axis
    I_arm = (
        (1.0 / 3.0) * params.upper_arm_mass * params.upper_arm**2
        + m_elbow * params.upper_arm**2
        + params.motor_inertia * (params.gear_ratio**2)
    )

    # Total effective mass at effector
    m_eff_tot = params.effector_mass + payload + 3.0 * (params.lower_arm_mass / 2.0)
    M_task = m_eff_tot * np.eye(3, dtype=np.float64)

    # Joint space inertia
    M_joint = I_arm * np.eye(3, dtype=np.float64) + J.T @ M_task @ J
    return M_joint


def gravity_vector(
    theta: np.ndarray,
    params: DeltaParams,
    pos: np.ndarray = None,
    payload: float = 0.0,
) -> np.ndarray:
    """Joint-space gravitational torque vector G(theta) in R^3.

    G(theta) = G_arm(theta) + J^T * F_gravity_effector
    """
    theta = np.asarray(theta, dtype=np.float64)
    if pos is None:
        pos = forward_kinematics(theta, params)
        if pos is None:
            raise ValueError(f"Cannot compute forward kinematics for theta={theta}")

    J = jacobian(theta, pos, params)

    # Gravity on upper arm + lumped elbow mass
    m_elbow = params.lower_arm_mass / 2.0
    G_arm = (
        (0.5 * params.upper_arm_mass + m_elbow)
        * params.upper_arm
        * params.gravity
        * np.cos(theta)
    )

    # Gravity force on the effector (acting downward along -Z)
    m_eff_tot = params.effector_mass + payload + 3.0 * (params.lower_arm_mass / 2.0)
    # Required holding force in Cartesian space: upward (+Z)
    F_eff_grav = np.array([0.0, 0.0, m_eff_tot * params.gravity], dtype=np.float64)

    # Projected into joint space via Jacobian transpose
    G_joint = G_arm + (J.T @ F_eff_grav)
    return G_joint


def coriolis_vector(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    params: DeltaParams,
    pos: np.ndarray = None,
    payload: float = 0.0,
) -> np.ndarray:
    """Coriolis and centrifugal joint torques C(theta, thetadot) * thetadot in R^3.

    Calculated via: J^T * M_task * J_dot * thetadot
    """
    theta = np.asarray(theta, dtype=np.float64)
    theta_dot = np.asarray(theta_dot, dtype=np.float64)
    if pos is None:
        pos = forward_kinematics(theta, params)
        if pos is None:
            raise ValueError(f"Cannot compute forward kinematics for theta={theta}")

    J = jacobian(theta, pos, params)

    # Finite difference approximation of J_dot * thetadot along the velocity direction
    eps = 1e-6
    theta_fwd = theta + theta_dot * eps
    theta_bwd = theta - theta_dot * eps
    pos_fwd = forward_kinematics(theta_fwd, params)
    pos_bwd = forward_kinematics(theta_bwd, params)

    if pos_fwd is None or pos_bwd is None:
        # Near singularity or workspace edge, fallback to 0
        return np.zeros(3, dtype=np.float64)

    J_fwd = jacobian(theta_fwd, pos_fwd, params)
    J_bwd = jacobian(theta_bwd, pos_bwd, params)
    J_dot = (J_fwd - J_bwd) / (2.0 * eps)

    # Effector convective acceleration: a_conv = J_dot * thetadot
    a_conv = J_dot @ theta_dot

    m_eff_tot = params.effector_mass + payload + 3.0 * (params.lower_arm_mass / 2.0)
    tau_coriolis = J.T @ (m_eff_tot * a_conv)
    return tau_coriolis


def inverse_dynamics(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    theta_ddot: np.ndarray,
    params: DeltaParams,
    payload: float = 0.0,
) -> np.ndarray:
    """Inverse Dynamics: computes required joint torques tau [Nm].

    tau = M(theta) * theta_ddot + C(theta, theta_dot) * theta_dot + G(theta)
    """
    pos = forward_kinematics(theta, params)
    M = mass_matrix(theta, params, pos=pos, payload=payload)
    C_term = coriolis_vector(theta, theta_dot, params, pos=pos, payload=payload)
    G = gravity_vector(theta, params, pos=pos, payload=payload)
    return (M @ theta_ddot) + C_term + G


def forward_dynamics(
    theta: np.ndarray,
    theta_dot: np.ndarray,
    tau: np.ndarray,
    params: DeltaParams,
    payload: float = 0.0,
) -> np.ndarray:
    """Forward Dynamics: computes resulting joint accelerations theta_ddot [rad/s^2].

    theta_ddot = M(theta)^-1 * (tau - C(theta, theta_dot) * theta_dot - G(theta))
    """
    pos = forward_kinematics(theta, params)
    M = mass_matrix(theta, params, pos=pos, payload=payload)
    C_term = coriolis_vector(theta, theta_dot, params, pos=pos, payload=payload)
    G = gravity_vector(theta, params, pos=pos, payload=payload)
    return np.linalg.solve(M, tau - C_term - G)
