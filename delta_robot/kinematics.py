"""Closed-form Forward and Inverse Kinematics and Jacobian for 3-DOF Delta Robot.

All computations are implemented in NumPy with analytical closed-form equations.
"""

from typing import Tuple, Optional
import numpy as np

from delta_robot.params import (
    DeltaParams,
    LEG_ANGLES,
    leg_radial_vector,
    leg_tangential_axis,
    base_joint_pos,
)


def elbow_pos(params: DeltaParams, leg_index: int, theta: float) -> np.ndarray:
    """Position of the elbow joint for leg i given joint angle theta [rad]."""
    r = leg_radial_vector(leg_index)
    rad_dist = params.base_radius + params.upper_arm * np.cos(theta)
    return np.array([
        r[0] * rad_dist,
        r[1] * rad_dist,
        params.upper_arm * np.sin(theta)
    ], dtype=np.float64)


def elbow_derivative(params: DeltaParams, leg_index: int, theta: float) -> np.ndarray:
    """Derivative of the elbow position with respect to joint angle theta (dE/dtheta)."""
    r = leg_radial_vector(leg_index)
    return np.array([
        -r[0] * params.upper_arm * np.sin(theta),
        -r[1] * params.upper_arm * np.sin(theta),
        params.upper_arm * np.cos(theta)
    ], dtype=np.float64)


def effector_joint_pos(params: DeltaParams, leg_index: int, pos: np.ndarray) -> np.ndarray:
    """Lower arm attachment point on the effector for leg i given effector position."""
    return np.asarray(pos, dtype=np.float64) + leg_radial_vector(leg_index) * params.effector_radius


def lower_arm_tilt(params: DeltaParams, leg_index: int, theta: float, pos: np.ndarray) -> float:
    """Out-of-plane angle of the lower arm rod (for ball joint angle limits)."""
    rod = effector_joint_pos(params, leg_index, pos) - elbow_pos(params, leg_index, theta)
    axis = leg_tangential_axis(leg_index)
    sin_tilt = np.clip(np.abs(np.dot(rod, axis)) / params.lower_arm, -1.0, 1.0)
    return float(np.arcsin(sin_tilt))


def inverse_kinematics(
    pos: np.ndarray, params: DeltaParams, margin: float = 0.0
) -> Tuple[Optional[np.ndarray], bool, str]:
    """Analytical closed-form Inverse Kinematics: effector position -> joint angles.

    Parameters:
        pos: 3D coordinates [x, y, z] of the end-effector.
        params: DeltaParams geometry specification.
        margin: Optional angular safety margin [rad].

    Returns:
        (thetas, success, reason): thetas is array of 3 angles [rad] if successful.
    """
    pos = np.asarray(pos, dtype=np.float64)
    thetas = []

    for i in range(3):
        phi = LEG_ANGLES[i]
        # Rotate target into leg frame (where leg lies along +x axis)
        c, s = np.cos(-phi), np.sin(-phi)
        a = c * pos[0] - s * pos[1] + params.effector_radius - params.base_radius
        b = s * pos[0] + c * pos[1]
        z = pos[2]

        # Parallelogram constraint: (a - U cos(theta))^2 + b^2 + (z - U sin(theta))^2 = L^2
        # <=> A cos(theta) + B sin(theta) = C
        A = 2.0 * a * params.upper_arm
        B = 2.0 * z * params.upper_arm
        C = (a * a + b * b + z * z + params.upper_arm * params.upper_arm - params.lower_arm * params.lower_arm)

        R = np.hypot(A, B)
        if R < 1e-12 or abs(C / R) > (1.0 + 1e-9):
            return None, False, f"unreachable_geometry_leg_{i}"

        ratio = np.clip(C / R, -1.0, 1.0)
        # "+" branch is physical elbow-out configuration
        theta = np.arctan2(B, A) + np.arccos(ratio)
        # Normalize to (-pi, pi]
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

        # Joint limit checks
        if theta < (params.theta_min + margin) or theta > (params.theta_max - margin):
            return None, False, f"joint_limit_violation_leg_{i}"

        # Ball joint tilt check
        tilt = lower_arm_tilt(params, i, theta, pos)
        if tilt > (params.ball_joint_max - margin):
            return None, False, f"ball_joint_limit_leg_{i}"

        thetas.append(theta)

    return np.array(thetas, dtype=np.float64), True, "ok"


def forward_kinematics(thetas: np.ndarray, params: DeltaParams) -> Optional[np.ndarray]:
    """Analytical closed-form Forward Kinematics: joint angles -> effector position.

    Intersects three spheres of radius L centered at the elbow joints
    (offset by the effector radius) and returns the lower physical solution (z < 0).
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    centers = [
        elbow_pos(params, i, thetas[i]) - leg_radial_vector(i) * params.effector_radius
        for i in range(3)
    ]
    c0, c1, c2 = centers[0], centers[1], centers[2]

    # Two plane equations from sphere subtraction: 2 * (c_i - c_0) . p = ||c_i||^2 - ||c_0||^2
    n1 = c1 - c0
    n2 = c2 - c0
    d1 = 0.5 * (np.dot(c1, c1) - np.dot(c0, c0))
    d2 = 0.5 * (np.dot(c2, c2) - np.dot(c0, c0))

    # Line of intersection
    u = np.cross(n1, n2)
    uu = np.dot(u, u)
    if uu < 1e-14:
        return None  # Planes are parallel / degenerate

    p0 = (np.cross(n2, u) * d1 + np.cross(u, n1) * d2) / uu

    # Substitute line p(t) = p0 + t * u into sphere 0: ||p0 + t * u - c0||^2 = L^2
    w = p0 - c0
    qa = uu
    qb = 2.0 * np.dot(u, w)
    qc = np.dot(w, w) - params.lower_arm * params.lower_arm
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0:
        return None  # Spheres do not intersect

    sq = np.sqrt(max(0.0, disc))
    t1 = (-qb + sq) / (2.0 * qa)
    t2 = (-qb - sq) / (2.0 * qa)
    s1 = p0 + u * t1
    s2 = p0 + u * t2

    # Lower solution is the physical one (robot works below base)
    return s1 if s1[2] < s2[2] else s2


def jacobian(thetas: np.ndarray, pos: np.ndarray, params: DeltaParams) -> np.ndarray:
    """Analytical Jacobian matrix J(theta) such that v_ee = J * thetadot and tau = J^T * F.

    Matrix equation: J_x * v_ee = J_theta * thetadot  =>  J = inv(J_x) * J_theta.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)

    Jx = np.zeros((3, 3), dtype=np.float64)
    Jtheta = np.zeros((3, 3), dtype=np.float64)

    for i in range(3):
        n_i = effector_joint_pos(params, i, pos) - elbow_pos(params, i, thetas[i])
        Ei_prime = elbow_derivative(params, i, thetas[i])
        Jx[i, :] = n_i
        Jtheta[i, i] = np.dot(n_i, Ei_prime)

    return np.linalg.inv(Jx) @ Jtheta


def inverse_jacobian(thetas: np.ndarray, pos: np.ndarray, params: DeltaParams) -> np.ndarray:
    """Inverse Jacobian matrix J_inv such that thetadot = J_inv * v_ee."""
    thetas = np.asarray(thetas, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)

    Jx = np.zeros((3, 3), dtype=np.float64)
    Jtheta_inv = np.zeros((3, 3), dtype=np.float64)

    for i in range(3):
        n_i = effector_joint_pos(params, i, pos) - elbow_pos(params, i, thetas[i])
        Ei_prime = elbow_derivative(params, i, thetas[i])
        den = np.dot(n_i, Ei_prime)
        Jx[i, :] = n_i
        Jtheta_inv[i, i] = 1.0 / den if abs(den) > 1e-12 else 0.0

    return Jtheta_inv @ Jx


def conditioning(thetas: np.ndarray, pos: np.ndarray, params: DeltaParams) -> float:
    """Conditioning measure in [0, 1]: 1 = well conditioned, near 0 = close to singularity."""
    rods = []
    inv_sing = 1.0
    for i in range(3):
        n = effector_joint_pos(params, i, pos) - elbow_pos(params, i, thetas[i])
        rod_unit = n / params.lower_arm
        rods.append(rod_unit)
        tangent = elbow_derivative(params, i, thetas[i]) / params.upper_arm
        inv_sing = min(inv_sing, float(abs(np.dot(rod_unit, tangent))))

    # Forward singularity: rods coplanar (cross product of rods)
    fwd_sing = float(abs(np.dot(rods[0], np.cross(rods[1], rods[2]))))
    return float(np.clip(min(inv_sing, fwd_sing), 0.0, 1.0))


def home_position(params: DeltaParams) -> Tuple[np.ndarray, np.ndarray]:
    """Nominal resting home position: arms at -30 deg (-pi/6).

    Returns:
        (pos, thetas): Cartesian 3D position and joint angles array.
    """
    thetas = np.array([-np.pi / 6.0, -np.pi / 6.0, -np.pi / 6.0], dtype=np.float64)
    pos = forward_kinematics(thetas, params)
    if pos is None:
        raise RuntimeError("Failed to compute forward kinematics for home position.")
    return pos, thetas
