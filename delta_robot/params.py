"""Robot and simulation parameters for the 3-DOF Delta robot.

All dimensions are in SI units (meters, radians, kilograms, seconds, Newtons).
Frame convention: Origin at the center of the base plate, +Z pointing upward.
Legs are spaced at 120-degree intervals (0, 120, 240 deg).
Joint angle convention: theta = 0 rad means the upper arm is horizontal
and points radially outward; positive theta raises the elbow (+Z).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DeltaParams:
    """Geometric and inertial parameters of the 3-DOF Delta robot."""

    # Kinematics geometry [m]
    upper_arm: float = 0.40       # Length of upper arm link (motor axis to elbow)
    lower_arm: float = 0.65       # Length of parallelogram lower rod
    base_radius: float = 0.20     # Radius from base center to motor axes
    effector_radius: float = 0.045  # Radius from effector center to rod ball joints

    # Joint limits [rad]
    theta_min: float = -np.radians(105.0)  # ~ -1.8326 rad (downwards)
    theta_max: float = np.radians(45.0)    # ~ +0.7854 rad (upwards)
    ball_joint_max: float = np.radians(45.0) # Maximum rod out-of-plane tilt

    # Dynamic inertial properties [kg, kg*m^2]
    upper_arm_mass: float = 0.60   # Mass of upper arm link
    lower_arm_mass: float = 0.12   # Mass of lower rod pair
    effector_mass: float = 1.50    # Mass of traveling plate / end-effector
    motor_inertia: float = 0.00015 # Motor rotor + gearbox inertia
    gear_ratio: float = 1.0        # Direct drive in simulation (or gearbox scaling)

    # Environment
    gravity: float = 9.81          # Gravitational acceleration [m/s^2]


# Leg azimuth angles around the Z axis
LEG_ANGLES = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])  # 0, 120, 240 deg


def leg_radial_vector(leg_index: int) -> np.ndarray:
    """Unit vector pointing radially outward in the XY base plane for leg i."""
    phi = LEG_ANGLES[leg_index]
    return np.array([np.cos(phi), np.sin(phi), 0.0], dtype=np.float64)


def leg_tangential_axis(leg_index: int) -> np.ndarray:
    """Motor hinge axis vector for leg i such that positive rotation raises the elbow (+Z)."""
    phi = LEG_ANGLES[leg_index]
    return np.array([np.sin(phi), -np.cos(phi), 0.0], dtype=np.float64)


def base_joint_pos(params: DeltaParams, leg_index: int) -> np.ndarray:
    """Position of the motor axis for leg i in the base frame."""
    return leg_radial_vector(leg_index) * params.base_radius
