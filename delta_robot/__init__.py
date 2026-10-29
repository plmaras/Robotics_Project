"""3-DOF Delta Robot Simulation and Control Package.

Modules:
  - params: Geometric and physical robot parameters
  - kinematics: Closed-form forward, inverse kinematics and Jacobian
  - dynamics: Analytical mass matrix, Coriolis, gravity, forward and inverse dynamics
  - estimation: Conveyor kinematics, noisy sensor, alpha-beta filter, intercept prediction
  - planning: Quintic polynomial trajectories and 3D RRT obstacle avoidance
  - control: PD and Computed-Torque Controllers with critical damping gain tuning
  - mujoco_env: MuJoCo physics simulation environment
"""

from delta_robot.params import DeltaParams
from delta_robot.kinematics import (
    inverse_kinematics,
    forward_kinematics,
    jacobian,
    inverse_jacobian,
    conditioning,
    home_position,
)
from delta_robot.dynamics import (
    mass_matrix,
    gravity_vector,
    coriolis_vector,
    inverse_dynamics,
    forward_dynamics,
)
from delta_robot.estimation import (
    ConveyorBelt,
    NoisySensor,
    AlphaBetaFilter3D,
    predict_interception,
)
from delta_robot.planning import (
    QuinticTrajectory,
    MultiSegmentTrajectory,
    SphereObstacle,
    RRTPlanner3D,
    build_obstacle_avoidance_trajectory,
)
from delta_robot.control import (
    PDController,
    ComputedTorqueController,
    tune_gains_critical_damping,
)
from delta_robot.mujoco_env import DeltaMuJoCoEnv
