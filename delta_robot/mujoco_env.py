"""MuJoCo Simulation Environment for the 3-DOF Delta Robot.

Manages physics, kinematically prescribed conveyor motion, moving target object,
obstacle collision checking, and visualization.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import mujoco

from delta_robot.params import DeltaParams, LEG_ANGLES
from delta_robot.kinematics import forward_kinematics


def generate_delta_xml(
    params: DeltaParams,
    with_obstacle: bool = False,
    obstacle_pos: Tuple[float, float, float] = (0.05, 0.0, -0.58),
    obstacle_radius: float = 0.04,
    timestep: float = 0.002,
) -> str:
    """Generate dynamic MJCF XML string for Delta robot environment."""
    rb = params.base_radius
    re = params.effector_radius
    L1 = params.upper_arm
    L2 = params.lower_arm

    # Rest position angles: theta = -pi/6
    theta_home = -np.pi / 6.0
    dr = (rb + L1 * np.cos(theta_home) - re)
    dz = np.sqrt(max(1e-6, L2**2 - dr**2))
    z_ee = L1 * np.sin(theta_home) - dz
    home_p = np.array([0.0, 0.0, z_ee], dtype=np.float64)

    arms_xml = ""
    eq_xml = ""
    exclude_xml = ""

    for i in range(3):
        phi = LEG_ANGLES[i]
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        bx, by = rb * c_phi, rb * s_phi
        # Tangential axis such that positive joint angle raises the elbow (+Z)
        ax, ay = np.sin(phi), -np.cos(phi)

        # Rotation of upper arm by theta_home around motor axis
        axis = np.array([ax, ay, 0.0], dtype=np.float64)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)
        R_arm = np.eye(3) + np.sin(theta_home) * K + (1.0 - np.cos(theta_home)) * (K @ K)

        E_world = np.array([
            (rb + L1 * np.cos(theta_home)) * c_phi,
            (rb + L1 * np.cos(theta_home)) * s_phi,
            L1 * np.sin(theta_home)
        ], dtype=np.float64)
        P_world = np.array([re * c_phi, re * s_phi, z_ee], dtype=np.float64)
        V_world = P_world - E_world
        # Vector from elbow to effector anchor in upper arm's local coordinate frame
        V_local = R_arm.T @ V_world

        ex_local, ey_local = L1 * c_phi, L1 * s_phi

        arms_xml += f"""
        <!-- Leg {i} -->
        <body name="upper_arm_{i}" pos="{bx:.6f} {by:.6f} 0">
          <joint name="joint_{i}" type="hinge" axis="{ax:.6f} {ay:.6f} 0" 
                 range="{params.theta_min:.4f} {params.theta_max:.4f}" damping="0.04"/>
          <geom name="upper_arm_geom_{i}" type="capsule" fromto="0 0 0 {ex_local:.6f} {ey_local:.6f} 0" 
                size="0.015" mass="{params.upper_arm_mass:.4f}" rgba="0.75 0.25 0.25 1"/>
          
          <body name="forearm_{i}" pos="{ex_local:.6f} {ey_local:.6f} 0">
            <joint name="ball_{i}" type="ball" damping="0.005"/>
            <geom name="forearm_geom_{i}" type="capsule" fromto="0 0 0 {V_local[0]:.6f} {V_local[1]:.6f} {V_local[2]:.6f}" 
                  size="0.008" mass="{params.lower_arm_mass:.4f}" rgba="0.25 0.45 0.85 1"/>
            <site name="forearm_tip_{i}" pos="{V_local[0]:.6f} {V_local[1]:.6f} {V_local[2]:.6f}"/>
          </body>
        </body>
        """

        eq_xml += f"""
        <connect site1="forearm_tip_{i}" site2="ee_anchor_{i}"/>
        """
        exclude_xml += f"""
        <exclude body1="base" body2="upper_arm_{i}"/>
        <exclude body1="upper_arm_{i}" body2="forearm_{i}"/>
        <exclude body1="forearm_{i}" body2="end_effector"/>
        """

    exclude_xml += """
    <exclude body1="end_effector" body2="target_object"/>
    <exclude body1="end_effector" body2="conveyor"/>
    """

    obstacle_xml = ""
    if with_obstacle:
        ox, oy, oz = obstacle_pos
        obstacle_xml = f"""
        <!-- Obstacle -->
        <body name="obstacle" pos="{ox:.4f} {oy:.4f} {oz:.4f}">
          <geom name="obstacle_geom" type="sphere" size="{obstacle_radius:.4f}" 
                rgba="0.95 0.5 0.1 0.85" conaffinity="1" contype="1"/>
        </body>
        """

    xml = f"""
<mujoco model="delta_conveyor_system">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -{params.gravity:.4f}" timestep="{timestep:.4f}">
    <flag equality="enable"/>
  </option>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
  </visual>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" castshadow="false"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" pos="0 0 -1.0" rgba="0.92 0.92 0.92 1"/>

    <!-- Base Plate -->
    <body name="base" pos="0 0 0">
      <geom name="base_geom" type="cylinder" size="{rb * 1.15:.4f} 0.015" rgba="0.3 0.32 0.35 1"/>
    </body>

    {arms_xml}

    <!-- Traveling End-Effector Plate -->
    <body name="end_effector" pos="0 0 {z_ee:.6f}">
      <joint name="ee_x" type="slide" axis="1 0 0" damping="0.01"/>
      <joint name="ee_y" type="slide" axis="0 1 0" damping="0.01"/>
      <joint name="ee_z" type="slide" axis="0 0 1" damping="0.01"/>
      <geom name="ee_geom" type="cylinder" size="{re:.4f} 0.01" mass="{params.effector_mass:.4f}" rgba="0.2 0.8 0.3 1"/>
      <site name="ee_anchor_0" pos="{re * np.cos(LEG_ANGLES[0]):.6f} {re * np.sin(LEG_ANGLES[0]):.6f} 0"/>
      <site name="ee_anchor_1" pos="{re * np.cos(LEG_ANGLES[1]):.6f} {re * np.sin(LEG_ANGLES[1]):.6f} 0"/>
      <site name="ee_anchor_2" pos="{re * np.cos(LEG_ANGLES[2]):.6f} {re * np.sin(LEG_ANGLES[2]):.6f} 0"/>
      <site name="tcp" pos="0 0 0" size="0.008" rgba="1 1 0 1"/>
    </body>

    <!-- Conveyor Belt -->
    <body name="conveyor" pos="0 0 -0.68">
      <geom name="conveyor_geom" type="box" size="0.55 0.12 0.015" rgba="0.2 0.22 0.25 1"/>
      <geom name="conveyor_rail_left" type="box" size="0.55 0.01 0.025" pos="0 -0.12 0.01" rgba="0.4 0.4 0.4 1"/>
      <geom name="conveyor_rail_right" type="box" size="0.55 0.01 0.025" pos="0 0.12 0.01" rgba="0.4 0.4 0.4 1"/>
    </body>

    <!-- Moving Target Object (Kinematic Mocap Body) -->
    <body name="target_object" mocap="true" pos="-0.4 0 -0.65">
      <geom name="target_geom" type="cylinder" size="0.022 0.015" rgba="0.05 0.75 0.95 1" conaffinity="1" contype="1"/>
      <site name="target_site" pos="0 0 0" size="0.005" rgba="1 0 0 1"/>
    </body>

    {obstacle_xml}
  </worldbody>

  <equality>
    {eq_xml}
  </equality>

  <contact>
    {exclude_xml}
  </contact>

  <actuator>
    <motor name="motor_0" joint="joint_0" gear="1" ctrlrange="-150 150"/>
    <motor name="motor_1" joint="joint_1" gear="1" ctrlrange="-150 150"/>
    <motor name="motor_2" joint="joint_2" gear="1" ctrlrange="-150 150"/>
  </actuator>
</mujoco>
"""
    return xml


class DeltaMuJoCoEnv:
    """MuJoCo Simulation environment for Delta robot intercept and pick."""

    def __init__(
        self,
        params: Optional[DeltaParams] = None,
        with_obstacle: bool = False,
        obstacle_pos: Tuple[float, float, float] = (0.05, 0.0, -0.58),
        obstacle_radius: float = 0.04,
        timestep: float = 0.002,
        pick_distance_threshold: float = 0.035,
    ):
        self.params = params if params is not None else DeltaParams()
        self.with_obstacle = with_obstacle
        self.obstacle_pos = obstacle_pos
        self.obstacle_radius = obstacle_radius
        self.timestep = timestep
        self.pick_threshold = pick_distance_threshold

        self.xml_string = generate_delta_xml(
            self.params,
            with_obstacle=with_obstacle,
            obstacle_pos=obstacle_pos,
            obstacle_radius=obstacle_radius,
            timestep=timestep,
        )

        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self.joint_ids = [self.model.joint(f"joint_{i}").qposadr[0] for i in range(3)]
        self.joint_dof_ids = [self.model.joint(f"joint_{i}").dofadr[0] for i in range(3)]
        self.ee_body_id = self.model.body("end_effector").id
        self.tcp_site_id = self.model.site("tcp").id
        self.target_mocap_id = self.model.body("target_object").mocapid[0]

        self.obstacle_geom_id = (
            self.model.geom("obstacle_geom").id if with_obstacle else -1
        )

        # Simulation state
        self.is_picked = False
        self.pick_time: Optional[float] = None
        self.collision_detected = False
        self.sim_time = 0.0

    def reset(self, initial_joints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.is_picked = False
        self.pick_time = None
        self.collision_detected = False
        self.sim_time = 0.0

        if initial_joints is None:
            initial_joints = np.array([-np.pi / 6.0, -np.pi / 6.0, -np.pi / 6.0], dtype=np.float64)

        for i in range(3):
            self.data.qpos[self.joint_ids[i]] = initial_joints[i]

        # Calculate initial forward kinematics for end-effector position
        init_pos = forward_kinematics(initial_joints, self.params)
        if init_pos is not None:
            # Set slide joints
            ee_x_id = self.model.joint("ee_x").qposadr[0]
            ee_y_id = self.model.joint("ee_y").qposadr[0]
            ee_z_id = self.model.joint("ee_z").qposadr[0]
            self.data.qpos[ee_x_id] = init_pos[0]
            self.data.qpos[ee_y_id] = init_pos[1]
            # Offset from base body origin pos
            z_offset = self.model.body("end_effector").pos[2]
            self.data.qpos[ee_z_id] = init_pos[2] - z_offset

        mujoco.mj_forward(self.model, self.data)
        return self.get_joint_state()

    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return joint angles theta and velocities thetadot in R^3."""
        theta = np.array([self.data.qpos[qid] for qid in self.joint_ids], dtype=np.float64)
        thetadot = np.array([self.data.qvel[vid] for vid in self.joint_dof_ids], dtype=np.float64)
        return theta, thetadot

    def get_ee_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return end-effector position and velocity in R^3."""
        pos = self.data.site_xpos[self.tcp_site_id].copy()
        # End effector velocity from slide joints
        ee_vx = self.data.qvel[self.model.joint("ee_x").dofadr[0]]
        ee_vy = self.data.qvel[self.model.joint("ee_y").dofadr[0]]
        ee_vz = self.data.qvel[self.model.joint("ee_z").dofadr[0]]
        vel = np.array([ee_vx, ee_vy, ee_vz], dtype=np.float64)
        return pos, vel

    def set_target_position(self, pos: np.ndarray):
        """Set position of kinematically driven target object."""
        self.data.mocap_pos[self.target_mocap_id] = np.asarray(pos, dtype=np.float64)

    def check_collisions(self) -> bool:
        """Check if any robot part penetrates the obstacle."""
        if not self.with_obstacle or self.obstacle_geom_id < 0:
            return False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.obstacle_geom_id or c.geom2 == self.obstacle_geom_id:
                self.collision_detected = True
                return True
        return False

    def step(
        self,
        torques: np.ndarray,
        target_pos: np.ndarray,
        allow_pick: bool = True,
    ) -> Dict[str, Any]:
        """Step simulation by one physics timestep."""
        torques = np.asarray(torques, dtype=np.float64)
        self.data.ctrl[:3] = np.clip(torques, -150.0, 150.0)

        # Update target object kinematics
        if not self.is_picked:
            self.set_target_position(target_pos)
        else:
            # Object follows TCP when grasped
            tcp_pos = self.data.site_xpos[self.tcp_site_id]
            self.set_target_position(tcp_pos - np.array([0.0, 0.0, 0.015]))

        # Step physics
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.timestep

        # Collision detection
        collided = self.check_collisions()

        # Check pick condition
        tcp_pos = self.data.site_xpos[self.tcp_site_id]
        tgt_pos = self.data.mocap_pos[self.target_mocap_id]
        dist_to_target = float(np.linalg.norm(tcp_pos - tgt_pos))

        if allow_pick and (not self.is_picked) and (dist_to_target <= self.pick_threshold):
            self.is_picked = True
            self.pick_time = self.sim_time

        theta, thetadot = self.get_joint_state()
        ee_pos, ee_vel = self.get_ee_state()

        return {
            "time": self.sim_time,
            "theta": theta,
            "thetadot": thetadot,
            "ee_pos": ee_pos,
            "ee_vel": ee_vel,
            "target_pos": tgt_pos,
            "dist_to_target": dist_to_target,
            "is_picked": self.is_picked,
            "pick_time": self.pick_time,
            "collision": collided,
        }
