from __future__ import annotations

from typing import Any

import numpy as np


class DeltaIKTorqueController:
    """Track Cartesian targets through delta IK and joint-space torque control.

    The CAD MJCF models the end effector as a separate body constrained to the
    three arms by equality constraints. Its ordinary site Jacobian therefore
    has no columns for the motor hinges. We solve the delta inverse kinematics
    directly and close the loop on the three actuated hinge angles instead.
    """

    def __init__(self, model, data, runtime: dict[str, Any], cad_config: dict[str, Any]):
        self.model, self.data = model, data
        cfg = runtime["controller"]
        self.site_name = cfg.get("site_name", "tool_tcp")
        self.site_id = model.site(self.site_name).id
        self.effector_site_id = model.site("tcp").id
        self.joint_ids = [model.joint(n).id for n in cfg["joint_names"]]
        self.dof_ids = [int(model.jnt_dofadr[j]) for j in self.joint_ids]
        self.qpos_ids = [int(model.jnt_qposadr[j]) for j in self.joint_ids]
        self.actuator_ids = [model.actuator(n).id for n in cfg["actuator_names"]]
        delta = cad_config["delta"]
        self.base_radius = float(delta["base_radius"]) / 1000.0
        self.effector_radius = float(delta["effector_radius"]) / 1000.0
        self.upper_arm = float(delta["upper_arm_length"]) / 1000.0
        self.lower_arm = float(delta["forearm_length"]) / 1000.0
        self.leg_angles = np.radians(np.asarray(delta["leg_angles_deg"], dtype=float))
        self.theta_min = np.radians(float(delta["theta_min_deg"]))
        self.theta_max = np.radians(float(delta["theta_max_deg"]))
        self.kp = float(cfg.get("joint_position_gain", 120.0))
        self.kd = float(cfg.get("joint_velocity_gain", 12.0))
        self.max_torque = float(cfg["max_torque_nm"])
        self.last_error = np.zeros(3)
        self.last_ik_error: str | None = None

    def position(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def _inverse_kinematics(self, effector_target: np.ndarray) -> np.ndarray | None:
        x, y, z = map(float, effector_target)
        angles = []
        for phi in self.leg_angles:
            c, s = np.cos(phi), np.sin(phi)
            # Rotate the target into the leg plane and account for the two radii.
            a = c * x + s * y + self.effector_radius - self.base_radius
            b = -s * x + c * y
            A = 2.0 * a * self.upper_arm
            B = 2.0 * z * self.upper_arm
            C = (a*a + b*b + z*z + self.upper_arm**2 - self.lower_arm**2)
            radius = float(np.hypot(A, B))
            if radius < 1e-12 or abs(C / radius) > 1.0 + 1e-9:
                return None
            theta = np.arctan2(B, A) + np.arccos(np.clip(C / radius, -1.0, 1.0))
            theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
            if theta < self.theta_min or theta > self.theta_max:
                return None
            angles.append(theta)
        return np.asarray(angles)

    def command(self, target: np.ndarray) -> None:
        # Convert the requested TCP position to the effector-centre frame.
        tcp_offset = self.data.site_xpos[self.site_id] - self.data.site_xpos[self.effector_site_id]
        effector_target = np.asarray(target, dtype=float) - tcp_offset
        desired = self._inverse_kinematics(effector_target)
        current = self.data.qpos[self.qpos_ids]
        self.last_ik_error = None if desired is not None else "unreachable_or_joint_limit"
        if desired is None:
            # Hold the current pose if a scheduler target falls outside the IK workspace.
            desired = current.copy()
        self.last_error = desired - current
        qvel = self.data.qvel[self.dof_ids]
        tau = self.kp * self.last_error - self.kd * qvel + self.data.qfrc_bias[self.dof_ids]
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        for actuator_id, value in zip(self.actuator_ids, tau):
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            if self.model.actuator_ctrllimited[actuator_id]:
                value = float(np.clip(value, lo, hi))
            self.data.ctrl[actuator_id] = value
