from __future__ import annotations

from typing import Any

import numpy as np

from delta_robot.kinematics import inverse_kinematics
from delta_robot.params import DeltaParams


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
        self.params = DeltaParams(
            base_radius=float(delta["base_radius"]) / 1000.0,
            effector_radius=float(delta["effector_radius"]) / 1000.0,
            upper_arm=float(delta["upper_arm_length"]) / 1000.0,
            lower_arm=float(delta["forearm_length"]) / 1000.0,
            theta_min=np.radians(float(delta["theta_min_deg"])),
            theta_max=np.radians(float(delta["theta_max_deg"])),
            ball_joint_max=np.pi / 2.0,
        )
        self.leg_angles = np.radians(np.asarray(delta["leg_angles_deg"], dtype=float))
        self.kp = float(cfg.get("joint_position_gain", 120.0))
        self.kd = float(cfg.get("joint_velocity_gain", 12.0))
        self.max_torque = float(cfg["max_torque_nm"])
        self.last_error = np.zeros(3)
        self.last_ik_error: str | None = None

    def position(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def _inverse_kinematics(
        self, effector_target: np.ndarray
    ) -> tuple[np.ndarray | None, str]:
        angles, reachable, reason = inverse_kinematics(
            effector_target, self.params, leg_angles=self.leg_angles
        )
        return (angles, "ok") if reachable else (None, reason)

    def command(self, target: np.ndarray) -> None:
        # Convert the requested TCP position to the effector-centre frame.
        tcp_offset = self.data.site_xpos[self.site_id] - self.data.site_xpos[self.effector_site_id]
        effector_target = np.asarray(target, dtype=float) - tcp_offset
        desired, reason = self._inverse_kinematics(effector_target)
        current = self.data.qpos[self.qpos_ids]
        self.last_ik_error = None if desired is not None else reason
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
