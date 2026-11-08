from __future__ import annotations

from typing import Any

import numpy as np

from .scheduler import OptimizedScheduler
from .gripper import VirtualWeldGripper


class PickPlaceController:
    """State machine for intercept, grasp, lift, transfer-to-bin and return."""

    def __init__(self, model, data, pool, robot, runtime: dict[str, Any]):
        self.model, self.data, self.pool, self.robot = model, data, pool, robot
        self.cfg = runtime["controller"]
        self.scheduler = OptimizedScheduler(pool, runtime)
        self.gripper = VirtualWeldGripper(model, data, pool, self.cfg.get("gripper_body_name", "gripper"))
        self.phase = "IDLE"
        self.phase_started = 0.0
        self.current_slot: int | None = None
        self.home = robot.position().copy()
        self.pick_xy: np.ndarray | None = None
        self.bin_xy = np.asarray(runtime["bin"]["position_xy_m"], dtype=float)
        self.bin_floor_z = self.pool.floor_z  # world-frame floor reference
        self.bin_wall_h = float(runtime["bin"]["wall_height_m"])
        self.completed_cycles: list[float] = []
        self.failed_grasps = 0
        self.failed_placements = 0
        self.phase_timeout = float(self.cfg["phase_timeout_s"])
        self.settle_time = float(self.cfg["settle_time_s"])
        self.last_pick_start: float | None = None

    def _object(self):
        return self.pool.object_for_slot(self.current_slot) if self.current_slot is not None else None

    def _predicted_top(self, obj, horizon: float) -> np.ndarray:
        x = obj.x + self.pool.speed * horizon
        return np.array([x, obj.y, self.pool.belt_top + obj.height + float(self.cfg["pick_clearance_m"])])

    def _return_to_idle(self, sim_time: float) -> None:
        self.phase = "RETURN"
        self.phase_started = sim_time

    def _target(self, sim_time: float) -> np.ndarray:
        ee = self.robot.position()
        obj = self._object()
        if self.phase == "APPROACH" and obj is not None:
            horizon = max(0.08, float(np.linalg.norm(ee - np.array([obj.x, obj.y, self.pool.spawn_z]))) / max(float(self.cfg["max_cartesian_speed_m_s"]), 1e-3))
            top = self._predicted_top(obj, horizon)
            return top + np.array([0.0, 0.0, float(self.cfg["approach_height_m"])])
        if self.phase == "DESCEND" and obj is not None:
            return self._predicted_top(obj, 0.07)
        if self.phase == "LIFT" and self.pick_xy is not None:
            return np.array([self.pick_xy[0], self.pick_xy[1], self.pick_xy[2] + float(self.cfg["lift_height_m"])])
        if self.phase == "TRANSFER":
            return np.array([self.bin_xy[0], self.bin_xy[1], self.bin_floor_z + self.bin_wall_h + 0.11])
        if self.phase == "RETURN":
            return self.home
        if self.phase == "IDLE":
            return self.home
        return ee

    def step(self, sim_time: float) -> None:
        ee = self.robot.position()
        obj = self._object()

        if self.phase == "IDLE":
            candidate = self.scheduler.choose(ee, sim_time)
            if candidate is not None:
                self.current_slot = candidate.obj.slot
                self.last_pick_start = sim_time
                self.phase = "APPROACH"
                self.phase_started = sim_time

        elif self.phase in ("APPROACH", "DESCEND"):
            if obj is None or obj.status != "on_belt" or obj.x >= self.pool.exit_x:
                if self.current_slot is not None and obj is not None and obj.status == "on_belt":
                    self.pool.mark_missed(self.current_slot)
                self._return_to_idle(sim_time)
            else:
                target = self._target(sim_time)
                distance = float(np.linalg.norm(target - ee))
                if self.phase == "APPROACH" and distance < 0.025:
                    self.phase = "DESCEND"
                    self.phase_started = sim_time
                elif self.phase == "DESCEND" and distance < float(self.cfg["grasp_distance_m"]):
                    self.pool.mark_grasping(self.current_slot)
                    self.phase = "GRASP"
                    self.phase_started = sim_time
                elif sim_time - self.phase_started > self.phase_timeout:
                    self.pool.mark_missed(self.current_slot)
                    self._return_to_idle(sim_time)

        elif self.phase == "GRASP":
            if obj is None or obj.status != "grasping":
                self.failed_grasps += 1
                self._return_to_idle(sim_time)
            else:
                tcp = self.robot.position()
                expected = np.array([obj.x, obj.y, self.pool.belt_top + obj.height + float(self.cfg["pick_clearance_m"])])
                if np.linalg.norm(tcp - expected) <= float(self.cfg["grasp_distance_m"]) * 1.5:
                    self.gripper.attach(self.current_slot)
                    self.pool.mark_picked(self.current_slot, sim_time)
                    self.pick_xy = tcp.copy()
                    self.phase = "LIFT"
                    self.phase_started = sim_time
                elif sim_time - self.phase_started > 0.20:
                    self.failed_grasps += 1
                    self.pool.mark_missed(self.current_slot)
                    self._return_to_idle(sim_time)

        elif self.phase == "LIFT":
            if np.linalg.norm(self._target(sim_time) - ee) < 0.025:
                self.phase = "TRANSFER"
                self.phase_started = sim_time
            elif sim_time - self.phase_started > self.phase_timeout * 2.0:
                self.failed_placements += 1
                self.gripper.detach(self.current_slot)
                self.pool.mark_missed(self.current_slot)
                self._return_to_idle(sim_time)

        elif self.phase == "TRANSFER":
            if np.linalg.norm(self._target(sim_time) - ee) < 0.035:
                self.phase = "RELEASE"
                self.phase_started = sim_time
            elif sim_time - self.phase_started > self.phase_timeout * 2.0:
                self.failed_placements += 1
                self.gripper.detach(self.current_slot)
                self.pool.mark_missed(self.current_slot)
                self._return_to_idle(sim_time)

        elif self.phase == "RELEASE":
            if sim_time - self.phase_started >= self.settle_time:
                self.gripper.detach(self.current_slot)
                self.pool.mark_placed(self.current_slot, sim_time)
                if self.last_pick_start is not None:
                    self.completed_cycles.append(sim_time - self.last_pick_start)
                self._return_to_idle(sim_time)

        elif self.phase == "RETURN":
            if np.linalg.norm(self.home - ee) < 0.035:
                self.phase = "IDLE"
                self.phase_started = sim_time
                self.current_slot = None
                self.pick_xy = None

        self.robot.command(self._target(sim_time))

    def status(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "current_slot": self.current_slot,
            "failed_grasps": self.failed_grasps,
            "failed_placements": self.failed_placements,
            "completed_cycles": len(self.completed_cycles),
            "mean_cycle_time_s": float(np.mean(self.completed_cycles)) if self.completed_cycles else None,
        }
