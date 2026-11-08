from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mujoco

from .object_pool import ObjectPool
from .tracking import ObjectTracker
from .robot_controller import DeltaIKTorqueController
from .pick_controller import PickPlaceController
from .evaluator import Evaluator


class MultiPickSimulation:
    def __init__(self, model, runtime: dict[str, Any], cad_config_path: Path, output_dir: Path):
        self.model = model
        self.data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, self.data, key_id)
        else:
            mujoco.mj_resetData(model, self.data)
        mujoco.mj_forward(model, self.data)
        cad_config = json.loads(cad_config_path.read_text())
        self.pool = ObjectPool(model, self.data, runtime, cad_config)
        mujoco.mj_forward(model, self.data)  # grasp welds are now disabled
        self.tracker = ObjectTracker(self.pool)
        self.robot = DeltaIKTorqueController(model, self.data, runtime, cad_config)
        self.controller = PickPlaceController(model, self.data, self.pool, self.robot, runtime)
        self.evaluator = Evaluator(output_dir, self.pool, self.controller)
        self.runtime = runtime

    def step(self) -> None:
        dt = float(self.model.opt.timestep)
        sim_time = float(self.data.time)
        self.pool.spawn_if_due(sim_time)
        self.pool.advance(dt)
        # Conveyor objects are kinematically carried by the belt. Forward
        # recomputation makes their updated poses visible to tracking/control.
        mujoco.mj_forward(self.model, self.data)
        self.controller.step(sim_time)
        mujoco.mj_step(self.model, self.data)

    def run(self, viewer=None, duration_s: float | None = None, overlay=None) -> dict[str, Any]:
        duration = float(duration_s or self.runtime["simulation"]["duration_s"])
        log_interval = float(self.runtime["simulation"]["log_interval_s"])
        next_log = 0.0
        realtime_factor = max(float(self.runtime["simulation"].get("realtime_factor", 1.0)), 1e-3)
        dt = float(self.model.opt.timestep)
        try:
            while self.data.time < duration:
                if viewer is not None and not viewer.is_running():
                    break
                self.step()
                if viewer is not None:
                    if overlay is not None:
                        overlay.update(viewer)
                    viewer.sync()
                if self.data.time >= next_log:
                    self.evaluator.update(float(self.data.time), log_telemetry=True)
                    state = self.controller.status()
                    print(f"t={self.data.time:6.2f}s phase={state['phase']:<9} spawned={self.pool.spawned:3d} placed={self.pool.placed:3d} missed={self.pool.missed:3d} active={len(self.pool.active_objects()):3d}")
                    next_log += log_interval
                if viewer is not None:
                    time.sleep(dt / realtime_factor)
            self.evaluator.update(float(self.data.time), log_telemetry=True)
            return self.evaluator.summary(float(self.data.time))
        finally:
            self.evaluator.close()
