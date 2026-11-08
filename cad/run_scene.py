#!/usr/bin/env python3
"""Run or verify the CAD-derived MuJoCo scene.

    python cad/run_scene.py                     # headless conveyor check
    mjpython cad/run_scene.py --viewer           # macOS interactive viewer
    python cad/run_scene.py --viewer             # Linux/Windows viewer
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def load_scene(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


class SceneDriver:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict[str, Any],
        speed: float,
    ) -> None:
        self.model = model
        self.data = data
        self.speed = speed

        delta = config["delta"]
        self.limit = float(delta["motor_torque_limit"])
        self.kp = float(delta["pd_kp"])
        self.kd = float(delta["pd_kd"])

        motor_joints = [model.joint(f"joint_{i}") for i in range(3)]
        self.motor_qadr = np.array([model.jnt_qposadr[j.id] for j in motor_joints])
        self.motor_dofadr = np.array([model.jnt_dofadr[j.id] for j in motor_joints])
        self.motor_act = np.array([model.actuator(f"motor_{i}").id for i in range(3)])
        self.home = data.qpos[self.motor_qadr].copy()

        belt_joint = model.joint("belt_slide")
        self.belt_qadr = int(model.jnt_qposadr[belt_joint.id])
        self.belt_dofadr = int(model.jnt_dofadr[belt_joint.id])
        self.belt_act = model.actuator("belt_velocity").id

    def apply_control(self) -> None:
        q = self.data.qpos[self.motor_qadr]
        qd = self.data.qvel[self.motor_dofadr]
        tau = np.clip(self.kp * (self.home - q) - self.kd * qd, -self.limit, self.limit)
        self.data.ctrl[self.motor_act] = tau
        self.data.ctrl[self.belt_act] = self.speed

    def step(self) -> None:
        self.data.qpos[self.belt_qadr] = 0.0
        self.apply_control()
        mujoco.mj_step(self.model, self.data)


def home_constraint_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    errors = []
    for index in range(3):
        for side in ("a", "b"):
            tip = data.site(f"forearm_tip_{index}_{side}").xpos
            anchor = data.site(f"ee_anchor_{index}_{side}").xpos
            errors.append(float(np.linalg.norm(tip - anchor)))
    return max(errors)


def object_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    return int(model.jnt_qposadr[model.joint(f"{name}_free").id])


def run_check(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    driver: SceneDriver,
    config: dict[str, Any],
    seconds: float,
) -> None:
    print(f"scene: nbody={model.nbody} nq={model.nq} nu={model.nu}")
    print(f"home constraint error: {home_constraint_error(model, data):.3e} m")
    print(f"belt speed target: {driver.speed:.3f} m/s")

    conveyor = config["conveyor"]
    belt_top = conveyor["world_position_mm"][2] / 1000.0 + conveyor["belt_thickness"] / 2000.0
    start = {}
    for obj in config["objects"]:
        name = obj["name"]
        adr = object_qpos_addr(model, name)
        start[name] = data.qpos[adr : adr + 3].copy()

    steps = int(round(seconds / model.opt.timestep))
    for _ in range(steps):
        driver.step()
        if not np.isfinite(data.qpos).all():
            raise RuntimeError("simulation diverged (non-finite qpos)")
        if data.qpos[driver.belt_qadr] != 0.0:
            data.qpos[driver.belt_qadr] = 0.0

    print(f"\nafter {seconds:.2f} s:")
    print(f"{'object':<14}{'dx m':>9}{'bottom z':>12}{'belt top':>11}{'z err mm':>10}")
    for obj in config["objects"]:
        name = obj["name"]
        adr = object_qpos_addr(model, name)
        pos = data.qpos[adr : adr + 3]
        dx = float(pos[0] - start[name][0])
        bottom = float(pos[2] - obj["height"] / 2000.0)
        z_err = (bottom - belt_top) * 1000.0
        print(
            f"{name:<14}{dx:>9.3f}{bottom:>12.4f}{belt_top:>11.4f}{z_err:>10.2f}"
        )
        if dx < 0.1:
            raise RuntimeError(f"{name} did not move with the belt (dx={dx:.3f} m)")
        if abs(z_err) > 10.0:
            raise RuntimeError(f"{name} left the belt surface (z error={z_err:.1f} mm)")

    print(f"\nbelt speed reached: {data.qvel[driver.belt_dofadr]:.3f} m/s")
    print("check passed")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    driver: SceneDriver,
) -> None:
    import mujoco.viewer

    wall_start = time.time()
    sim_start = data.time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            driver.step()
            data.qpos[driver.belt_qadr] = 0.0
            viewer.sync()
            behind = (data.time - sim_start) - (time.time() - wall_start)
            if behind > 0.0:
                time.sleep(behind)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path(__file__).with_name("out") / "scene.xml",
    )
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=None)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    speed = args.speed if args.speed is not None else config["conveyor"]["belt_speed_m_s"]

    model, data = load_scene(args.scene)
    driver = SceneDriver(model, data, config, float(speed))

    if args.viewer:
        run_viewer(model, data, driver)
    else:
        run_check(model, data, driver, config, args.seconds)


if __name__ == "__main__":
    main()
