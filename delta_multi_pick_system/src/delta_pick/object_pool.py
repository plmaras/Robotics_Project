from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackedObject:
    slot: int
    name: str
    spawn_id: int
    x: float
    y: float
    z: float
    radius: float
    height: float
    spawn_time: float
    status: str = "on_belt"
    pick_time: float | None = None
    place_time: float | None = None


class ObjectPool:
    """Finite MJCF body pool used to implement continuous object spawning.

    Objects on the conveyor are kinematic in X/Z so the belt behaves like an
    endless conveyor even though the CAD scene contains a finite belt mesh.
    Once grasped, the object is handed back to MuJoCo dynamics and attached to
    the gripper with a weld equality. Placed objects remain in the collection
    bin; missed objects are parked below the scene.
    """

    def __init__(self, model, data, runtime: dict[str, Any], cad_config: dict[str, Any]):
        self.model, self.data, self.cfg = model, data, runtime
        self.obj_cfg = runtime["object"]
        self.radius = float(self.obj_cfg["radius_m"])
        self.height = float(self.obj_cfg["height_m"])
        self.mass = float(self.obj_cfg["mass_kg"])
        self.speed = float(runtime["belt_speed_m_s"])
        self.spawn_interval = float(runtime["spawn_interval_s"])
        self.pool_size = int(runtime["pool_size"])
        self.lanes = list(map(float, runtime.get("lane_positions_m", [0.0])))
        conv = cad_config["conveyor"]
        self.conv_x, self.conv_y, self.conv_z = [float(v) / 1000 for v in conv["world_position_mm"]]
        self.length = float(conv["length"]) / 1000
        self.width = float(conv["belt_width"]) / 1000
        self.thickness = float(conv["belt_thickness"]) / 1000
        self.belt_top = self.conv_z + self.thickness / 2
        self.floor_z = float(cad_config.get("scene", {}).get("floor_z_m", self.conv_z - 0.2))
        self.spawn_x = self.conv_x - self.length / 2 + float(runtime.get("spawn_margin_m", 0.07))
        self.exit_x = self.conv_x + self.length / 2 - float(runtime.get("exit_margin_m", 0.07))
        self.object_gap = float(cad_config.get("scene", {}).get("object_gap_m", 0.002))
        self.spawn_z = self.belt_top + self.height / 2 + self.object_gap
        self.next_spawn_time = 0.25
        self.spawned = 0
        self.missed = 0
        self.placed = 0
        self.objects: dict[int, TrackedObject] = {}
        self.slot_state = ["unused"] * self.pool_size
        self.freejoint_qadr: list[int] = []
        self.freejoint_dadr: list[int] = []
        self.geom_ids: list[int] = []
        self.eq_ids: list[int] = []
        self.eq_names: list[str] = []
        for i in range(self.pool_size):
            name = f"pick_object_{i:03d}"
            joint = model.joint(f"{name}_free")
            self.freejoint_qadr.append(int(model.jnt_qposadr[joint.id]))
            self.freejoint_dadr.append(int(model.jnt_dofadr[joint.id]))
            self.geom_ids.append(model.geom(f"{name}_geom").id)
            eq_name = f"grasp_{name}"
            import mujoco
            eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
            if eq_id < 0:
                raise KeyError(f"Missing grasp equality {eq_name}")
            self.eq_ids.append(eq_id)
            self.eq_names.append(eq_name)
            data.eq_active[eq_id] = 0
            model.geom_contype[self.geom_ids[-1]] = 0
            model.geom_conaffinity[self.geom_ids[-1]] = 0

    def spawn_if_due(self, sim_time: float) -> TrackedObject | None:
        if sim_time + 1e-9 < self.next_spawn_time:
            return None
        slot = next((i for i, state in enumerate(self.slot_state) if state == "unused"), None)
        if slot is None:
            return None
        self.next_spawn_time = sim_time + self.spawn_interval
        spawn_id = self.spawned
        self.spawned += 1
        lane = self.lanes[spawn_id % len(self.lanes)]
        # Small deterministic lane offset avoids random seed-dependent tests.
        y = self.conv_y + lane
        obj = TrackedObject(
            slot=slot, name=f"pick_object_{slot:03d}", spawn_id=spawn_id,
            x=self.spawn_x, y=y, z=self.spawn_z, radius=self.radius,
            height=self.height, spawn_time=sim_time,
        )
        self.objects[slot] = obj
        self.slot_state[slot] = "on_belt"
        qadr, dadr = self.freejoint_qadr[slot], self.freejoint_dadr[slot]
        self.data.qpos[qadr:qadr + 7] = [obj.x, obj.y, obj.z, 1, 0, 0, 0]
        self.data.qvel[dadr:dadr + 6] = 0
        gid = self.geom_ids[slot]
        self.model.geom_contype[gid] = 1
        self.model.geom_conaffinity[gid] = 1
        return obj

    def advance(self, dt: float) -> list[TrackedObject]:
        newly_missed = []
        for slot, obj in list(self.objects.items()):
            if obj.status != "on_belt":
                continue
            obj.x += self.speed * dt
            qadr, dadr = self.freejoint_qadr[slot], self.freejoint_dadr[slot]
            self.data.qpos[qadr:qadr + 3] = [obj.x, obj.y, obj.z]
            self.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
            self.data.qvel[dadr:dadr + 6] = 0
            if obj.x >= self.exit_x:
                self.mark_missed(slot)
                newly_missed.append(obj)
        return newly_missed

    def mark_grasping(self, slot: int) -> None:
        obj = self.objects[slot]
        if obj.status == "on_belt":
            obj.status = "grasping"
            self.slot_state[slot] = "grasping"

    def mark_picked(self, slot: int, sim_time: float) -> None:
        obj = self.objects[slot]
        obj.status = "picked"
        obj.pick_time = sim_time
        self.slot_state[slot] = "picked"

    def mark_placed(self, slot: int, sim_time: float) -> None:
        obj = self.objects[slot]
        body_id = self.model.body(obj.name).id
        obj.x, obj.y, obj.z = map(float, self.data.xpos[body_id])
        obj.status = "placed"
        obj.place_time = sim_time
        self.slot_state[slot] = "placed"
        self.placed += 1

    def mark_missed(self, slot: int) -> None:
        obj = self.objects.get(slot)
        if obj is None or obj.status in ("missed", "placed"):
            return
        obj.status = "missed"
        self.slot_state[slot] = "missed"
        self.missed += 1
        qadr, dadr = self.freejoint_qadr[slot], self.freejoint_dadr[slot]
        self.data.qpos[qadr:qadr + 3] = [0, 0, -5 - slot * 0.01]
        self.data.qvel[dadr:dadr + 6] = 0
        gid = self.geom_ids[slot]
        self.model.geom_contype[gid] = 0
        self.model.geom_conaffinity[gid] = 0

    def active_objects(self) -> list[TrackedObject]:
        return [o for o in self.objects.values() if o.status == "on_belt"]

    def object_for_slot(self, slot: int) -> TrackedObject | None:
        return self.objects.get(slot)
