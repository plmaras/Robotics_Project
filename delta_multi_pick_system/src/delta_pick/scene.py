from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _inertia_cylinder(mass: float, radius: float, height: float) -> tuple[float, float, float]:
    ix = mass * (3.0 * radius * radius + height * height) / 12.0
    iz = 0.5 * mass * radius * radius
    return ix, ix, iz


def _remove_named_body(parent: ET.Element, names: set[str]) -> None:
    for body in list(parent.findall("body")):
        if body.get("name") in names:
            parent.remove(body)
        else:
            _remove_named_body(body, names)


def augment_scene(
    source_xml: Path,
    output_xml: Path,
    cad_config_path: Path,
    runtime: dict[str, Any],
) -> Path:
    """Add a reusable object pool, one collection bin and inactive grasp welds.

    The CAD builder's original one-off objects are removed. The CAD meshes,
    robot, conveyor and six delta-rod equalities are left intact.
    """
    cad = json.loads(cad_config_path.read_text())
    conveyor = cad["conveyor"]
    scene_cfg = cad["scene"]
    conv = [float(v) / 1000.0 for v in conveyor["world_position_mm"]]
    belt_length = float(conveyor["length"]) / 1000.0
    belt_width = float(conveyor["belt_width"]) / 1000.0
    belt_thickness = float(conveyor["belt_thickness"]) / 1000.0
    floor_z = float(scene_cfg["floor_z_m"])
    belt_top = conv[2] + belt_thickness / 2.0

    tree = ET.parse(source_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF is missing <worldbody>")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    original_names = {str(obj["name"]) for obj in cad.get("objects", [])}
    _remove_named_body(worldbody, original_names)
    contact = root.find("contact")
    if contact is not None:
        for exclude in list(contact.findall("exclude")):
            if exclude.get("body1") in original_names or exclude.get("body2") in original_names:
                contact.remove(exclude)

    obj_cfg = runtime["object"]
    radius = float(obj_cfg["radius_m"])
    height = float(obj_cfg["height_m"])
    mass = float(obj_cfg["mass_kg"])
    ix, iy, iz = _inertia_cylinder(mass, radius, height)
    friction = _fmt(obj_cfg.get("friction", [0.8, 0.01, 0.001]))
    pool_size = int(runtime["pool_size"])
    if pool_size < 1:
        raise ValueError("pool_size must be at least 1")

    # The pool bodies start below the floor with contacts disabled. Runtime
    # code activates their collision masks and moves their free joints on spawn.
    for i in range(pool_size):
        name = f"pick_object_{i:03d}"
        body = ET.SubElement(worldbody, "body", {"name": name, "pos": f"0 0 -5"})
        ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
        ET.SubElement(body, "inertial", {
            "pos": "0 0 0", "mass": f"{mass:.9g}",
            "diaginertia": _fmt([ix, iy, iz]),
        })
        ET.SubElement(body, "geom", {
            "name": f"{name}_geom", "type": "cylinder",
            "size": _fmt([radius, height / 2.0]),
            "friction": friction, "rgba": _color(i),
            "contype": "0", "conaffinity": "0", "condim": "4",
        })
        ET.SubElement(equality, "weld", {
            "name": f"grasp_{name}", "body1": name,
            "body2": runtime["controller"].get("gripper_body_name", "gripper"),
            "solref": "0.002 1",
        })

    # Collection bin: a static open-top box just outside one side of the belt.
    bin_cfg = runtime["bin"]
    bx, by = map(float, bin_cfg["position_xy_m"])
    hx, hy = map(float, bin_cfg["inner_half_size_m"])
    wall_h = float(bin_cfg["wall_height_m"])
    wall_t = float(bin_cfg["wall_thickness_m"])
    bin_body = ET.SubElement(worldbody, "body", {
        "name": "collection_bin", "pos": _fmt([bx, by, floor_z])
    })
    common = {"type": "box", "rgba": "0.18 0.42 0.72 1", "friction": "0.8 0.01 0.001", "condim": "4"}
    ET.SubElement(bin_body, "geom", {
        **common, "name": "bin_floor", "pos": _fmt([0, 0, wall_t / 2]),
        "size": _fmt([hx + wall_t, hy + wall_t, wall_t / 2]),
    })
    walls = [
        ("bin_wall_xp", [hx + wall_t / 2, 0, wall_h / 2], [wall_t / 2, hy + wall_t, wall_h / 2]),
        ("bin_wall_xn", [-hx - wall_t / 2, 0, wall_h / 2], [wall_t / 2, hy + wall_t, wall_h / 2]),
        ("bin_wall_yp", [0, hy + wall_t / 2, wall_h / 2], [hx, wall_t / 2, wall_h / 2]),
        ("bin_wall_yn", [0, -hy - wall_t / 2, wall_h / 2], [hx, wall_t / 2, wall_h / 2]),
    ]
    for name, pos, size in walls:
        ET.SubElement(bin_body, "geom", {**common, "name": name, "pos": _fmt(pos), "size": _fmt(size)})

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="unicode", xml_declaration=True)
    # Keep the meshdir relative to the original generated scene directory.
    return output_xml


def _color(index: int) -> str:
    palette = [
        "0.08 0.65 0.92 1", "0.95 0.48 0.12 1", "0.48 0.32 0.82 1",
        "0.20 0.72 0.42 1", "0.88 0.78 0.18 1", "0.88 0.30 0.38 1",
    ]
    return palette[index % len(palette)]
