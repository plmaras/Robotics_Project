#!/usr/bin/env python3
"""Generate cad/out/scene.xml and frames.json from config and CAD manifest.

The scene mirrors the CAD parts as visual mesh geoms, uses simple primitives for
collision, takes <inertial> values from the CAD manifest, and adds a
velocity-driven belt body so free objects ride the conveyor.

Visual style: muted grey material palette, key + fill lights, soft headlight
and a dark floor.

Coordinate frames are included for the interactive viewer overlay.

Run from the repository root:
    python cad/build_mjcf.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Part name -> material name (materials are defined in MATERIALS_XML below).
PART_MATERIALS = {
    "base": "plate_mat",
    "motor": "motor_mat",
    "motor_mount": "steel_mat",
    "upper_arm": "arm_mat",
    "forearm": "rod_mat",
    "effector": "effector_mat",
    "gripper": "steel_mat",
    "conveyor_belt": "belt_mat",
    "conveyor_frame": "frame_mat",
}

OBJECT_MATERIALS = [
    "puck_mat",
]

MATERIALS_XML = """    <material name="plate_mat" rgba="0.72 0.74 0.78 1"/>
    <material name="steel_mat" rgba="0.55 0.58 0.62 1"/>
    <material name="motor_mat" rgba="0.12 0.13 0.15 1"/>
    <material name="arm_mat" rgba="0.80 0.82 0.85 1"/>
    <material name="rod_mat" rgba="0.14 0.15 0.17 1"/>
    <material name="effector_mat" rgba="0.70 0.72 0.76 1"/>
    <material name="frame_mat" rgba="0.36 0.38 0.42 1"/>
    <material name="belt_mat" rgba="0.10 0.11 0.13 1" specular="0.15" shininess="0.2"/>
    <material name="puck_mat" rgba="0.95 0.52 0.10 1" specular="0.4" shininess="0.5"/>"""

VISUAL_XML = """  <visual>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="800"/>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.28 0.28 0.30" specular="0.25 0.25 0.25"/>
  </visual>"""
VISUAL_XML = """  <visual>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="800"/>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.28 0.28 0.30" specular="0.25 0.25 0.25"/>
    <scale framelength="0.6" framewidth="0.02"/>
  </visual>"""

FRAME_DEFAULT_XML = """    <default class="frame">
      <site type="sphere" size="0.006" rgba="1 1 1 1" group="2"/>
    </default>"""

LIGHTS_XML = """    <light name="key" pos="2.2 -2.4 3.0" dir="-0.55 0.62 -0.72" diffuse="0.85 0.85 0.85" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill" pos="-2.8 1.6 2.2" dir="0.75 -0.45 -0.6" diffuse="0.35 0.36 0.38" castshadow="false"/>"""

FLOOR_RGBA = "0.20 0.21 0.23 1"
TCP_RGBA = "1 0.6 0 1"


def fnum(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def vec(values: Any, digits: int = 6) -> str:
    return " ".join(fnum(v, digits) for v in values)


def quat_z(deg: float) -> str:
    half = math.radians(deg) / 2.0
    return vec([math.cos(half), 0.0, 0.0, math.sin(half)], 9)


def radial(phi_deg: float) -> np.ndarray:
    phi = math.radians(phi_deg)
    return np.array([math.cos(phi), math.sin(phi), 0.0])


def tangent(phi_deg: float) -> np.ndarray:
    phi = math.radians(phi_deg)
    return np.array([math.sin(phi), -math.cos(phi), 0.0])


def tangent_pos(radius: float, tangent_offset: float, phi_deg: float) -> np.ndarray:
    return radius * radial(phi_deg) + tangent_offset * tangent(phi_deg)


def inertial_xml(part: dict[str, Any], rotation_deg: float = 0.0) -> str:
    inertial = part["mjcf"]["inertial"]
    pos = np.asarray(inertial["pos"], dtype=float)
    full = np.asarray(inertial["fullinertia"], dtype=float)
    if rotation_deg:
        rot = Rotation.from_euler("z", rotation_deg, degrees=True).as_matrix()
        pos = rot @ pos
        inertia = np.array(
            [
                [full[0], full[3], full[4]],
                [full[3], full[1], full[5]],
                [full[4], full[5], full[2]],
            ]
        )
        inertia = rot @ inertia @ rot.T
        full = np.array(
            [
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            ]
        )
    return (
        f'<inertial pos="{vec(pos, 9)}" mass="{fnum(inertial["mass"], 9)}" '
        f'fullinertia="{vec(full, 9)}"/>'
    )


def mesh_visual(part_name: str, material: str, quat: str | None = None) -> str:
    quat_attr = f' quat="{quat}"' if quat else ""
    return (
        f'<geom type="mesh" mesh="{part_name}"{quat_attr} '
        f'class="visual" material="{material}"/>'
    )


def compute_home_qpos(
    model: mujoco.MjModel,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> np.ndarray:
    qpos = model.qpos0.copy()
    delta = config["delta"]
    theta = math.radians(delta["home_theta_deg"])
    rb = delta["base_radius"] / 1000.0
    re = delta["effector_radius"] / 1000.0
    l1 = delta["upper_arm_length"] / 1000.0
    ee_z = manifest["home_pose"]["effector_z_m"]

    for index, phi_deg in enumerate(delta["leg_angles_deg"]):
        r = radial(phi_deg)
        hinge = Rotation.from_rotvec(tangent(phi_deg) * theta)
        elbow = rb * r + hinge.apply(l1 * r)
        anchor = re * r + np.array([0.0, 0.0, ee_z])
        direction = anchor - elbow
        direction /= np.linalg.norm(direction)
        forearm_world = Rotation.align_vectors([direction], [[1.0, 0.0, 0.0]])[0]
        ball = hinge.inv() * forearm_world
        quat = ball.as_quat()
        if quat[3] < 0.0:
            quat = -quat

        hinge_id = model.joint(f"joint_{index}").id
        qpos[model.jnt_qposadr[hinge_id]] = theta
        for side in ("a", "b"):
            ball_id = model.joint(f"ball_{index}_{side}").id
            adr = model.jnt_qposadr[ball_id]
            qpos[adr : adr + 4] = [quat[3], quat[0], quat[1], quat[2]]
    return qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("out") / "manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("out") / "scene.xml",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    parts = {part["name"]: part for part in manifest["parts"]}

    delta = config["delta"]
    conveyor = config["conveyor"]
    scene = config["scene"]
    objects = config["objects"]

    rb = delta["base_radius"] / 1000.0
    re = delta["effector_radius"] / 1000.0
    l1 = delta["upper_arm_length"] / 1000.0
    l2 = delta["forearm_length"] / 1000.0
    angles = delta["leg_angles_deg"]
    theta_min = math.radians(delta["theta_min_deg"])
    theta_max = math.radians(delta["theta_max_deg"])
    ee_z = manifest["home_pose"]["effector_z_m"]

    base = delta["base"]
    motor = delta["motor"]
    motor_mount = delta["motor_mount"]
    upper_arm = delta["upper_arm"]
    forearm = delta["forearm"]
    effector = delta["effector"]
    gripper = delta["gripper"]
    motor_limit = delta["motor_torque_limit"]
    pair_offset = forearm["pair_offset"] / 1000.0
    sides = (("a", -1.0), ("b", 1.0))

    outer_radius = base["outer_radius"] / 1000.0
    base_half = base["thickness"] / 2000.0
    arm_collision_radius = upper_arm["collision_radius"] / 1000.0
    forearm_collision_radius = forearm["rod_radius"] / 1000.0
    plate_radius = effector["plate_radius"] / 1000.0
    plate_half = effector["plate_thickness"] / 2000.0
    tool_flange = effector["tool_flange_thickness"] / 1000.0

    body_l = gripper["body_length"] / 1000.0
    body_w = gripper["body_width"] / 1000.0
    body_h = gripper["body_height"] / 1000.0
    finger_t = gripper["finger_thickness"] / 1000.0
    finger_w = gripper["finger_width"] / 1000.0
    finger_l = gripper["finger_length"] / 1000.0
    opening = gripper["opening"] / 1000.0

    belt_l = conveyor["length"] / 1000.0
    belt_w = conveyor["belt_width"] / 1000.0
    belt_t = conveyor["belt_thickness"] / 1000.0
    rail_w = conveyor["rail_width"] / 1000.0
    rail_h = conveyor["rail_height"] / 1000.0
    rail_gap = conveyor["rail_gap"] / 1000.0
    leg_size = conveyor["leg_size"] / 1000.0
    leg_inset_x = conveyor["leg_inset_x"] / 1000.0
    floor_drop = conveyor["floor_drop"] / 1000.0
    conv_pos = np.asarray(conveyor["world_position_mm"], dtype=float) / 1000.0
    belt_friction = vec(conveyor["belt_friction"], 6)
    object_friction = vec(conveyor["object_friction"], 6)

    belt_top = conv_pos[2] + belt_t / 2.0
    tool_tip_z = ee_z - plate_half - tool_flange + 0.001 - body_h - finger_l
    clearance = tool_tip_z - belt_top
    if clearance < 0.01:
        raise ValueError(
            f"home-pose tool tip ({tool_tip_z:.3f} m) is only {clearance * 1000:.1f} mm "
            f"above the belt ({belt_top:.3f} m); lower the conveyor or shorten the gripper"
        )

    assets = "\n".join(
        f'    <mesh name="{name}" file="{name}.stl" scale="0.001 0.001 0.001"/>'
        for name in [
            "base",
            "motor",
            "motor_mount",
            "upper_arm",
            "forearm",
            "effector",
            "gripper",
            "conveyor_belt",
            "conveyor_frame",
            *[obj["name"] for obj in objects],
        ]
    )

    motor_geoms = "\n".join(
        f'      <geom type="mesh" mesh="motor_mount" '
        f'pos="{vec(tangent_pos(rb, -motor_mount["offset"] / 1000.0, phi))}" '
        f'quat="{quat_z(phi)}" class="visual" material="{PART_MATERIALS["motor_mount"]}"/>\n'
        f'      <geom type="mesh" mesh="motor" '
        f'pos="{vec(tangent_pos(rb, -motor["tangent_offset"] / 1000.0, phi))}" '
        f'quat="{quat_z(phi - 90.0)}" class="visual" material="{PART_MATERIALS["motor"]}"/>'
        for phi in angles
    )
    motor_frames = "\n".join(
        f'      <site name="frame_M{index}_origin" pos="{vec(rb * radial(phi))}" '
        f'quat="{quat_z(phi)}" class="frame"/>'
        for index, phi in enumerate(angles)
    )

    arms_xml = []
    for index, phi in enumerate(angles):
        elbow = l1 * radial(phi)
        rod_bodies = []
        for side, sign in sides:
            attach = elbow + sign * pair_offset * tangent(phi)
            rod_bodies.append(
                f"""
      <body name="forearm_{index}_{side}" pos="{vec(attach)}">
        <joint name="ball_{index}_{side}" type="ball" damping="{fnum(delta['ball_damping'])}"/>
        {inertial_xml(parts['forearm'])}
        {mesh_visual('forearm', PART_MATERIALS['forearm'])}
        <geom name="forearm_col_{index}_{side}" type="capsule"
              fromto="0 0 0 {fnum(l2)} 0 0" size="{fnum(forearm_collision_radius)}"
              class="collision"/>
        <site name="forearm_tip_{index}_{side}" pos="{fnum(l2)} 0 0" size="0.006"/>
      </body>"""
            )
        arms_xml.append(
            f"""
    <body name="upper_arm_{index}" pos="{vec(rb * radial(phi))}">
      <joint name="joint_{index}" type="hinge" axis="{vec(tangent(phi))}"
             range="{fnum(theta_min)} {fnum(theta_max)}"
             damping="{fnum(delta['joint_damping'])}"/>
      <site name="frame_arm_joint_{index}" pos="{vec(elbow)}"
            quat="{quat_z(phi)}" class="frame"/>
      {inertial_xml(parts['upper_arm'], phi)}
      {mesh_visual('upper_arm', PART_MATERIALS['upper_arm'], quat_z(phi))}
      <geom name="upper_arm_col_{index}" type="capsule"
            fromto="0 0 0 {vec(elbow)}" size="{fnum(arm_collision_radius)}"
            class="collision"/>{''.join(rod_bodies)}
    </body>"""
        )
    arms_block = "\n".join(arms_xml)

    anchors = "\n".join(
        f'      <site name="ee_anchor_{index}_{side}" '
        f'pos="{vec(tangent_pos(re, sign * pair_offset, phi))}" size="0.005"/>'
        for index, phi in enumerate(angles)
        for side, sign in sides
    )
    finger_y = opening / 2.0 + finger_w / 2.0
    finger_z = -body_h - finger_l / 2.0 + 0.001
    effector_xml = f"""
    <body name="end_effector" pos="0 0 {fnum(ee_z)}">
      <joint name="ee_x" type="slide" axis="1 0 0" damping="0.01"/>
      <joint name="ee_y" type="slide" axis="0 1 0" damping="0.01"/>
      <joint name="ee_z" type="slide" axis="0 0 1" damping="0.01"/>
      {inertial_xml(parts['effector'])}
      {mesh_visual('effector', PART_MATERIALS['effector'])}
      <geom name="effector_col" type="cylinder"
            size="{fnum(plate_radius)} {fnum(plate_half)}" class="collision"/>
{anchors}
      <site name="tcp" pos="0 0 0" size="0.008" rgba="{TCP_RGBA}"/>
      <site name="frame_P_origin" pos="0 0 0" class="frame"/>
      <body name="gripper" pos="0 0 {fnum(-plate_half - tool_flange + 0.001)}">
        {inertial_xml(parts['gripper'])}
        {mesh_visual('gripper', PART_MATERIALS['gripper'])}
        <geom name="gripper_body_col" type="box"
              size="{fnum(body_l / 2)} {fnum(body_w / 2)} {fnum(body_h / 2)}"
              pos="0 0 {fnum(-body_h / 2)}" class="collision"/>
        <geom name="gripper_finger_0_col" type="box"
              size="{fnum(finger_t / 2)} {fnum(finger_w / 2)} {fnum(finger_l / 2)}"
              pos="0 {fnum(-finger_y)} {fnum(finger_z)}" class="collision"/>
        <geom name="gripper_finger_1_col" type="box"
              size="{fnum(finger_t / 2)} {fnum(finger_w / 2)} {fnum(finger_l / 2)}"
              pos="0 {fnum(finger_y)} {fnum(finger_z)}" class="collision"/>
        <site name="tool_tcp" pos="0 0 {fnum(-body_h - finger_l + 0.001)}"
              size="0.006" rgba="{TCP_RGBA}"/>
        <site name="frame_T_origin" pos="0 0 {fnum(-body_h - finger_l + 0.001)}" class="frame"/>
      </body>
    </body>"""

    rail_y = belt_w / 2.0 + rail_gap + rail_w / 2.0
    rail_z = (rail_h - belt_t) / 2.0
    rail_bottom = rail_z - rail_h / 2.0
    leg_top = rail_bottom + 0.002
    leg_length = floor_drop + leg_top
    leg_z = -floor_drop + leg_length / 2.0

    frame_collisions = []
    for sign in (-1.0, 1.0):
        frame_collisions.append(
            f'      <geom name="rail_col_{"l" if sign < 0 else "r"}" type="box" '
            f'size="{fnum(belt_l / 2)} {fnum(rail_w / 2)} {fnum(rail_h / 2)}" '
            f'pos="0 {fnum(sign * rail_y)} {fnum(rail_z)}" class="collision"/>'
        )
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            frame_collisions.append(
                f'      <geom name="leg_col_{int(sx > 0)}_{int(sy > 0)}" type="box" '
                f'size="{fnum(leg_size / 2)} {fnum(leg_size / 2)} {fnum(leg_length / 2)}" '
                f'pos="{fnum(sx * (belt_l / 2 - leg_inset_x))} '
                f'{fnum(sy * rail_y)} {fnum(leg_z)}" class="collision"/>'
            )
    frame_block = "\n".join(frame_collisions)

    frame_xml = f"""
    <body name="conveyor_frame" pos="{vec(conv_pos)}">
      {mesh_visual('conveyor_frame', PART_MATERIALS['conveyor_frame'])}
{frame_block}
    </body>"""

    belt_xml = f"""
    <body name="conveyor_belt" pos="{vec(conv_pos)}">
      <joint name="belt_slide" type="slide" axis="1 0 0" damping="0"/>
      {inertial_xml(parts['conveyor_belt'])}
      {mesh_visual('conveyor_belt', PART_MATERIALS['conveyor_belt'])}
      <geom name="belt_surface" type="box"
            size="{fnum(belt_l / 2)} {fnum(belt_w / 2)} {fnum(belt_t / 2)}"
            friction="{belt_friction}" class="collision"/>
    </body>"""

    objects_xml = []
    object_names = []
    for index, obj in enumerate(objects):
        name = obj["name"]
        object_names.append(name)
        instance = parts[name]["instances"][0]
        x, y, z = instance["pos_m"]
        z += scene["object_gap_m"]
        radius = obj["radius"] / 1000.0
        half_height = obj["height"] / 2000.0
        material = OBJECT_MATERIALS[index % len(OBJECT_MATERIALS)]
        puck_frame = f'<site name="frame_{name}_origin" class="frame"/>'
        objects_xml.append(
            f"""
    <body name="{name}" pos="{fnum(x)} {fnum(y)} {fnum(z)}">
      <freejoint name="{name}_free"/>
      {inertial_xml(parts[name])}
      {mesh_visual(name, material)}
      <geom name="{name}_col" type="cylinder"
            size="{fnum(radius)} {fnum(half_height)}"
            friction="{object_friction}" class="collision"/>
      {puck_frame}
    </body>"""
        )
    objects_block = "\n".join(objects_xml)

    equalities = "\n".join(
        f'    <connect name="connect_{index}_{side}" '
        f'site1="forearm_tip_{index}_{side}" site2="ee_anchor_{index}_{side}" '
        f'solref="0.002 1"/>'
        for index in range(3)
        for side, _ in sides
    )

    forearm_bodies = [
        f"forearm_{index}_{side}" for index in range(3) for side, _ in sides
    ]
    excludes = []
    for index in range(3):
        excludes.append(f'    <exclude body1="base" body2="upper_arm_{index}"/>')
        for side, _ in sides:
            excludes.append(
                f'    <exclude body1="upper_arm_{index}" '
                f'body2="forearm_{index}_{side}"/>'
            )
            excludes.append(
                f'    <exclude body1="forearm_{index}_{side}" '
                f'body2="end_effector"/>'
            )
    for position, body_a in enumerate(forearm_bodies):
        for body_b in forearm_bodies[position + 1 :]:
            excludes.append(f'    <exclude body1="{body_a}" body2="{body_b}"/>')
    excludes.extend(
        [
            '    <exclude body1="end_effector" body2="conveyor_belt"/>',
            '    <exclude body1="end_effector" body2="conveyor_frame"/>',
            '    <exclude body1="gripper" body2="conveyor_belt"/>',
            '    <exclude body1="gripper" body2="conveyor_frame"/>',
        ]
    )
    for name in object_names:
        excludes.append(f'    <exclude body1="end_effector" body2="{name}"/>')
    excludes_block = "\n".join(excludes)

    actuators = "\n".join(
        f'    <motor name="motor_{index}" joint="joint_{index}" gear="1" '
        f'ctrlrange="-{fnum(motor_limit)} {fnum(motor_limit)}"/>'
        for index in range(3)
    )
    actuators += (
        '\n    <velocity name="belt_velocity" joint="belt_slide" '
        'kv="2000" ctrlrange="-2 2"/>'
    )

    keyframe_sentinel = "  <!-- KEYFRAME -->"
    xml = f"""<mujoco model="delta_cad_conveyor">
  <compiler angle="radian" meshdir="stl" autolimits="true"/>
  <option gravity="0 0 -{fnum(scene['gravity_m_s2'])}"
          timestep="{fnum(scene['timestep_s'])}"
          integrator="{scene['integrator']}">
    <flag equality="enable"/>
  </option>
  <default>
    <default class="frame">
      <site type="sphere" size="0.006" rgba="1 1 1 1" group="2"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1"/>
    </default>
    <default class="collision">
      <geom group="3" rgba="1 0.62 0.1 0.25"/>
    </default>
  </default>
{VISUAL_XML}
  <asset>
{assets}
{MATERIALS_XML}
  </asset>
  <worldbody>
{LIGHTS_XML}
    <site name="frame_W_origin" pos="0 0 0" class="frame"/>
    <geom name="floor" type="plane" size="2 2 0.05"
      pos="0 0 {fnum(scene['floor_z_m'])}"
      rgba="{FLOOR_RGBA}"
      friction="1 0.005 0.0001"/>
    <body name="base" pos="0 0 0">
      {mesh_visual('base', PART_MATERIALS['base'])}
      <geom name="base_collision" type="cylinder"
            size="{fnum(outer_radius)} {fnum(base_half)}" class="collision"/>
      <site name="frame_B_origin" pos="0 0 0" class="frame"/>
{motor_frames}
{motor_geoms}
{arms_block}
    </body>
{effector_xml}
{frame_xml}
{belt_xml}
{objects_block}
  </worldbody>
  <equality>
{equalities}
  </equality>
  <contact>
{excludes_block}
  </contact>
  <actuator>
{actuators}
  </actuator>
{keyframe_sentinel}
</mujoco>
"""

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f".{out_path.name}.tmp")

    temp_path.write_text(xml.replace(keyframe_sentinel, ""))
    try:
        model = mujoco.MjModel.from_xml_path(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

    qpos = compute_home_qpos(model, config, manifest)
    keyframe = (
        "  <keyframe>\n"
        f'    <key name="home" qpos="{vec(qpos, 9)}"/>\n'
        "  </keyframe>"
    )
    out_path.write_text(xml.replace(keyframe_sentinel, keyframe))

    frames = (
        [["W", "site", "frame_W_origin"],
         ["B", "site", "frame_B_origin"],
         ["P", "site", "frame_P_origin"],
         ["T", "site", "frame_T_origin"]]
        + [[f"M{i}", "site", f"frame_M{i}_origin"] for i in range(3)]
        + [[f"E{i}", "site", f"frame_arm_joint_{i}"] for i in range(3)]
        + [[name, "site", f"frame_{name}_origin"] for name in object_names]
    )
    frames_path = out_path.parent / "frames.json"
    frames_path.write_text(json.dumps(frames, indent=2) + "\n")

    scene_model = mujoco.MjModel.from_xml_path(str(out_path))
    data = mujoco.MjData(scene_model)
    mujoco.mj_resetDataKeyframe(scene_model, data, 0)
    mujoco.mj_forward(scene_model, data)

    errors = []
    for index in range(3):
        for side, _ in sides:
            tip = data.site(f"forearm_tip_{index}_{side}").xpos
            anchor = data.site(f"ee_anchor_{index}_{side}").xpos
            errors.append(float(np.linalg.norm(tip - anchor)))
    max_error = max(errors)
    if max_error > 1e-6:
        raise RuntimeError(
            f"home keyframe violates the forearm connect constraints by {max_error:.3e} m"
        )

    print(f"Wrote {out_path}")
    print(f"Wrote {frames_path} ({len(frames)} frames)")
    print(
        f"  bodies={scene_model.nbody} qpos={scene_model.nq} "
        f"actuators={scene_model.nu} equalities=6"
    )
    print(f"  home constraint error: {max_error:.3e} m")
    print(
        f"  belt top z={belt_top:.4f} m | home tool tip z={tool_tip_z:.4f} m "
        f"| clearance={clearance * 1000:.1f} mm"
    )
    print(f"  belt speed: {conveyor['belt_speed_m_s']} m/s")


if __name__ == "__main__":
    main()
