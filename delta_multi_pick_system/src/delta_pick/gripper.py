from __future__ import annotations

import numpy as np


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


class VirtualWeldGripper:
    """Suction-style grasp using preallocated MuJoCo weld equalities."""

    def __init__(self, model, data, pool, gripper_body_name: str = "gripper"):
        self.model, self.data, self.pool = model, data, pool
        self.gripper_body_id = model.body(gripper_body_name).id

    def attach(self, slot: int) -> None:
        obj_body_id = self.model.body(f"pick_object_{slot:03d}").id
        p_obj = self.data.xpos[obj_body_id].copy()
        q_obj = self.data.xquat[obj_body_id].copy()  # MuJoCo order: w, x, y, z
        p_grip = self.data.xpos[self.gripper_body_id].copy()
        q_grip = self.data.xquat[self.gripper_body_id].copy()
        rel_pos = _quat_to_matrix(q_obj).T @ (p_grip - p_obj)
        rel_quat = _quat_multiply(_quat_conjugate(q_obj), q_grip)
        eq_id = self.pool.eq_ids[slot]
        # Weld eq_data: anchor[3], relpose position[3], relpose quat[4], torque scale.
        self.model.eq_data[eq_id, 3:6] = rel_pos
        self.model.eq_data[eq_id, 6:10] = rel_quat / np.linalg.norm(rel_quat)
        self.data.eq_active[eq_id] = 1

    def detach(self, slot: int) -> None:
        self.data.eq_active[self.pool.eq_ids[slot]] = 0
