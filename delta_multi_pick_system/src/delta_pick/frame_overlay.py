"""Live frame axes, labels and coordinates for the MuJoCo passive viewer."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_FRAMES = [
    ("W", "site", "frame_W_origin"),
    ("B", "site", "frame_B_origin"),
    ("P", "site", "frame_P_origin"),
    ("T", "site", "frame_T_origin"),
    ("M0", "site", "frame_M0_origin"),
    ("M1", "site", "frame_M1_origin"),
    ("M2", "site", "frame_M2_origin"),
    ("E0", "site", "frame_arm_joint_0"),
    ("E1", "site", "frame_arm_joint_1"),
    ("E2", "site", "frame_arm_joint_2"),
]

AXIS_COLORS = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]

# Reference frames the coordinates can be expressed in (cycled with a key)
REFERENCES = [("W", None), ("B", "frame_B_origin")]


def load_frames(frames_path: Path | None):
    if frames_path is not None and Path(frames_path).exists():
        return [tuple(f) for f in json.loads(Path(frames_path).read_text())]
    return DEFAULT_FRAMES


class FrameOverlay:
    def __init__(self, model, data, frames=DEFAULT_FRAMES, axis_len=0.05,
                 axis_radius=0.0025, show_axes=True, show_labels=True):
        self.model, self.data = model, data
        self.axis_len, self.axis_radius = axis_len, axis_radius
        self.show_axes, self.show_labels = show_axes, show_labels
        self.ref_i = 0
        self.items = []
        for label, kind, name in frames:
            obj = mujoco.mjtObj.mjOBJ_SITE if kind == "site" else mujoco.mjtObj.mjOBJ_BODY
            idx = mujoco.mj_name2id(model, obj, name)
            if idx < 0:
                print(f"[FrameOverlay] '{name}' not found, skipped")
                continue
            self.items.append((label, kind, idx))

    # ---- reference frame -------------------------------------------------
    def cycle_reference(self) -> str:
        self.ref_i = (self.ref_i + 1) % len(REFERENCES)
        return REFERENCES[self.ref_i][0]

    def set_reference(self, label: str) -> None:
        for i, (lab, _) in enumerate(REFERENCES):
            if lab == label:
                self.ref_i = i

    @property
    def ref_label(self) -> str:
        return REFERENCES[self.ref_i][0]

    def _rel(self, p):
        site = REFERENCES[self.ref_i][1]
        if site is None:
            return p
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            return p
        R = self.data.site_xmat[sid].reshape(3, 3)
        return R.T @ (p - self.data.site_xpos[sid])

    # ---- poses -----------------------------------------------------------
    def _pose(self, kind, idx):
        d = self.data
        if kind == "site":
            return d.site_xpos[idx], d.site_xmat[idx].reshape(3, 3)
        return d.xpos[idx], d.xmat[idx].reshape(3, 3)

    def _add(self, scn, gtype, size, pos, mat, rgba, label=""):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float),
                            np.asarray(pos, float), np.asarray(mat, float).flatten(),
                            np.asarray(rgba, np.float32))
        if label:
            try:
                g.label = label
            except Exception:
                pass
        scn.ngeom += 1
        return g

    # ---- drawing ---------------------------------------------------------
    def update(self, viewer) -> None:
        scn = viewer.user_scn
        with viewer.lock():
            scn.ngeom = 0
            for label, kind, idx in self.items:
                p, R = self._pose(kind, idx)
                q = self._rel(p)
                text = (f"{label} ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f})"
                        if self.show_labels else "")
                self._add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0],
                          p, np.eye(3), (1, 1, 1, 1), text)
                if self.show_axes:
                    for k in range(3):
                        g = self._add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                      [self.axis_radius, 0, 0], p, np.eye(3),
                                      AXIS_COLORS[k])
                        if g is not None:
                            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                 self.axis_radius, p,
                                                 p + R[:, k] * self.axis_len)

    def hud(self) -> tuple[str, str]:
        names, vals = [f"ref = {self.ref_label}"], [""]
        for label, kind, idx in self.items:
            p, _ = self._pose(kind, idx)
            q = self._rel(p)
            names.append(label)
            vals.append(f"{q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f}")
        return "\n".join(names), "\n".join(vals)

    def key_callback(self, keycode: int) -> None:
        """Pass to launch_passive(key_callback=overlay.key_callback).
        R = cycle reference frame, L = toggle labels, A = toggle axes."""
        c = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if c == "r":
            print(f"[FrameOverlay] reference -> {self.cycle_reference()}")
        elif c == "l":
            self.show_labels = not self.show_labels
        elif c == "a":
            self.show_axes = not self.show_axes
