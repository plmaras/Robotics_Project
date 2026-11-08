from __future__ import annotations

import mujoco

from .frame_overlay import FrameOverlay, load_frames


def launch_interactive(simulation, duration_s: float | None = None):
    """Run the simulation in MuJoCo's interactive passive viewer."""
    import mujoco.viewer

    frames_path = simulation.runtime["scene_xml"].parent / "frames.json"
    frames = [frame for frame in load_frames(frames_path) if not frame[0].startswith("puck_")]
    overlay = FrameOverlay(simulation.model, simulation.data, frames)
    with mujoco.viewer.launch_passive(
        simulation.model, simulation.data, key_callback=overlay.key_callback
    ) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        return simulation.run(viewer=viewer, duration_s=duration_s, overlay=overlay)
