from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import mujoco
from delta_pick.config import load_runtime_config
from delta_pick.scene import augment_scene
from delta_pick.simulation import MultiPickSimulation
from delta_pick.visualization import launch_interactive


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous multi-object delta robot pick-and-place")
    parser.add_argument("--config", default=str(ROOT / "runtime_config.json"))
    parser.add_argument("--duration", type=float, default=None, help="simulation seconds")
    parser.add_argument("--headless", action="store_true", help="run without the interactive viewer")
    parser.add_argument("--no-build-scene", action="store_true", help="reuse the already augmented MJCF")
    args = parser.parse_args()

    runtime, project_root = load_runtime_config(args.config)
    cad_cfg_path = runtime["cad_config"]
    source_scene = runtime["scene_xml"]
    output_scene = runtime["output_scene_xml"]
    if not cad_cfg_path.exists():
        raise FileNotFoundError(f"CAD config not found: {cad_cfg_path}")
    if not source_scene.exists():
        raise FileNotFoundError(
            f"Generated MJCF not found: {source_scene}\n"
            "First run your existing CAD pipeline: python cad/build_cad.py && python cad/build_mjcf.py"
        )

    if not args.no_build_scene:
        augment_scene(source_scene, output_scene, cad_cfg_path, runtime)
    if not output_scene.exists():
        raise FileNotFoundError(f"Augmented MJCF not found: {output_scene}")

    model = mujoco.MjModel.from_xml_path(str(output_scene))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = runtime["simulation"]["output_dir"] / run_id
    simulation = MultiPickSimulation(model, runtime, cad_cfg_path, output_dir)

    if args.headless:
        summary = simulation.run(viewer=None, duration_s=args.duration)
    else:
        summary = launch_interactive(simulation, duration_s=args.duration)

    print("\n=== MULTI-OBJECT PICKING EVALUATION ===")
    print(json.dumps(summary, indent=2))
    print(f"\nLogs saved to: {output_dir}")


if __name__ == "__main__":
    main()
