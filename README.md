# Delta Robot Simulation

One repository for the delta robot's CAD assets, analytical robot modules,
MuJoCo models, continuous conveyor pick-and-place demo, and benchmark scripts.

## Repository map

```text
cad/                         CAD config and exported STEP/STL/MJCF robot assets
delta_robot/                 Kinematics, dynamics, motion planning, control, estimation
delta_multi_pick_system/     Continuous CAD robot pick-and-place application
  src/delta_pick/             Scheduler, tracking, controller, scene, visualization
  tests/                      Pick application tests
docs/                        Design document
results/                     Benchmark plots
run_pick_place.py             Root entry point for the visual pick demo
run_demo.py                   Analytical robot benchmark/demo entry point
verify_mujoco.py              Core model verification routines
experiments.py                PD/CTC benchmark experiments
tests.py                      Core robot module tests
```

The CAD pipeline creates MuJoCo MJCF directly. This checkout has STEP and STL
CAD assets, but no URDF. The pick application uses the CAD-derived MJCF scene
and the `delta_robot` package holds reusable analytical robot functions.

## Set up one Python environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The CAD meshes, manifest, and base MJCF scene are included in `cad/out/`.
Regenerate the MuJoCo scene after changing the config or manifest:

```bash
python cad/build_mjcf.py
```

`cad/build_cad.py` remains as a compatibility alias for the same scene builder.

## Run the continuous pick-and-place visualization

On macOS, MuJoCo's passive viewer needs its `mjpython` launcher. In the
environment above, run:

```bash
.venv/bin/mjpython run_pick_place.py --duration 30
```

On Linux or Windows:

```bash
python run_pick_place.py --duration 30
```

Omit `--duration 30` to use the configured 90-second run. In the viewer, press
`A` to toggle frame axes, `L` to toggle frame labels, and `R` to cycle the
coordinate readout between world and robot-base frames. The robot shows one
frame at each arm's elbow joint, where the upper arm meets its forearm links.

To produce evaluation logs without opening a viewer:

```bash
python run_pick_place.py --headless --duration 30
```

Runs write a JSON summary and CSV event/telemetry logs under
`delta_multi_pick_system/runs/`. The runtime builds
`cad/out/scene_multi_pick.xml`; it keeps the base `cad/out/scene.xml` intact.

## Other robot demos

The root `delta_robot/` package includes analytical forward/inverse kinematics,
dynamics, trajectory generation, RRT obstacle planning, state estimation, and
controllers. `run_demo.py` runs the core verification and controller
benchmarks. Its `--visualize` option opens its standalone MuJoCo demonstration;
on macOS, use `.venv/bin/mjpython run_demo.py --visualize`.

## Pick-and-place behavior

The scheduler chooses a reachable conveyor object with enough predicted time
for interception and placement. The controller follows approach, moving-target
intercept, descent, lift, bin transfer, release, and return waypoints. Objects
that are not selected in time keep moving and may be missed. Grasping is
represented by a MuJoCo weld constraint attached to the visible CAD gripper.
It is a stable suction-style simulation grasp, not a detailed finger-contact
model.

Tune object flow, belt speed, lanes, bin position, joint gains, and scheduling
limits in `delta_multi_pick_system/runtime_config.json`.
