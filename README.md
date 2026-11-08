# Delta Robot Simulation

One Python package for analytical delta-robot control and the CAD-backed
multi-object pick-and-place simulation. A unified command line exposes the
benchmark, verification, CAD-scene, and pick-and-place workflows; the CAD
build and viewer scripts also remain directly runnable.

## Repository layout

```text
cad/                         CAD configuration, MJCF generation, and scene viewer
  out/                       Generated MJCF, mesh exports, and frame metadata
config/pick_place.json       Pick-and-place runtime configuration
delta_robot/                 Shared kinematics, dynamics, planning, and control
  pick/                      CAD-backed pick-and-place application
scripts/                     Benchmark, verification, and pick launchers
tests/                       Robot and pick-and-place tests
docs/                        Design and application notes
results/                     Benchmark plots and pick simulation logs
```

The CAD pipeline generates MuJoCo MJCF directly. The checked-in `cad/out/`
assets include STEP/STL exports and the base scene. The analytical benchmark
uses its own compact MuJoCo model; the multi-pick application and CAD scene
viewer use the CAD-generated scene.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Regenerate the CAD scene after changing the CAD configuration or manifest:

```bash
python cad/build_mjcf.py
```

`cad/build_cad.py` remains as a compatibility alias for the scene builder.

## Unified command line

Run commands from the repository root:

```bash
python -m delta_robot --help
python -m delta_robot verify
python -m delta_robot benchmark --trials 5
python -m delta_robot benchmark --visualize
python -m delta_robot pick --headless --duration 30
```

The pick-and-place viewer is enabled by default. On macOS, launch it with
MuJoCo's Python viewer launcher:

```bash
.venv/bin/mjpython -m delta_robot pick --duration 30
```

Omit `--duration` to use the configured 90-second run. Pick runs write
summaries and CSV logs under `results/pick_place/runs/`. The runtime generates
`cad/out/scene_multi_pick.xml` and leaves the base `cad/out/scene.xml` intact.

## CAD scene tools

CAD commands can also be run directly, without the unified CLI:

```bash
python cad/run_scene.py                   # headless conveyor check
python cad/run_scene.py --viewer           # interactive viewer
python -m delta_robot cad-scene --viewer   # same viewer via unified CLI
```

On macOS, use `.venv/bin/mjpython` instead of `python` for either interactive
viewer command.

## Capabilities

- Analytical forward/inverse kinematics, dynamics, trajectory generation,
  obstacle planning, state estimation, and PD/computed-torque controllers.
- Verification against MuJoCo and repeatable benchmark experiments.
- CAD-backed continuous multi-object conveyor picking, with tracking,
  scheduling, suction-style weld grasping, and evaluation logs.
- Interactive visualization for the analytical demo, CAD scene, and
  multi-object pick simulation.

See [the pick-and-place notes](docs/multi_pick.md) and
[the design document](docs/DesignDocument.pdf) for additional details.
