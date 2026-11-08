# Continuous Multi-Object Pick and Place

This module connects the CAD-generated delta robot scene to a conveyor pick
application. The repository setup, launch commands, and full folder map are in
the [root README](../README.md).

## Runtime components

- `src/delta_pick/object_pool.py`: preallocated moving objects and conveyor feed.
- `src/delta_pick/tracking.py`: object pose, velocity, and lifecycle tracking.
- `src/delta_pick/scheduler.py`: selects reachable objects with enough time for
  an intercept and place cycle.
- `src/delta_pick/pick_controller.py`: approach, intercept, descend, grasp,
  lift, transfer, release, and return phases.
- `src/delta_pick/robot_controller.py`: delta inverse kinematics and
  joint-space torque tracking for the three motor hinges.
- `src/delta_pick/gripper.py`: suction-style weld grasp.
- `src/delta_pick/scene.py`: adds the object pool and collection bin to the
  generated CAD MJCF scene.
- `src/delta_pick/frame_overlay.py`: live reference-frame axes and coordinates
  in the interactive viewer.
- `src/delta_pick/evaluator.py`: run summary, event CSV, and telemetry CSV.

Run the demo from the repository root with `run_pick_place.py`. The runtime
reads `runtime_config.json`, augments `cad/out/scene.xml` into
`cad/out/scene_multi_pick.xml`, and stores run outputs in `runs/`.

## Scene assumptions

The generated MJCF uses these names:

- Motor joints: `joint_0`, `joint_1`, `joint_2`
- Motor actuators: `motor_0`, `motor_1`, `motor_2`
- Gripper body: `gripper`
- Tool-center-point site: `tool_tcp`
- Conveyor body and belt surface: `conveyor_belt`, `belt_surface`

Objects ride the belt kinematically until picked. The weld grasp is a stable
simulation shortcut; it does not model finger contact. The scheduler is a
real-time heuristic, so some objects can pass the robot unpicked when the
incoming rate exceeds its cycle rate.

## Tuning

Edit `runtime_config.json` to change the spawn interval, belt speed, lanes,
object size, bin position, scheduler reach limits, joint gains, or simulation
duration. The collection bin and conveyor targets must stay within the robot's
reachable workspace. Run output metrics include placed/missed counts, grasp
and placement failures, cycle time, throughput, and placement success rate.
