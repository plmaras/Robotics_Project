"""Main Demonstration and Benchmark Runner for 3-DOF Delta Robot Project.

Executes:
  1. NumPy vs MuJoCo Kinematics & Dynamics Verification
  2. The Three Benchmark Experiments (Nominal, Fast/Noisy, Obstacle/Payload)
  3. Formatted Performance Metrics Comparison Table (PD vs CTC)
  4. Optional MuJoCo Interactive Visualization
"""

import argparse
import sys
import os

from verify_mujoco import run_verification
from experiments import run_experiment_suite, plot_experiment_comparisons, print_results_summary_table
import delta_robot as dr


def run_interactive_visualization():
    """Launch interactive MuJoCo passive viewer demonstrating intercept and pick."""
    try:
        from mujoco import viewer
    except ImportError:
        print("MuJoCo viewer not supported on this platform/configuration.")
        return

    import numpy as np

    print("\nLaunching interactive MuJoCo viewer (Press ESC to close)...")
    params = dr.DeltaParams()
    env = dr.DeltaMuJoCoEnv(params=params, with_obstacle=True)
    home_p, home_th = dr.home_position(params)
    env.reset(home_th)

    v_belt = 0.22
    conveyor = dr.ConveyorBelt(velocity=v_belt, start_x=-0.38, y_pos=0.0, z_pos=-0.65)
    ab_filter = dr.AlphaBetaFilter3D(alpha=0.35, beta=0.03, dt=0.002)

    ctc_gains, _ = dr.tune_gains_critical_damping(omega_n=25.0, zeta=1.0)
    controller = dr.ComputedTorqueController(params, ctc_gains)

    # Observation
    sim_time = 0.0
    for _ in range(int(0.20 / 0.002)):
        p_true = conveyor.true_position(sim_time)
        p_est, v_est = ab_filter.update(p_true, dt=0.002)
        sim_time += 0.002

    p_intercept, t_intercept = dr.predict_interception(
        home_p, p_est, v_est, params, t_start=sim_time, min_travel_time=0.60
    )
    travel_duration = t_intercept - sim_time
    target_vel_contact = np.array([v_belt, 0.0, 0.0])

    obstacle = dr.SphereObstacle(center=np.array([0.04, 0.0, -0.58]), radius=0.04, margin=0.02)
    rrt_planner = dr.RRTPlanner3D(params=params, obstacles=[obstacle])
    waypoints = rrt_planner.plan(home_p, p_intercept)
    traj = dr.build_obstacle_avoidance_trajectory(waypoints, v_final=target_vel_contact, total_time=travel_duration)

    sim_time = 0.0
    env.reset(home_th)

    try:
        with viewer.launch_passive(env.model, env.data) as v:
            t_traj_start = 0.20
            while v.is_running():
                if sim_time < t_traj_start:
                    pos_d = home_p
                    vel_d = np.zeros(3)
                    acc_d = np.zeros(3)
                else:
                    t_rel = sim_time - t_traj_start
                    if t_rel <= travel_duration:
                        pos_d, vel_d, acc_d = traj.sample(t_rel)
                    else:
                        pos_d = p_intercept + np.array([v_belt * (t_rel - travel_duration), 0, 0.1 * min(1.0, (t_rel - travel_duration)/0.3)])
                        vel_d = target_vel_contact
                        acc_d = np.zeros(3)

                th_d, ok, _ = dr.inverse_kinematics(pos_d, params)
                J = dr.jacobian(th_d, pos_d, params)
                th_dot_d = np.linalg.inv(J) @ vel_d
                th_cur, th_dot_cur = env.get_joint_state()
                tau = controller.compute_torque(th_d, th_dot_d, np.zeros(3), th_cur, th_dot_cur)

                p_true_obj = conveyor.true_position(sim_time)
                env.step(tau, p_true_obj, allow_pick=(sim_time >= t_traj_start + travel_duration * 0.90))

                v.sync()
                sim_time += 0.002
                if sim_time > 2.5:
                    sim_time = 0.0
                    env.reset(home_th)
    except Exception as e:
        print(f"Interactive viewer session ended: {e}")


def main():
    parser = argparse.ArgumentParser(description="3-DOF Delta Robot Benchmark Suite")
    parser.add_argument("--verify-only", action="store_true", help="Run only NumPy vs MuJoCo verification checks")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per experiment (default: 5)")
    parser.add_argument("--visualize", action="store_true", help="Launch interactive 3D MuJoCo viewer")
    args = parser.parse_args()

    # Always run verification first
    verification_results = run_verification()

    if args.verify_only:
        sys.exit(0)

    # Run Benchmark Experiments
    benchmark_results = run_experiment_suite(num_trials=args.trials)
    plot_experiment_comparisons(benchmark_results, output_dir="results")
    print_results_summary_table(benchmark_results)

    if args.visualize:
        run_interactive_visualization()


if __name__ == "__main__":
    main()
