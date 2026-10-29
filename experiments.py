"""Benchmark Experiments comparing PD and Computed-Torque Control (CTC).

Runs three rigorous robotics experiments:
  1. Nominal Experiment: Standard belt velocity, low sensor noise, nominal dynamics.
  2. Faster Target with Noisy Position: Fast belt speed, high sensor noise, Alpha-Beta filter.
  3. Added Obstacle and Payload: 3D obstacle avoidance via RRT planner, payload mass uncertainty.

Evaluates and compares:
  - Trajectory tracking error (RMSE and Max Error)
  - Interception success rate (%)
  - Pick cycle time (s)
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import delta_robot as dr


def run_trial(
    exp_num: int,
    ctrl_type: str,  # 'PD' or 'CTC'
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute a single simulation trial for a specific experiment and controller."""
    params = dr.DeltaParams()
    dt = 0.002
    t_obs = 0.20  # Observation window for state estimation [s]

    # Setup experiment-specific conditions
    if exp_num == 1:
        # Experiment 1: Nominal
        v_belt = 0.20
        sensor_noise = 0.001
        with_obstacle = False
        payload = 0.0
        x_start = -0.35
    elif exp_num == 2:
        # Experiment 2: Faster Target & High Noise
        v_belt = 0.40
        sensor_noise = 0.008
        with_obstacle = False
        payload = 0.0
        x_start = -0.45
    elif exp_num == 3:
        # Experiment 3: Obstacle Avoidance & Payload Uncertainty
        v_belt = 0.22
        sensor_noise = 0.002
        with_obstacle = True
        payload = 0.25  # 250 g added payload in physical environment
        x_start = -0.38
    else:
        raise ValueError(f"Unknown experiment number: {exp_num}")

    obs_pos = (0.04, 0.0, -0.58)
    obs_radius = 0.04

    # Instantiate Environment
    env = dr.DeltaMuJoCoEnv(
        params=params,
        with_obstacle=with_obstacle,
        obstacle_pos=obs_pos,
        obstacle_radius=obs_radius,
        timestep=dt,
        pick_distance_threshold=0.035,
    )

    conveyor = dr.ConveyorBelt(velocity=v_belt, start_x=x_start, y_pos=0.0, z_pos=-0.65)
    sensor = dr.NoisySensor(std_dev=sensor_noise, seed=seed)
    ab_filter = dr.AlphaBetaFilter3D(alpha=0.35, beta=0.03, dt=dt)

    ctc_gains, pd_gains = dr.tune_gains_critical_damping(omega_n=25.0, zeta=1.0)
    if ctrl_type == "PD":
        controller = dr.PDController(params, pd_gains, with_gravity_comp=True)
    else:
        controller = dr.ComputedTorqueController(params, ctc_gains)

    # Initial Robot State: Home Position
    home_p, home_th = dr.home_position(params)
    env.reset(initial_joints=home_th)

    # Phase 1: Observation & Target Estimation
    sim_time = 0.0
    meas_history = []
    est_history = []
    true_history = []

    steps_obs = int(t_obs / dt)
    for _ in range(steps_obs):
        p_true = conveyor.true_position(sim_time)
        z_meas = sensor.measure(p_true)
        p_est, v_est = ab_filter.update(z_meas, dt=dt)

        meas_history.append(z_meas)
        est_history.append(p_est.copy())
        true_history.append(p_true.copy())

        # Robot holds home position
        th_cur, th_dot_cur = env.get_joint_state()
        if ctrl_type == "PD":
            tau = controller.compute_torque(home_th, np.zeros(3), th_cur, th_dot_cur, payload=0.0)
        else:
            tau = controller.compute_torque(home_th, np.zeros(3), np.zeros(3), th_cur, th_dot_cur, payload=0.0)
        env.step(tau, p_true, allow_pick=False)
        sim_time += dt

    # Phase 2: Interception Planning
    p_intercept, t_intercept = dr.predict_interception(
        robot_current_pos=home_p,
        target_est_pos=p_est,
        target_est_vel=v_est,
        params=params,
        t_start=sim_time,
        min_travel_time=0.60,
        constrain_plane_z=-0.65,
        constrain_plane_y=0.0,
    )

    if p_intercept is None:
        p_intercept = np.array([-0.10, 0.0, -0.65], dtype=np.float64)
        t_intercept = sim_time + 0.60

    travel_duration = max(0.20, t_intercept - sim_time)
    target_vel_contact = np.array([v_belt, 0.0, 0.0], dtype=np.float64)

    # Build Trajectory (RRT if obstacle present, otherwise Direct Quintic)
    rrt_waypoints = None
    if with_obstacle:
        obstacle = dr.SphereObstacle(center=np.array(obs_pos), radius=obs_radius, margin=0.02)
        rrt_planner = dr.RRTPlanner3D(params=params, obstacles=[obstacle], seed=seed)
        rrt_waypoints = rrt_planner.plan(home_p, p_intercept)
        if rrt_waypoints is None:
            rrt_waypoints = [home_p, home_p + np.array([0.0, -0.08, 0.0]), p_intercept]
        intercept_traj = dr.build_obstacle_avoidance_trajectory(
            rrt_waypoints, v_final=target_vel_contact, total_time=travel_duration
        )
    else:
        intercept_traj = dr.QuinticTrajectory(
            p0=home_p, pf=p_intercept, vf=target_vel_contact, duration=travel_duration
        )

    # Post-pick lift trajectory: lift 10 cm after interception
    p_lift = p_intercept + np.array([v_belt * 0.35, 0.0, 0.10])
    lift_traj = dr.QuinticTrajectory(
        p0=p_intercept, pf=p_lift, v0=target_vel_contact, vf=np.zeros(3), duration=0.35
    )

    # Phase 3: Trajectory Execution & Tracking
    t_traj_start = sim_time
    total_run_time = t_traj_start + travel_duration + 0.35

    times = []
    p_desired_list = []
    p_actual_list = []
    th_desired_list = []
    th_actual_list = []
    torques_list = []
    errors_cartesian = []
    errors_joint = []
    dist_to_target_list = []

    picked = False
    pick_time_res = None
    contact_dist = 999.0

    while sim_time < total_run_time:
        t_rel = sim_time - t_traj_start
        if t_rel <= travel_duration:
            pos_d, vel_d, acc_d = intercept_traj.sample(t_rel)
            is_lifting = False
        else:
            t_lift_rel = t_rel - travel_duration
            pos_d, vel_d, acc_d = lift_traj.sample(t_lift_rel)
            is_lifting = True

        # Map desired Cartesian kinematics to Joint Space
        th_d, ok_ik, _ = dr.inverse_kinematics(pos_d, params)
        if not ok_ik:
            th_d = home_th.copy()

        J = dr.jacobian(th_d, pos_d, params)
        J_inv = np.linalg.inv(J)
        th_dot_d = J_inv @ vel_d

        eps_fd = 1e-6
        th_fwd = th_d + th_dot_d * eps_fd
        th_bwd = th_d - th_dot_d * eps_fd
        J_fwd = dr.jacobian(th_fwd, dr.forward_kinematics(th_fwd, params), params)
        J_bwd = dr.jacobian(th_bwd, dr.forward_kinematics(th_bwd, params), params)
        J_dot = (J_fwd - J_bwd) / (2.0 * eps_fd)
        th_ddot_d = J_inv @ (acc_d - J_dot @ th_dot_d)

        # Control calculation (in Exp 3, nominal controller assumes nominal payload=0 to test robustness)
        th_cur, th_dot_cur = env.get_joint_state()
        ctrl_payload = 0.0 if exp_num == 3 else payload
        if ctrl_type == "PD":
            tau = controller.compute_torque(th_d, th_dot_d, th_cur, th_dot_cur, payload=ctrl_payload)
        else:
            tau = controller.compute_torque(th_d, th_dot_d, th_ddot_d, th_cur, th_dot_cur, payload=ctrl_payload)

        p_true_obj = conveyor.true_position(sim_time)
        step_res = env.step(tau, p_true_obj, allow_pick=(t_rel >= travel_duration * 0.85))

        ee_pos = step_res["ee_pos"]
        cart_err = float(np.linalg.norm(ee_pos - pos_d))
        joint_err = float(np.linalg.norm(th_cur - th_d))

        times.append(sim_time)
        p_desired_list.append(pos_d.copy())
        p_actual_list.append(ee_pos.copy())
        th_desired_list.append(th_d.copy())
        th_actual_list.append(th_cur.copy())
        torques_list.append(tau.copy())
        errors_cartesian.append(cart_err)
        errors_joint.append(joint_err)
        dist_to_target_list.append(step_res["dist_to_target"])

        # Capture distance at exact intercept moment
        if abs(t_rel - travel_duration) < dt:
            contact_dist = step_res["dist_to_target"]

        if step_res["is_picked"] and not picked:
            picked = True
            pick_time_res = sim_time - t_traj_start

        sim_time += dt

    # Performance metrics
    cart_errors_arr = np.array(errors_cartesian) * 1000.0  # mm
    joint_errors_deg = np.degrees(np.array(errors_joint))   # deg

    rmse_cart = float(np.sqrt(np.mean(cart_errors_arr**2)))
    max_err_cart = float(np.max(cart_errors_arr))
    rmse_joint = float(np.sqrt(np.mean(joint_errors_deg**2)))

    success = bool(picked or (contact_dist <= 0.035))
    final_pick_time = float(pick_time_res if pick_time_res is not None else travel_duration)

    return {
        "exp_num": exp_num,
        "ctrl_type": ctrl_type,
        "times": np.array(times),
        "p_desired": np.array(p_desired_list),
        "p_actual": np.array(p_actual_list),
        "th_desired": np.array(th_desired_list),
        "th_actual": np.array(th_actual_list),
        "torques": np.array(torques_list),
        "cart_errors_mm": cart_errors_arr,
        "joint_errors_deg": joint_errors_deg,
        "dist_to_target": np.array(dist_to_target_list),
        "rmse_cart_mm": rmse_cart,
        "max_err_cart_mm": max_err_cart,
        "rmse_joint_deg": rmse_joint,
        "contact_dist_mm": contact_dist * 1000.0,
        "success": success,
        "pick_time_s": final_pick_time,
        "collision": env.collision_detected,
        "meas_history": np.array(meas_history),
        "est_history": np.array(est_history),
        "true_history": np.array(true_history),
        "rrt_waypoints": rrt_waypoints,
        "obs_pos": obs_pos,
        "obs_radius": obs_radius,
    }


def run_experiment_suite(num_trials: int = 5) -> Dict[str, Any]:
    """Run all three experiments across multiple randomized trials."""
    print("=" * 70)
    print("RUNNING 3-DOF DELTA ROBOT CONTROL BENCHMARK EXPERIMENTS")
    print("=" * 70)

    results = {1: {"PD": [], "CTC": []}, 2: {"PD": [], "CTC": []}, 3: {"PD": [], "CTC": []}}

    for exp_num in [1, 2, 3]:
        exp_names = {
            1: "EXPERIMENT 1: Nominal Condition (v=0.20 m/s, clean sensor)",
            2: "EXPERIMENT 2: Fast Target & High Sensor Noise (v=0.40 m/s, noise=10mm)",
            3: "EXPERIMENT 3: Obstacle Avoidance (RRT) & Payload Mass (250g)",
        }
        print(f"\n>>> {exp_names[exp_num]}")

        for ctrl_type in ["PD", "CTC"]:
            print(f"  Running {ctrl_type} controller over {num_trials} trials...", end="", flush=True)
            for trial_idx in range(num_trials):
                seed = 100 * exp_num + trial_idx
                trial_data = run_trial(exp_num, ctrl_type, seed=seed)
                results[exp_num][ctrl_type].append(trial_data)
            print(" Done.")

    return results


def plot_experiment_comparisons(results: Dict[str, Any], output_dir: str = "results"):
    """Generate publication-ready plots for each experiment and summary charts."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True, "axes.grid": True})

    # 1. Experiment 1 Plot: Nominal Tracking
    fig1, axs1 = plt.subplots(2, 2, figsize=(11, 7))
    pd_res1 = results[1]["PD"][0]
    ctc_res1 = results[1]["CTC"][0]
    t1 = pd_res1["times"] - pd_res1["times"][0]

    # Tracking Error comparison
    axs1[0, 0].plot(t1, pd_res1["cart_errors_mm"], "r--", label="PD Tracking Error", lw=1.8)
    axs1[0, 0].plot(t1, ctc_res1["cart_errors_mm"], "b-", label="CTC Tracking Error", lw=1.8)
    axs1[0, 0].set_title("Exp 1 (Nominal): Cartesian Tracking Error")
    axs1[0, 0].set_xlabel("Time [s]")
    axs1[0, 0].set_ylabel("Error [mm]")
    axs1[0, 0].legend()

    # X, Z Position Profiles
    axs1[0, 1].plot(t1, pd_res1["p_desired"][:, 0], "k-", label="Desired X", lw=1.5, alpha=0.7)
    axs1[0, 1].plot(t1, pd_res1["p_actual"][:, 0], "r--", label="PD Actual X", lw=1.5)
    axs1[0, 1].plot(t1, ctc_res1["p_actual"][:, 0], "b:", label="CTC Actual X", lw=1.8)
    axs1[0, 1].set_title("End-Effector X Motion Profile")
    axs1[0, 1].set_xlabel("Time [s]")
    axs1[0, 1].set_ylabel("X Position [m]")
    axs1[0, 1].legend()

    # Joint Torques
    axs1[1, 0].plot(t1, pd_res1["torques"][:, 0], "r--", label="PD Tau Leg 0", lw=1.5)
    axs1[1, 0].plot(t1, ctc_res1["torques"][:, 0], "b-", label="CTC Tau Leg 0", lw=1.5)
    axs1[1, 0].set_title("Motor Joint Torque (Leg 0)")
    axs1[1, 0].set_xlabel("Time [s]")
    axs1[1, 0].set_ylabel("Torque [Nm]")
    axs1[1, 0].legend()

    # Distance to Moving Target
    axs1[1, 1].plot(t1, pd_res1["dist_to_target"] * 1000.0, "r--", label="PD Target Dist", lw=1.8)
    axs1[1, 1].plot(t1, ctc_res1["dist_to_target"] * 1000.0, "b-", label="CTC Target Dist", lw=1.8)
    axs1[1, 1].axhline(35.0, color="g", linestyle=":", label="Pick Threshold (35 mm)")
    axs1[1, 1].set_title("Distance to Moving Conveyor Object")
    axs1[1, 1].set_xlabel("Time [s]")
    axs1[1, 1].set_ylabel("Distance [mm]")
    axs1[1, 1].legend()

    fig1.savefig(os.path.join(output_dir, "exp1_nominal_comparison.png"), dpi=200)
    plt.close(fig1)

    # 2. Experiment 2 Plot: Fast Target & Noisy Filter
    fig2, axs2 = plt.subplots(2, 2, figsize=(11, 7))
    pd_res2 = results[2]["PD"][0]
    ctc_res2 = results[2]["CTC"][0]
    t2 = pd_res2["times"] - pd_res2["times"][0]

    # Alpha-Beta Filter Noisy vs Filtered vs True
    axs2[0, 0].plot(pd_res2["meas_history"][:, 0], "gray", alpha=0.5, label="Noisy Sensor Readings (X)")
    axs2[0, 0].plot(pd_res2["true_history"][:, 0], "k-", lw=1.5, label="True Target X")
    axs2[0, 0].plot(pd_res2["est_history"][:, 0], "g-", lw=2.0, label="Alpha-Beta Estimate X")
    axs2[0, 0].set_title("Exp 2: Alpha-Beta Filter State Estimation")
    axs2[0, 0].set_xlabel("Observation Sample")
    axs2[0, 0].set_ylabel("X Position [m]")
    axs2[0, 0].legend()

    # Tracking Error Comparison
    axs2[0, 1].plot(t2, pd_res2["cart_errors_mm"], "r--", label="PD Tracking Error", lw=1.8)
    axs2[0, 1].plot(t2, ctc_res2["cart_errors_mm"], "b-", label="CTC Tracking Error", lw=1.8)
    axs2[0, 1].set_title("Tracking Error under High Acceleration")
    axs2[0, 1].set_xlabel("Time [s]")
    axs2[0, 1].set_ylabel("Error [mm]")
    axs2[0, 1].legend()

    # Joint Errors
    axs2[1, 0].plot(t2, pd_res2["joint_errors_deg"], "r--", label="PD Joint Error Norm", lw=1.5)
    axs2[1, 0].plot(t2, ctc_res2["joint_errors_deg"], "b-", label="CTC Joint Error Norm", lw=1.5)
    axs2[1, 0].set_title("Joint Space Error Norm [deg]")
    axs2[1, 0].set_xlabel("Time [s]")
    axs2[1, 0].set_ylabel("Error [deg]")
    axs2[1, 0].legend()

    # Target Interception Distance
    axs2[1, 1].plot(t2, pd_res2["dist_to_target"] * 1000.0, "r--", label="PD Dist to Target", lw=1.8)
    axs2[1, 1].plot(t2, ctc_res2["dist_to_target"] * 1000.0, "b-", label="CTC Dist to Target", lw=1.8)
    axs2[1, 1].axhline(35.0, color="g", linestyle=":", label="Pick Threshold (35 mm)")
    axs2[1, 1].set_title("Fast Target Intercept Convergence")
    axs2[1, 1].set_xlabel("Time [s]")
    axs2[1, 1].set_ylabel("Distance [mm]")
    axs2[1, 1].legend()

    fig2.savefig(os.path.join(output_dir, "exp2_fast_noisy_comparison.png"), dpi=200)
    plt.close(fig2)

    # 3. Experiment 3 Plot: Obstacle Avoidance & Payload
    fig3, axs3 = plt.subplots(1, 2, figsize=(11, 5))
    pd_res3 = results[3]["PD"][0]
    ctc_res3 = results[3]["CTC"][0]
    t3 = pd_res3["times"] - pd_res3["times"][0]

    # Workspace X-Z Trajectory showing Obstacle Clearance
    axs3[0].plot(pd_res3["p_desired"][:, 0], pd_res3["p_desired"][:, 2], "k-", label="Planned Path (RRT)", lw=1.8)
    axs3[0].plot(pd_res3["p_actual"][:, 0], pd_res3["p_actual"][:, 2], "r--", label="PD Path", lw=1.5)
    axs3[0].plot(ctc_res3["p_actual"][:, 0], ctc_res3["p_actual"][:, 2], "b:", label="CTC Path", lw=2.0)

    # Draw Obstacle circle
    obs_circ = plt.Circle(
        (pd_res3["obs_pos"][0], pd_res3["obs_pos"][2]),
        pd_res3["obs_radius"],
        color="orange",
        alpha=0.6,
        label="Obstacle",
    )
    axs3[0].add_patch(obs_circ)
    axs3[0].set_title("Exp 3: X-Z Obstacle Avoidance Trajectory")
    axs3[0].set_xlabel("X Position [m]")
    axs3[0].set_ylabel("Z Position [m]")
    axs3[0].legend()

    # Tracking error with payload disturbance
    axs3[1].plot(t3, pd_res3["cart_errors_mm"], "r--", label="PD Error (250g Payload)", lw=1.8)
    axs3[1].plot(t3, ctc_res3["cart_errors_mm"], "b-", label="CTC Error (250g Payload)", lw=1.8)
    axs3[1].set_title("Tracking Error with Obstacle & Added Payload")
    axs3[1].set_xlabel("Time [s]")
    axs3[1].set_ylabel("Error [mm]")
    axs3[1].legend()

    fig3.savefig(os.path.join(output_dir, "exp3_obstacle_payload_comparison.png"), dpi=200)
    plt.close(fig3)

    # 4. Overall Benchmark Summary Bar Charts
    fig4, axs4 = plt.subplots(2, 2, figsize=(11, 8))
    categories = ["Exp 1: Nominal", "Exp 2: Fast & Noisy", "Exp 3: Obstacle & Load"]
    x_idx = np.arange(len(categories))
    bar_w = 0.35

    # Compute averages across trials
    rmse_pd = [np.mean([t["rmse_cart_mm"] for t in results[e]["PD"]]) for e in [1, 2, 3]]
    rmse_ctc = [np.mean([t["rmse_cart_mm"] for t in results[e]["CTC"]]) for e in [1, 2, 3]]

    max_e_pd = [np.mean([t["max_err_cart_mm"] for t in results[e]["PD"]]) for e in [1, 2, 3]]
    max_e_ctc = [np.mean([t["max_err_cart_mm"] for t in results[e]["CTC"]]) for e in [1, 2, 3]]

    succ_pd = [np.mean([100.0 if t["success"] else 0.0 for t in results[e]["PD"]]) for e in [1, 2, 3]]
    succ_ctc = [np.mean([100.0 if t["success"] else 0.0 for t in results[e]["CTC"]]) for e in [1, 2, 3]]

    time_pd = [np.mean([t["pick_time_s"] for t in results[e]["PD"]]) for e in [1, 2, 3]]
    time_ctc = [np.mean([t["pick_time_s"] for t in results[e]["CTC"]]) for e in [1, 2, 3]]

    # Bar 1: RMSE
    axs4[0, 0].bar(x_idx - bar_w/2, rmse_pd, bar_w, label="PD", color="#e74c3c")
    axs4[0, 0].bar(x_idx + bar_w/2, rmse_ctc, bar_w, label="CTC", color="#3498db")
    axs4[0, 0].set_title("Root Mean Square Error (RMSE) [mm]")
    axs4[0, 0].set_xticks(x_idx)
    axs4[0, 0].set_xticklabels(["Exp 1", "Exp 2", "Exp 3"])
    axs4[0, 0].set_ylabel("RMSE [mm]")
    axs4[0, 0].legend()

    # Bar 2: Max Error
    axs4[0, 1].bar(x_idx - bar_w/2, max_e_pd, bar_w, label="PD", color="#e74c3c")
    axs4[0, 1].bar(x_idx + bar_w/2, max_e_ctc, bar_w, label="CTC", color="#3498db")
    axs4[0, 1].set_title("Maximum Tracking Error [mm]")
    axs4[0, 1].set_xticks(x_idx)
    axs4[0, 1].set_xticklabels(["Exp 1", "Exp 2", "Exp 3"])
    axs4[0, 1].set_ylabel("Max Error [mm]")
    axs4[0, 1].legend()

    # Bar 3: Success Rate
    axs4[1, 0].bar(x_idx - bar_w/2, succ_pd, bar_w, label="PD", color="#e74c3c")
    axs4[1, 0].bar(x_idx + bar_w/2, succ_ctc, bar_w, label="CTC", color="#3498db")
    axs4[1, 0].set_title("Interception Success Rate [%]")
    axs4[1, 0].set_xticks(x_idx)
    axs4[1, 0].set_xticklabels(["Exp 1", "Exp 2", "Exp 3"])
    axs4[1, 0].set_ylabel("Success Rate [%]")
    axs4[1, 0].set_ylim(0, 115)
    axs4[1, 0].legend()

    # Bar 4: Pick Time
    axs4[1, 1].bar(x_idx - bar_w/2, time_pd, bar_w, label="PD", color="#e74c3c")
    axs4[1, 1].bar(x_idx + bar_w/2, time_ctc, bar_w, label="CTC", color="#3498db")
    axs4[1, 1].set_title("Average Pick Intercept Time [s]")
    axs4[1, 1].set_xticks(x_idx)
    axs4[1, 1].set_xticklabels(["Exp 1", "Exp 2", "Exp 3"])
    axs4[1, 1].set_ylabel("Pick Time [s]")
    axs4[1, 1].legend()

    fig4.savefig(os.path.join(output_dir, "overall_benchmark_summary.png"), dpi=200)
    plt.close(fig4)

    print(f"\nAll comparison figures successfully saved to '{output_dir}/'")


def print_results_summary_table(results: Dict[str, Any]):
    """Print clean formatted markdown comparison table."""
    print("\n" + "=" * 88)
    print("           EXPERIMENTAL RESULTS SUMMARY: PD vs COMPUTED-TORQUE CONTROL (CTC)")
    print("=" * 88)
    header = f"{'Experiment':<24} | {'Controller':<5} | {'RMSE [mm]':<10} | {'Max Err [mm]':<12} | {'Success [%]':<11} | {'Pick Time [s]':<12}"
    print(header)
    print("-" * 88)

    exp_labels = {
        1: "1. Nominal (v=0.20m/s)",
        2: "2. Fast & Noisy (0.4m/s)",
        3: "3. Obstacle + 250g Load",
    }

    for exp_num in [1, 2, 3]:
        for ctrl_type in ["PD", "CTC"]:
            trials = results[exp_num][ctrl_type]
            avg_rmse = np.mean([t["rmse_cart_mm"] for t in trials])
            avg_max_e = np.mean([t["max_err_cart_mm"] for t in trials])
            succ_pct = np.mean([100.0 if t["success"] else 0.0 for t in trials])
            avg_time = np.mean([t["pick_time_s"] for t in trials])

            label = exp_labels[exp_num] if ctrl_type == "PD" else ""
            row = f"{label:<24} | {ctrl_type:<5} | {avg_rmse:<10.2f} | {avg_max_e:<12.2f} | {succ_pct:<11.1f} | {avg_time:<12.3f}"
            print(row)
        print("-" * 88)


if __name__ == "__main__":
    benchmark_results = run_experiment_suite(num_trials=5)
    plot_experiment_comparisons(benchmark_results, output_dir="results")
    print_results_summary_table(benchmark_results)
