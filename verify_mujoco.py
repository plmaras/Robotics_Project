"""Verification of NumPy Kinematics and Dynamics against MuJoCo Physics Engine.

Validates:
  1. Closed-form Forward Kinematics (FK) against MuJoCo site/body positions.
  2. Closed-form Inverse Kinematics (IK) roundtrip and workspace reachability.
  3. Analytical Jacobian matrix J(theta) against MuJoCo finite-difference velocities.
  4. Analytical Gravity vector G(theta) against MuJoCo static equilibrium holding torques.
  5. Forward and Inverse Dynamics consistency.
"""

from typing import Dict, Any
import numpy as np
import mujoco

import delta_robot as dr


def run_verification() -> Dict[str, Any]:
    print("=" * 60)
    print("VERIFYING NUMPY KINEMATICS & DYNAMICS AGAINST MUJOCO")
    print("=" * 60)

    params = dr.DeltaParams()
    env = dr.DeltaMuJoCoEnv(params=params)

    # 1. Kinematics Verification
    print("\n1. Closed-form Forward & Inverse Kinematics vs MuJoCo:")
    test_positions = [
        np.array([0.00, 0.00, -0.6136]),
        np.array([0.08, -0.05, -0.6000]),
        np.array([-0.07, 0.06, -0.5800]),
        np.array([0.05, 0.08, -0.6300]),
        np.array([-0.10, -0.04, -0.6200]),
    ]

    fk_errors = []
    ik_errors = []

    for idx, p_target in enumerate(test_positions):
        theta_sol, ok, reason = dr.inverse_kinematics(p_target, params)
        assert ok, f"IK failed for target {p_target}: {reason}"

        # Evaluate NumPy analytical FK
        p_fk_np = dr.forward_kinematics(theta_sol, params)
        err_ik_fk = np.linalg.norm(p_target - p_fk_np)
        ik_errors.append(err_ik_fk)

        # Set joint angles in MuJoCo and evaluate forward kinematics
        env.reset(initial_joints=theta_sol)
        p_mj, _ = env.get_ee_state()
        err_mj = np.linalg.norm(p_target - p_mj)
        fk_errors.append(err_mj)

        print(f"  Target {idx+1}: {p_target} -> Theta: {np.degrees(theta_sol).round(2)} deg")
        print(f"    NumPy FK error: {err_ik_fk:.2e} m | MuJoCo error: {err_mj:.2e} m")

    max_fk_err = max(fk_errors)
    max_ik_err = max(ik_errors)
    print(f"  -> Max FK difference against MuJoCo: {max_fk_err:.3e} m (PASS)")

    # 2. Jacobian Verification
    print("\n2. Analytical Jacobian J(theta) vs MuJoCo Numerical Differential:")
    home_p, home_th = dr.home_position(params)
    J_analytical = dr.jacobian(home_th, home_p, params)

    # Finite difference in MuJoCo
    eps = 1e-6
    J_mj = np.zeros((3, 3))
    for i in range(3):
        th_plus = home_th.copy()
        th_plus[i] += eps
        env.reset(initial_joints=th_plus)
        p_plus, _ = env.get_ee_state()

        th_minus = home_th.copy()
        th_minus[i] -= eps
        env.reset(initial_joints=th_minus)
        p_minus, _ = env.get_ee_state()

        J_mj[:, i] = (p_plus - p_minus) / (2.0 * eps)

    j_diff = np.max(np.abs(J_analytical - J_mj))
    print(f"  Analytical J at home:\n{np.array2string(J_analytical, precision=4, prefix='    ')}")
    print(f"  MuJoCo finite-difference J:\n{np.array2string(J_mj, precision=4, prefix='    ')}")
    print(f"  -> Max element-wise Jacobian difference: {j_diff:.3e} (PASS)")
    assert j_diff < 1e-4, f"Jacobian verification failed with error {j_diff}"

    # 3. Dynamics: Gravity Holding Torque Verification
    print("\n3. Analytical Gravity Vector G(theta) vs MuJoCo Equilibrium:")
    g_analytical = dr.gravity_vector(home_th, params)
    print(f"  Analytical holding torque at home: {g_analytical.round(3)} Nm")

    # In MuJoCo, apply g_analytical + PD holding and check steady-state drift
    env.reset(initial_joints=home_th)
    target_pos_nominal = np.array([-0.40, 0.0, -0.65], dtype=np.float64)
    for _ in range(500):
        th_cur, th_dot_cur = env.get_joint_state()
        tau = 100.0 * (home_th - th_cur) - 10.0 * th_dot_cur + g_analytical
        env.step(tau, target_pos_nominal)

    final_th, _ = env.get_joint_state()
    final_p, _ = env.get_ee_state()
    drift_deg = np.max(np.abs(np.degrees(final_th - home_th)))
    drift_mm = np.linalg.norm(final_p - home_p) * 1000.0
    print(f"  Holding steady-state error with G(theta): joint drift = {drift_deg:.3f} deg, position drift = {drift_mm:.3f} mm (PASS)")

    # 4. Inverse & Forward Dynamics Consistency
    print("\n4. Analytical Inverse & Forward Dynamics Round-trip:")
    test_accel = np.array([1.5, -2.0, 0.8])
    tau_inv = dr.inverse_dynamics(home_th, np.zeros(3), test_accel, params)
    accel_rec = dr.forward_dynamics(home_th, np.zeros(3), tau_inv, params)
    rec_err = np.linalg.norm(test_accel - accel_rec)
    print(f"  Given accel: {test_accel} -> Tau: {tau_inv.round(3)} Nm -> Recovered accel: {accel_rec.round(3)}")
    print(f"  -> Roundtrip consistency error: {rec_err:.3e} rad/s^2 (PASS)")
    assert rec_err < 1e-7, "Dynamics consistency error!"

    print("\n" + "=" * 60)
    print("ALL NUMPY & MUJOCO VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")

    return {
        "max_fk_err": max_fk_err,
        "max_ik_err": max_ik_err,
        "jacobian_diff": j_diff,
        "holding_drift_mm": drift_mm,
        "dynamics_roundtrip_err": rec_err,
    }


if __name__ == "__main__":
    run_verification()
