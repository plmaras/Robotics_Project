"""Unit Test Suite for 3-DOF Delta Robot Python Simulation.
"""

import unittest
import numpy as np

import delta_robot as dr


class TestKinematics(unittest.TestCase):
    def setUp(self):
        self.params = dr.DeltaParams()

    def test_fk_ik_roundtrip(self):
        test_points = [
            np.array([0.0, 0.0, -0.6136]),
            np.array([0.08, -0.05, -0.60]),
            np.array([-0.06, 0.07, -0.58]),
            np.array([0.05, 0.08, -0.63]),
        ]
        for pt in test_points:
            theta, ok, reason = dr.inverse_kinematics(pt, self.params)
            self.assertTrue(ok, f"IK failed for {pt}: {reason}")
            recovered_pt = dr.forward_kinematics(theta, self.params)
            self.assertIsNotNone(recovered_pt)
            err = np.linalg.norm(pt - recovered_pt)
            self.assertLess(err, 1e-6, f"FK/IK error {err} exceeded tolerance")

    def test_jacobian_analytical_vs_numerical(self):
        home_p, home_th = dr.home_position(self.params)
        J_ana = dr.jacobian(home_th, home_p, self.params)

        eps = 1e-6
        J_num = np.zeros((3, 3))
        for i in range(3):
            th_plus = home_th.copy()
            th_plus[i] += eps
            th_minus = home_th.copy()
            th_minus[i] -= eps
            p_plus = dr.forward_kinematics(th_plus, self.params)
            p_minus = dr.forward_kinematics(th_minus, self.params)
            J_num[:, i] = (p_plus - p_minus) / (2.0 * eps)

        diff = np.max(np.abs(J_ana - J_num))
        self.assertLess(diff, 1e-6)

    def test_conditioning_metric(self):
        home_p, home_th = dr.home_position(self.params)
        cond = dr.conditioning(home_th, home_p, self.params)
        self.assertGreater(cond, 0.1)
        self.assertLessEqual(cond, 1.0)


class TestDynamics(unittest.TestCase):
    def setUp(self):
        self.params = dr.DeltaParams()
        self.home_p, self.home_th = dr.home_position(self.params)

    def test_mass_matrix_positive_definite(self):
        M = dr.mass_matrix(self.home_th, self.params)
        self.assertEqual(M.shape, (3, 3))
        eigvals = np.linalg.eigvals(M)
        self.assertTrue(np.all(eigvals > 0), "Mass matrix is not positive definite")

    def test_forward_inverse_dynamics_consistency(self):
        theta = self.home_th
        thetadot = np.array([0.5, -0.3, 0.2])
        thetaddot = np.array([1.2, -0.8, 1.5])

        tau = dr.inverse_dynamics(theta, thetadot, thetaddot, self.params)
        acc_rec = dr.forward_dynamics(theta, thetadot, tau, self.params)

        err = np.linalg.norm(thetaddot - acc_rec)
        self.assertLess(err, 1e-6, f"Forward/Inverse dynamics mismatch: {err}")


class TestEstimation(unittest.TestCase):
    def test_alpha_beta_filter_convergence(self):
        conveyor = dr.ConveyorBelt(velocity=0.20, start_x=-0.30, y_pos=0.0, z_pos=-0.65)
        sensor = dr.NoisySensor(std_dev=0.002, seed=123)
        filt = dr.AlphaBetaFilter3D(alpha=0.35, beta=0.03, dt=0.002)

        for s in range(250):
            t = s * 0.002
            p_true = conveyor.true_position(t)
            z_meas = sensor.measure(p_true)
            p_est, v_est = filt.update(z_meas, dt=0.002)

        self.assertAlmostEqual(v_est[0], 0.20, delta=0.06)
        self.assertAlmostEqual(v_est[1], 0.00, delta=0.06)
        self.assertAlmostEqual(v_est[2], 0.00, delta=0.06)


class TestPlanning(unittest.TestCase):
    def setUp(self):
        self.params = dr.DeltaParams()

    def test_quintic_boundary_conditions(self):
        p0 = np.array([0.0, 0.0, -0.60])
        pf = np.array([0.1, -0.05, -0.65])
        v0 = np.array([0.0, 0.0, 0.0])
        vf = np.array([0.2, 0.0, 0.0])
        T = 0.8
        traj = dr.QuinticTrajectory(p0, pf, v0=v0, vf=vf, duration=T)

        p_start, v_start, a_start = traj.sample(0.0)
        p_end, v_end, a_end = traj.sample(T)

        np.testing.assert_allclose(p_start, p0, atol=1e-6)
        np.testing.assert_allclose(v_start, v0, atol=1e-6)
        np.testing.assert_allclose(p_end, pf, atol=1e-6)
        np.testing.assert_allclose(v_end, vf, atol=1e-6)

    def test_rrt_obstacle_avoidance(self):
        obs = dr.SphereObstacle(center=np.array([0.06, 0.0, -0.58]), radius=0.04, margin=0.01)
        rrt = dr.RRTPlanner3D(self.params, obstacles=[obs], seed=42)

        start = np.array([0.0, 0.0, -0.6136])
        goal = np.array([0.12, 0.0, -0.65])

        path = rrt.plan(start, goal)
        self.assertIsNotNone(path, "RRT should find a valid path")
        self.assertGreaterEqual(len(path), 2)
        # Verify no segment collides
        for i in range(len(path) - 1):
            self.assertFalse(obs.is_segment_colliding(path[i], path[i + 1]))


class TestMuJoCoEnv(unittest.TestCase):
    def test_env_step_and_reset(self):
        env = dr.DeltaMuJoCoEnv()
        theta, thetadot = env.reset()
        self.assertEqual(len(theta), 3)
        self.assertEqual(len(thetadot), 3)

        res = env.step(np.zeros(3), np.array([-0.3, 0.0, -0.65]))
        self.assertIn("ee_pos", res)
        self.assertIn("dist_to_target", res)
        self.assertFalse(res["is_picked"])


if __name__ == "__main__":
    unittest.main()
