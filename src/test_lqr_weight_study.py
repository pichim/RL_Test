import unittest

import numpy as np

from furuta_env import matlab_reward
from lqr_weight_study import (
    BASE_WEIGHTS, CONFIG, augmented_problem, model_checks, plant_matrices,
    pole_distance, pole_metrics, scenarios, solve_problem,
)


class LQRWeightStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.transition, cls.action_input = plant_matrices()

    def test_endpoint_cost_matches_environment_reward(self):
        dynamics, control, quadratic, cross, effort = augmented_problem(
            self.transition, self.action_input, BASE_WEIGHTS,
        )
        generator = np.random.default_rng(17)
        for _ in range(20):
            state = generator.normal(0.0, 0.02, 5)
            action = generator.normal(0.0, 0.02, 1)
            endpoint = (dynamics @ state + control @ action)[:4]
            physical = endpoint + np.array([0.0, np.pi, 0.0, 0.0])
            reward = matlab_reward(physical, action[0], state[4], CONFIG)
            cost = float(state @ quadratic @ state + 2 * state @ cross @ action + action @ effort @ action)
            self.assertAlmostEqual(cost, (CONFIG.constraint_reward_weight - reward) / CONFIG.reward_scale, places=12)

    def test_riccati_solution_and_sweep_stability(self):
        for gamma in (1.0, 0.99):
            for name, weights in scenarios():
                with self.subTest(gamma=gamma, scenario=name):
                    problem = augmented_problem(self.transition, self.action_input, weights)
                    gain, closed, value, residual = solve_problem(problem, gamma)
                    self.assertLess(residual, 1e-8)
                    self.assertGreater(np.linalg.eigvalsh(value).min(), 0.0)
                    self.assertLess(max(abs(np.linalg.eigvals(closed))), 1.0)
                    dynamics, control, _, cross, effort = problem
                    np.testing.assert_allclose(
                        (effort + gamma * control.T @ value @ control) @ gain,
                        cross.T + gamma * control.T @ value @ dynamics, atol=1e-10,
                    )

    def test_exact_zoh_matches_nonlinear_rk4(self):
        checks = model_checks(self.transition, self.action_input)
        self.assertLess(checks["zoh_vs_rk4_jacobian_max_error"], 1e-6)

    def test_matching_reports_extra_mode(self):
        target = pole_metrics(np.diag([0.9, 0.8, 0.7, 0.6]), 0.005)
        candidate = pole_metrics(np.diag([0.6, 0.7, 0.8, 0.9, 0.5]), 0.005)
        distance, unmatched = pole_distance(candidate, target)
        self.assertEqual(distance, 0.0)
        self.assertEqual(len(unmatched), 1)
        self.assertAlmostEqual(unmatched[0]["discrete_abs"], 0.5)


if __name__ == "__main__":
    unittest.main()
