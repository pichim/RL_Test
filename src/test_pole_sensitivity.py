import unittest

import numpy as np

from analyze_damping import jacobian
from pole_sensitivity import closed_loop, match_poles, metrics, qualifies, track_poles


class PoleSensitivityTests(unittest.TestCase):
    def test_matrix_matches_augmented_step(self):
        plant_a = np.diag([.8, .85, .9, .95])
        plant_b = np.array([[.1], [.2], [.3], [.4]])
        gain = np.array([.2, -.5, .3, -.7, .4])
        def step(state):
            action = gain @ state
            return np.r_[plant_a @ state[:4] + plant_b[:, 0] * action, action]
        np.testing.assert_allclose(closed_loop(plant_a, plant_b, gain), jacobian(step, np.zeros(5), 1e-6))
        poles = np.linalg.eigvals(closed_loop(plant_a, plant_b, gain)).astype(complex)
        np.testing.assert_allclose(track_poles(plant_a, plant_b, gain, gain, poles), poles)

    def test_assignment_ignores_eigenvalue_order(self):
        reference = np.array([.9, .4 + .2j, .4 - .2j])
        np.testing.assert_allclose(match_poles(reference, reference[[2, 0, 1]]), reference)

    def test_screen_rejects_instability_and_slow_mode_regression(self):
        poles = np.exp(.005 * np.array([-200 + 333j, -200 - 333j, -7 + 2j, -7 - 2j, -3.6]))
        reference = metrics(poles, [0, 1])
        improved = np.exp(.005 * np.array([-100 + 100j, -100 - 100j, -7 + 2j, -7 - 2j, -3.5]))
        self.assertTrue(qualifies(metrics(improved, [0, 1]), reference))
        for slow in (-1.0, 1.0):
            changed = improved.copy()
            changed[-1] = np.exp(.005 * slow)
            self.assertFalse(qualifies(metrics(changed, [0, 1]), reference))
        real_pair = improved.copy()
        real_pair[:2] = [.5, .6]
        self.assertFalse(qualifies(metrics(real_pair, [0, 1]), reference))
        zero_pole = improved.copy()
        zero_pole[-1] = 0
        zero_metrics = metrics(zero_pole, [0, 1])
        self.assertFalse(zero_metrics["finite_log"])
        self.assertFalse(qualifies(zero_metrics, reference))


if __name__ == "__main__":
    unittest.main()
