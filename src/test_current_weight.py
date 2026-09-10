from dataclasses import asdict, replace
import unittest

import numpy as np

from furuta_env import FurutaConfig, matlab_reward
from nominal import CONFIG


class CurrentWeightTests(unittest.TestCase):
    def test_default_and_legacy_config_preserve_reward(self):
        legacy = asdict(CONFIG)
        legacy.pop("current_weight")
        self.assertEqual(FurutaConfig(**legacy), CONFIG)
        state = np.array([0.2, np.pi + 0.1, 0.3, -0.4])
        action, previous = 0.25, -0.1
        expected = 1 - .1 * (1.5 * .2**2 + .1**2 + .015 * .3**2
                             + .01 * .4**2 + action**2 + 3 * (action - previous)**2)
        self.assertAlmostEqual(matlab_reward(state, action, previous, CONFIG), expected)

    def test_only_current_cost_changes(self):
        candidate = replace(CONFIG, current_weight=1.5)
        changed = [name for name, value in asdict(CONFIG).items() if value != asdict(candidate)[name]]
        self.assertEqual(changed, ["current_weight"])
        for action in (-1.0, -0.25, 0.0, 0.5, 1.0):
            state = np.array([0.1, np.pi, 0.2, 0.3])
            difference = matlab_reward(state, action, .2, candidate) - matlab_reward(state, action, .2, CONFIG)
            self.assertAlmostEqual(difference, -CONFIG.reward_scale * .5 * action**2)

    def test_invalid_current_weight_rejected(self):
        for value in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                replace(CONFIG, current_weight=value)


if __name__ == "__main__":
    unittest.main()
