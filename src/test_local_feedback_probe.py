import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from analyze_damping import jacobian
from local_feedback_probe import FACTOR, INNER, OUTER, LocalFeedbackPolicy, cases, correction, report, run


class ConstantPolicy:
    def __init__(self, action=0.0):
        self.action = action

    def predict(self, observation, deterministic=True):
        return np.full(np.asarray(observation).shape[:-1] + (1,), self.action), None


def observation(state):
    return np.array([np.sin(state[0]), np.cos(state[0]), state[2] / 30,
                     np.sin(state[1]), np.cos(state[1]), state[3] / 30, 0])


class LocalFeedbackProbeTests(unittest.TestCase):
    def test_local_derivative_and_zero_equilibrium_correction(self):
        point = np.array([.16, 0, 0, 0, 0])
        derivative = -1.9833
        gradient = jacobian(lambda state: np.atleast_1d(correction(state, .16, derivative)), point, 1e-6)
        np.testing.assert_allclose(gradient, [[0, 0, 0, (FACTOR - 1) * derivative, 0]], atol=1e-10)
        self.assertEqual(correction(point, .16, derivative), 0)

    def test_taper_bounds_and_wrapper(self):
        wrapper = LocalFeedbackPolicy(ConstantPolicy(.1), .16, -2)
        point = np.array([.16, 0, 0, .1])
        action, _ = wrapper.predict(observation(point))
        self.assertAlmostEqual(action[0], .15)
        for axis in range(4):
            outside = point.copy()
            outside[axis] = ([.16, 0, 0, 0][axis] + OUTER[axis] + .01)
            np.testing.assert_allclose(wrapper.predict(observation(outside))[0], [.1])
        middle = point.copy()
        middle[0] += (INNER[0] + OUTER[0]) / 2
        self.assertAlmostEqual(correction(middle, .16, -2), .025)
        saturated = LocalFeedbackPolicy(ConstantPolicy(.99), .16, -2)
        self.assertEqual(saturated.predict(observation(point))[0][0], 1)
        batch = np.stack([observation(point), observation(point)])
        self.assertEqual(wrapper.predict(batch)[0].shape, (2, 1))

    def test_paired_case_coverage(self):
        planned = cases(.16)
        self.assertEqual(len(planned), 25)
        self.assertEqual(len({case["name"] for case in planned}), 25)
        offsets = np.array([case["state"] for case in planned[1:]]) - [.16, np.pi, 0, 0]
        np.testing.assert_allclose(offsets.sum(axis=0), np.zeros(4), atol=1e-14)

    def test_dry_run_and_existing_output_guard(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "probe"
            with patch("local_feedback_probe.prepare", return_value=(None, None, None, {"equilibrium_arm_rad": .16})), \
                    patch("local_feedback_probe.run_prepared_episode") as simulate:
                run(output=output)
                self.assertFalse(output.exists())
                simulate.assert_not_called()
                output.mkdir()
                with self.assertRaises(FileExistsError):
                    run(output=output, execute=True)

    def test_report_has_no_speed_gate_and_pairs_complete_windows(self):
        episodes = [{"controller": label, "unsafe": 0, "completed_horizon": 1,
                     "centered_success_end": 1, "centered_lost_after_5s": 0}
                    for label in ("original", "modified")]
        currents = [{"controller": "original", "case": "only_one", "window": "first_5s",
                     "current_rms_A": .1, "change_rms_A": .1, "total_variation_A": .1, "hf_rms_A": .1}]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            report(output, episodes, currents)
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["settling_speed_gate"])
            self.assertEqual(summary["current_windows"]["first_5s"]["paired_cases"], 0)


if __name__ == "__main__":
    unittest.main()
