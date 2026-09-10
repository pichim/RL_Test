from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from furuta_env import balanced_state_ready, matlab_reward, unsafe
from nominal import CONFIG, run


class NoCenteringTests(unittest.TestCase):
    def test_zero_centering_preserves_safety_and_velocity_checks(self):
        config = replace(CONFIG, arm_angle_weight=0.0, balance_theta1=CONFIG.arm_angle_limit)
        centered = np.array([0.0, np.pi, 0.0, 0.0])
        offset = np.array([1.0, np.pi, 0.0, 0.0])
        self.assertEqual(matlab_reward(centered, 0.0, 0.0, config), matlab_reward(offset, 0.0, 0.0, config))
        self.assertTrue(balanced_state_ready(offset, 0.0, config))
        self.assertFalse(balanced_state_ready(offset, 0.0, CONFIG))
        offset[2] = 0.51
        self.assertFalse(balanced_state_ready(offset, 0.0, config))
        offset[2] = 0.0
        offset[0] = config.arm_angle_limit + 0.001
        self.assertTrue(unsafe(offset, config))
        self.assertFalse(balanced_state_ready(offset, 0.0, config))

    def test_invalid_centering_weights_rejected(self):
        for value in (-1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                replace(CONFIG, arm_angle_weight=value)

    def test_nominal_runner_uses_explicit_task_configuration(self):
        config = replace(CONFIG, arm_angle_weight=0.0, balance_theta1=CONFIG.arm_angle_limit)
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            with patch("nominal.train", side_effect=RuntimeError("probe")) as training:
                with self.assertRaisesRegex(RuntimeError, "probe"):
                    run(job, True, 4.0, task_config=config)
            expected = replace(config, action_change_weight=4.0)
            self.assertEqual(training.call_args.kwargs["config"], expected)
            saved = json.loads((job / "experiment.json").read_text())
            self.assertEqual(saved["config"]["arm_angle_weight"], 0.0)
            self.assertEqual(saved["config"]["balance_theta1"], CONFIG.arm_angle_limit)
            self.assertEqual(saved["training"]["gamma"], 0.99)

    def test_dual_scoring_retains_arm_speed_and_safety(self):
        from no_centering import TASK_CONFIG, score

        row = {"time_s": 0.0, "theta1_rad": 1.0, "upright_error_rad": 0.0,
               "omega1_rad_s": 0.0, "omega2_rad_s": 0.0, "unsafe": 0}
        trace = [{**row, "time_s": index * .001} for index in range(5001)]
        result = {"trace": trace, "unsafe": False}
        metrics = score(result, TASK_CONFIG)
        self.assertEqual(metrics["free_success_5s"], 1)
        self.assertEqual(metrics["centered_success_5s"], 0)
        trace[4500]["omega1_rad_s"] = 0.51
        self.assertEqual(score(result, TASK_CONFIG)["free_success_5s"], 0)
        result["unsafe"] = True
        self.assertEqual(score(result, TASK_CONFIG)["free_success_end"], 0)

    def test_status_lock_falls_back_to_event(self):
        from no_centering import status

        with TemporaryDirectory() as directory:
            job = Path(directory)
            with patch("no_centering.atomic_write_json", side_effect=[PermissionError("locked"), None]) as writer:
                status(job, "completed", "report")
            self.assertEqual(writer.call_count, 2)
            self.assertTrue(writer.call_args.args[0].name.startswith("status_event_"))

    def test_pipeline_order_and_fresh_seeds(self):
        from no_centering import TASK_CONFIG, experiment_cases, run as experiment

        seeds = [case["seed"] for case in experiment_cases() if case["suite"] == "fresh_long"]
        self.assertEqual(seeds, list(range(380000, 380100)))
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            calls = []

            def training(path, **kwargs):
                calls.append("train")
                self.assertEqual(kwargs["task_config"], TASK_CONFIG)
                self.assertEqual(kwargs["action_change_weight"], 4.0)

            with patch("no_centering.shutil.copy2"), patch("no_centering.file_sha256", return_value="hash"), \
                  patch("no_centering.Path.read_text", side_effect=[json.dumps(asdict(CONFIG)), json.dumps(asdict(replace(CONFIG, action_change_weight=4.0)))]), \
                  patch("no_centering.Path.is_file", return_value=True), patch("no_centering.diagnose", return_value=[]), \
                 patch("no_centering.run_nominal", side_effect=training), \
                 patch("no_centering.compare", side_effect=lambda *args: calls.append("compare")):
                experiment(job, True)
            self.assertEqual(calls, ["train", "compare"])
            self.assertEqual(json.loads((job / "completed.json").read_text())["state"], "completed")


if __name__ == "__main__":
    unittest.main()
