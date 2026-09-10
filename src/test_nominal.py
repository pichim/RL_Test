"""Focused checks for the single-run nominal SAC workflow."""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from furuta_env import DEFAULT_CONFIG
from nominal import CONFIG, run


class NominalTests(unittest.TestCase):
    def test_nominal_config_preserves_task_without_uncertainty(self):
        settings = asdict(CONFIG)
        for field, value in settings.items():
            if "randomization" in field or "current_filter" in field or "dead_time" in field:
                self.assertEqual(value, 0, field)
            else:
                self.assertEqual(value, getattr(DEFAULT_CONFIG, field), field)

    def test_dry_run_and_existing_output_protection(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            with patch("nominal.train") as training:
                run(job)
                training.assert_not_called()
            self.assertFalse(job.exists())
            job.mkdir()
            with self.assertRaises(FileExistsError):
                run(job, execute=True)

    def test_training_then_nominal_sac_evaluation(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            def evaluation(**kwargs):
                kwargs["results_dir"].mkdir()
                (kwargs["results_dir"] / "summary.csv").write_text(
                    "success_count,episodes,unsafe_count,capture_count\n100,100,0,100\n"
                )
            with patch("nominal.train") as training, patch("nominal.evaluate", side_effect=evaluation) as evaluate:
                run(job, execute=True)
            self.assertEqual(training.call_count, 1)
            self.assertEqual(training.call_args.kwargs["gamma"], 0.99)
            self.assertEqual(training.call_args.kwargs["actor_network"], [64, 64])
            self.assertEqual(training.call_args.kwargs["total_timesteps"], 2_000_000)
            self.assertEqual(evaluate.call_args.kwargs["mode"], "nominal")
            self.assertEqual(evaluate.call_args.kwargs["controller"], "sac")
            self.assertEqual(evaluate.call_args.kwargs["base_seed"], 80000)
            self.assertEqual(json.loads((job / "status.json").read_text())["state"], "completed")
            self.assertIn("100/100", (job / "FINDINGS.md").read_text())

    def test_training_failure_stops_evaluation(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            with patch("nominal.train", side_effect=RuntimeError("training failed")), patch("nominal.evaluate") as evaluate:
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    run(job, execute=True)
                evaluate.assert_not_called()
            self.assertEqual(json.loads((job / "status.json").read_text())["state"], "failed")

    def test_smooth_current_changes_only_reward_weight(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            with patch("nominal.train", side_effect=RuntimeError("probe")) as training:
                with self.assertRaisesRegex(RuntimeError, "probe"):
                    run(job, execute=True, action_change_weight=5.0)
            config = training.call_args.kwargs["config"]
            self.assertEqual(config.action_change_weight, 5.0)
            original = asdict(CONFIG)
            original["action_change_weight"] = 5.0
            self.assertEqual(asdict(config), original)
            recorded = json.loads((job / "experiment.json").read_text())
            self.assertEqual(recorded["config"]["action_change_weight"], 5.0)

    def test_invalid_weight_creates_no_output(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            for weight in (-1.0, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    run(job, execute=True, action_change_weight=weight)
                self.assertFalse(job.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
