"""Tests for the current-penalty comparison workflow."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from nominal import CONFIG
from smooth_current import compare, run, window_metrics


def trace_result(commands, capture=0.5):
    trace = [{"command_current_A": 0.0}]
    for command in commands:
        trace.extend([{"command_current_A": command}] * CONFIG.action_repeat)
    return {"trace": trace, "capture_time": capture, "success": True, "unsafe": False, "finish_time": 2.0}


class SmoothCurrentTests(unittest.TestCase):
    def test_command_sampling_and_initial_change(self):
        result = trace_result([0.2] * 1000)
        full = window_metrics(result, CONFIG, "full_episode")
        self.assertEqual(full["command_samples"], 1000)
        self.assertAlmostEqual(full["total_variation_A"], 0.2)
        self.assertAlmostEqual(full["change_rms_A"], 0.2 / np.sqrt(1000))
        first = window_metrics(result, CONFIG, "first_second")
        self.assertEqual(first["command_samples"], 200)
        self.assertAlmostEqual(first["change_rms_A"], 0.2 / np.sqrt(200))

    def test_high_frequency_and_missing_capture(self):
        result = trace_result([0.1, -0.1] * 500, capture=None)
        metrics = window_metrics(result, CONFIG, "full_episode")
        self.assertAlmostEqual(metrics["hf_power_fraction"], 1.0)
        self.assertEqual(window_metrics(result, CONFIG, "capture_window")["command_samples"], 0)
        with self.assertRaises(ValueError):
            window_metrics(result, CONFIG, "unknown")

    def test_capture_window_and_boundary_change(self):
        result = trace_result([0.0] * 60 + [0.2] * 940)
        metrics = window_metrics(result, CONFIG, "capture_window")
        self.assertEqual(metrics["command_samples"], 80)
        self.assertAlmostEqual(metrics["total_variation_A"], 0.2)

    def test_comparison_generates_paired_report(self):
        with TemporaryDirectory() as directory:
            job = Path(directory)
            with patch("smooth_current.SEEDS", range(80001, 80003)), \
                 patch("smooth_current.SAC.load"), \
                 patch("smooth_current.run_episode", return_value=trace_result([0.1, -0.1] * 500)):
                compare(job)
            summary = json.loads((job / "comparison.json").read_text())
            self.assertEqual(summary["candidate"]["successes"], 2)
            self.assertEqual(summary["baseline"]["windows"]["capture_window"]["episodes"], 2)
            self.assertIn("reliability gate retained", (job / "FINDINGS.md").read_text())
            self.assertTrue((job / "comparison" / "current_windows.csv").is_file())

    def test_dry_run_and_failure_handling(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            with patch("smooth_current.validate_baseline"), patch("smooth_current.run_nominal") as training:
                run(job)
                self.assertFalse(job.exists())
                training.assert_not_called()
            with patch("smooth_current.validate_baseline"), \
                 patch("smooth_current.shutil.copy2"), \
                 patch("smooth_current.file_sha256", return_value="test"), \
                 patch("smooth_current.run_nominal", side_effect=RuntimeError("failed training")), \
                 patch("smooth_current.compare") as comparison:
                with self.assertRaisesRegex(RuntimeError, "failed training"):
                    run(job, execute=True)
                comparison.assert_not_called()
            self.assertEqual(json.loads((job / "status.json").read_text())["state"], "failed")

    def test_unattended_sequence_completes(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            calls = []

            def training(path, **kwargs):
                calls.append("training_and_evaluation")
                self.assertEqual(path, job / "candidate")
                self.assertEqual(kwargs, {"execute": True, "action_change_weight": 5.0})

            def comparison(path):
                calls.append("comparison")
                self.assertEqual(path, job)
                self.assertEqual(json.loads((job / "status.json").read_text())["phase"], "jitter_comparison")
                (job / "FINDINGS.md").write_text("Completed comparison")

            with patch("smooth_current.validate_baseline"), \
                 patch("smooth_current.shutil.copy2"), \
                 patch("smooth_current.file_sha256", return_value="test"), \
                 patch("smooth_current.run_nominal", side_effect=training), \
                 patch("smooth_current.compare", side_effect=comparison):
                run(job, execute=True)
            self.assertEqual(calls, ["training_and_evaluation", "comparison"])
            self.assertEqual(json.loads((job / "status.json").read_text())["state"], "completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
