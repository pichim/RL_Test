import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from weight4 import CONFIG, LABELS, build_cases, make_episode, run, summarize, write_report


class Weight4Tests(unittest.TestCase):
    def test_neighborhood_preserves_center_and_reset_support(self):
        cases = build_cases()
        neighborhood = [case for case in cases if case["suite"] == "neighborhood"]
        center = next(case for case in neighborhood if case["name"] == "offset_0_0_0_0")
        np.testing.assert_allclose(center["state"], make_episode(180051, CONFIG, False, "training").x, atol=1e-14)
        for case in neighborhood:
            state = case["state"]
            self.assertLessEqual(abs(state[0]), CONFIG.reset_theta1_half_range)
            self.assertLessEqual(abs(state[2]), CONFIG.reset_omega1_half_range)
            self.assertLessEqual(abs(state[3]), CONFIG.reset_omega2_half_range)
        self.assertEqual(len([case for case in cases if case["suite"] == "fresh_long"]), 100)
        self.assertEqual(len({case["name"] for case in neighborhood}), len(neighborhood))

    def test_summary_pairs_windows_and_flags_failures(self):
        episodes = []
        currents = []
        for label in LABELS:
            for index in range(2):
                success = int(not (label == "weight4" and index == 1))
                episodes.append({"model": label, "suite": "fresh_long", "case": str(index),
                                 "success_at_5s": success, "success_at_end": success,
                                 "unsafe": 1 - success, "balance_lost_after_5s": 0,
                                 "finish_time_s": 2.0 if success else None})
                if success:
                    currents.append({"model": label, "suite": "fresh_long", "case": str(index),
                                     "window": "first_5s", "change_rms_A": 0.02,
                                     "total_variation_A": 4.0, "hf_power_fraction": 0.03})
        summary = summarize(episodes, currents)
        self.assertEqual(summary["fresh_long"]["weight4"]["flagged_cases"], ["1"])
        self.assertEqual(summary["fresh_long"]["weight3"]["windows"]["first_5s"]["paired_episodes"], 1)
        with TemporaryDirectory() as directory:
            write_report(Path(directory), summary)
            self.assertIn("FAILED", (Path(directory) / "FINDINGS.md").read_text())

    def test_sequence_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            calls = []

            def train(path, **kwargs):
                calls.append("train")
                self.assertEqual(kwargs, {"execute": True, "action_change_weight": 4.0})
                self.assertEqual(path, job / "candidate")

            def compare(path, cases, progress):
                calls.append("compare")
                progress("weight4", "fresh_long", 1, 1)

            with patch("weight4.validate_references", return_value={}), \
                 patch("weight4.shutil.copy2"), patch("weight4.file_sha256", return_value="hash"), \
                 patch("weight4.run_nominal", side_effect=train), patch("weight4.compare", side_effect=compare):
                run(job)
                self.assertFalse(job.exists())
                run(job, True)
                self.assertEqual(calls, ["train", "compare"])
                self.assertEqual(json.loads((job / "status.json").read_text())["state"], "completed")
                with self.assertRaises(FileExistsError):
                    run(job, True)

    def test_training_failure_skips_comparison(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            with patch("weight4.validate_references", return_value={}), \
                 patch("weight4.shutil.copy2"), \
                 patch("weight4.run_nominal", side_effect=RuntimeError("probe")), \
                 patch("weight4.compare") as comparison:
                with self.assertRaisesRegex(RuntimeError, "probe"):
                    run(job, True)
                comparison.assert_not_called()
            self.assertEqual(json.loads((job / "status.json").read_text())["state"], "failed")


if __name__ == "__main__":
    unittest.main()
