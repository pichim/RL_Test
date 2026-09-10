from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from current_penalty import CONFIG, TASK_CONFIG, TRAINING, cases, evaluate_job, review, run, summarize, window_current


class CurrentPenaltyTests(unittest.TestCase):
    def test_protocol_cases_and_single_change(self):
        planned = cases()
        self.assertEqual(len(planned), 284)
        self.assertEqual([case["seed"] for case in planned if case["suite"] == "fresh_long"], list(range(480000, 480100)))
        self.assertEqual(len({(case["suite"], case["name"]) for case in planned}), len(planned))
        self.assertEqual([key for key in asdict(CONFIG) if asdict(CONFIG)[key] != asdict(TASK_CONFIG)[key]], ["current_weight"])
        self.assertEqual(TRAINING["total_timesteps"], 2_000_000)

    def test_absolute_spectrum_and_late_window_boundary(self):
        time = np.arange(1000) * .005
        commands = .1 + .02 * np.sin(2 * np.pi * 40 * time) + .03 * (-1.)**np.arange(1000)
        trace = [{"command_current_A": 0.0}] + [{"command_current_A": float(value)} for value in np.repeat(commands, 5)]
        metrics = window_current({"trace": trace}, CONFIG, 0, 5)
        self.assertAlmostEqual(metrics["hf_rms_A"], np.sqrt(.02**2 / 2 + .03**2))
        self.assertAlmostEqual(metrics["current_rms_A"], np.sqrt(.1**2 + .02**2 / 2 + .03**2))
        constant = {"trace": [{"command_current_A": .2} for _ in range(30001)]}
        self.assertEqual(window_current(constant, CONFIG, 25, 30)["change_rms_A"], 0.0)
        self.assertIsNone(window_current({"trace": trace[:-1]}, CONFIG, 0, 5))

    def test_pairing_does_not_hide_failures(self):
        episodes = [{"suite": "fresh_long", "model": label, "case": name,
                     "unsafe": int(label == "candidate" and name == "bad"),
                     "centered_success_5s": int(name == "good"), "centered_success_end": int(name == "good"),
                     "centered_lost_after_5s": 0, "centered_finish_s": 2.0 if name == "good" else None,
                     "minimum_travel_margin_rad": .1} for label in ("baseline", "candidate") for name in ("good", "bad")]
        metric = {key: .1 for key in ("change_rms_A", "total_variation_A", "current_rms_A", "peak_current_A", "ac_rms_A", "hf_rms_A")}
        currents = [{"suite": "fresh_long", "model": label, "case": name, "window": "first_5s", **metric}
                    for label, name in (("baseline", "good"), ("baseline", "bad"), ("candidate", "good"))]
        result = summarize(episodes, currents)
        self.assertEqual(result["fresh_long"]["candidate"]["unsafe"], 1)
        self.assertEqual(result["fresh_long"]["candidate"]["flagged_cases"], ["bad"])
        self.assertEqual(result["fresh_long"]["baseline"]["windows"]["first_5s"]["paired_episodes"], 1)

    def test_pipeline_order_and_failure_marker(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "success"
            order = []
            def training(**kwargs):
                order.append("train")
                self.assertEqual(kwargs["config"], TASK_CONFIG)
                self.assertEqual(kwargs["gamma"], .99)
            with patch("current_penalty.ANALYSIS_PYTHON", Path(sys.executable)), \
                    patch("current_penalty.preflight", return_value={}), patch("current_penalty.shutil.copy2"), \
                    patch("current_penalty.file_sha256", return_value="hash"), patch("current_penalty.train", side_effect=training), \
                    patch("current_penalty.evaluate", side_effect=lambda **kwargs: order.append("evaluate")), \
                    patch("current_penalty.compare", side_effect=lambda *args: order.append("compare") or {}), \
                    patch("current_penalty.subprocess.run", side_effect=lambda *args, **kwargs: order.append("poles")), \
                    patch("current_penalty.review", side_effect=lambda *args: order.append("review") or {}):
                run(job, True)
            self.assertEqual(order, ["train", "evaluate", "compare", "poles", "review"])
            self.assertTrue((job / "completed.json").exists())
            self.assertFalse((job / "failed.json").exists())
            with self.assertRaises(FileExistsError):
                run(job, True)
            failed = Path(directory) / "failure"
            with patch("current_penalty.preflight", return_value={}), patch("current_penalty.shutil.copy2"), \
                    patch("current_penalty.file_sha256", return_value="hash"), patch("current_penalty.train", side_effect=RuntimeError("probe")):
                with self.assertRaisesRegex(RuntimeError, "probe"):
                    run(failed, True)
            self.assertTrue((failed / "failed.json").exists())
            self.assertFalse((failed / "completed.json").exists())

    def test_train_only_then_independent_evaluation(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "manual"
            with patch("current_penalty.ANALYSIS_PYTHON", Path(sys.executable)), \
                    patch("current_penalty.preflight", return_value={}), patch("current_penalty.shutil.copy2"), \
                    patch("current_penalty.file_sha256", return_value="hash"), \
                    patch("current_penalty.train") as training, patch("current_penalty.evaluate") as evaluation:
                run(job, True, train_only=True)
                training.assert_called_once()
                evaluation.assert_not_called()
            self.assertTrue((job / "training_completed.json").exists())
            self.assertFalse((job / "evaluation_started.json").exists())
            self.assertFalse((job / "completed.json").exists())
            with patch("current_penalty.file_sha256", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "model hash differs"):
                    evaluate_job(job, True)
            self.assertFalse((job / "evaluation_started.json").exists())
            with patch("current_penalty.file_sha256", return_value="hash"), \
                    patch("current_penalty.train") as training, patch("current_penalty.evaluate") as evaluation, \
                    patch("current_penalty.compare", return_value={}), patch("current_penalty.review", return_value={}), \
                    patch("current_penalty.subprocess.run") as analysis, \
                    patch("current_penalty.ANALYSIS_PYTHON", Path(directory) / "wrong-interpreter"):
                evaluate_job(job)
                evaluation.assert_not_called()
                self.assertFalse((job / "evaluation_started.json").exists())
                evaluate_job(job, True)
                training.assert_not_called()
                evaluation.assert_called_once()
                self.assertEqual(evaluation.call_args.kwargs["config_path"], job / "candidate/training/config.json")
                self.assertEqual(analysis.call_args.args[0][0], str(Path(sys.executable).resolve()))
                self.assertTrue((job / "completed.json").exists())
                with self.assertRaises(FileExistsError):
                    evaluate_job(job, True)

    def test_evaluation_requires_completed_training(self):
        with TemporaryDirectory() as directory:
            with patch("current_penalty.train") as training, patch("current_penalty.evaluate") as evaluation:
                with self.assertRaises(FileNotFoundError):
                    evaluate_job(Path(directory), True)
                training.assert_not_called()
                evaluation.assert_not_called()

    def test_review_fails_closed_without_centered_root_or_current_pairs(self):
        with TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "poles").mkdir()
            (job / "poles/analysis.json").write_text(json.dumps({"models": {"candidate": {"equilibria": []}}}))
            data = {"episodes": 1, "flagged_cases": ["failure"], "centered_success_5s": 0,
                    "centered_success_end": 0, "unsafe": 1, "centered_lost_after_5s": 0,
                    "finish_s": None, "windows": {name: {"paired_episodes": 0,
                        **{metric: None for metric in ("change_rms_A", "total_variation_A", "current_rms_A", "hf_rms_A")}}
                        for name in ("first_5s", "last_5s")}}
            result = review(job, {"fresh_long": {"baseline": data, "candidate": data}})
            for key in ("all_case_reliability", "local_pole_screen", "fresh_current_nonregression", "fresh_mean_finish_within_10_percent"):
                self.assertFalse(result[key])
            self.assertTrue((job / "FINDINGS.md").exists())


if __name__ == "__main__":
    unittest.main()
