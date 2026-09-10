from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from current15_delta4 import CONFIG, REFERENCE_CONFIG, SEEDS, TRAINING, evaluation_job, training_job


class Current15Delta4Tests(unittest.TestCase):
    def test_single_change_and_budget(self):
        self.assertEqual([key for key in asdict(CONFIG) if asdict(CONFIG)[key] != asdict(REFERENCE_CONFIG)[key]], ["action_change_weight"])
        self.assertEqual(CONFIG.action_change_weight, 4)
        self.assertEqual(CONFIG.current_weight, 1.5)
        self.assertEqual(TRAINING["total_timesteps"], 2000000)
        self.assertEqual(TRAINING["evaluation_episodes"], 50)
        self.assertEqual(SEEDS, list(range(580000, 580020)))

    def test_training_does_not_evaluate(self):
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            with patch("current15_delta4.preflight", return_value=Path(directory) / "model.zip"), \
                    patch("current15_delta4.shutil.copy2"), patch("current15_delta4.file_sha256", return_value="hash"), \
                    patch("current15_delta4.train") as train, patch("current15_delta4.evaluation_job") as evaluate:
                training_job(job)
                self.assertFalse(job.exists())
                train.assert_not_called()
                training_job(job, True)
                self.assertEqual(train.call_args.kwargs["config"], CONFIG)
                self.assertNotIn("resume_model_path", train.call_args.kwargs)
                evaluate.assert_not_called()
            self.assertTrue((job / "training_completed.json").exists())
            self.assertFalse((job / "completed.json").exists())
            with self.assertRaises(FileExistsError):
                training_job(job, True)

    def test_evaluation_missing_training_rejected(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                evaluation_job(Path(directory), True)

    def test_evaluation_guard_and_hashes(self):
        with TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "training_completed.json").write_text(json.dumps({"hashes": {"original": "hash", "modified": "hash"}}))
            (job / "protocol.json").write_text(json.dumps({"config": asdict(CONFIG), "reference_config": asdict(REFERENCE_CONFIG),
                "seeds": SEEDS, "training": TRAINING, "evaluation_horizon_s": 30.0, "source_sha256": {}}))
            with patch("current15_delta4.file_sha256", return_value="wrong"):
                with self.assertRaises(ValueError):
                    evaluation_job(job)
            with patch("current15_delta4.file_sha256", return_value="hash"):
                evaluation_job(job)
                self.assertFalse((job / "evaluation_started.json").exists())
                (job / "evaluation_started.json").write_text("{}")
                with self.assertRaises(FileExistsError):
                    evaluation_job(job, True)

    def test_compact_evaluation_completes_without_training(self):
        with TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "training_completed.json").write_text(json.dumps({"hashes": {"original": "hash", "modified": "hash"}}))
            (job / "protocol.json").write_text(json.dumps({"config": asdict(CONFIG), "reference_config": asdict(REFERENCE_CONFIG),
                "seeds": SEEDS, "training": TRAINING, "evaluation_horizon_s": 30.0, "source_sha256": {}}))
            measures = {"unsafe": 0, "centered_success_5s": 0, "centered_success_end": 1,
                        "centered_lost_after_5s": 0, "centered_finish_s": 8.0}
            current = {key: .1 for key in ("current_rms_A", "change_rms_A", "total_variation_A", "hf_rms_A")}
            with patch("current15_delta4.file_sha256", return_value="hash"), \
                    patch("current15_delta4.SAC.load"), patch("current15_delta4.train") as training, \
                    patch("current15_delta4.make_episode", return_value=SimpleNamespace(x=np.zeros(4))), \
                    patch("current15_delta4.run_prepared_episode", return_value={"trace": [{}] * 30001, "unsafe": False}) as episodes, \
                    patch("current15_delta4.score", return_value=measures), \
                    patch("current15_delta4.window_current", return_value=current), \
                    patch("current15_delta4.save_detailed_episode"), \
                    patch("analyze_damping.policy_analysis", return_value={"equilibria": []}) as poles:
                evaluation_job(job, True)
                training.assert_not_called()
                self.assertEqual(episodes.call_count, 40)
                self.assertEqual(poles.call_count, 2)
            self.assertTrue((job / "completed.json").is_file())
            self.assertTrue((job / "REPORT.md").is_file())
            summary = json.loads((job / "evaluation/summary.json").read_text())["fresh20"]
            self.assertEqual(summary["modified"]["centered_success_end"], 20)
            self.assertEqual(summary["modified"]["mean_finish_s"], 8.0)
            self.assertEqual(summary["modified"]["windows"]["first_5s"]["paired_cases"], 20)


if __name__ == "__main__":
    unittest.main()
