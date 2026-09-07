"""Checks for the unattended job's process and reporting behavior."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from overnight import run_command, run_job, write_report, write_status
import reproduce


class OvernightTests(unittest.TestCase):
    def test_command_status_and_log(self):
        with TemporaryDirectory() as directory:
            job = Path(directory)
            run_command(job, "probe", [sys.executable, "-c", "print('completed probe')"])
            self.assertIn("completed probe", (job / "probe.log").read_text())
            self.assertEqual(json.loads((job / "status.json").read_text())["phase"], "probe")
            with self.assertRaises(subprocess.CalledProcessError):
                run_command(job, "failure", [sys.executable, "-c", "raise SystemExit(2)"])

    def test_report_reads_existing_evidence(self):
        root = Path(__file__).resolve().parent.parent
        rows = root / "models" / "stage3c_half_rps_v0" / "evidence" / "holdout_sac_summary.csv"
        with TemporaryDirectory() as directory:
            job = Path(directory)
            results = job / "evaluation"
            results.mkdir()
            (results / "summary.csv").write_bytes(rows.read_bytes())
            write_report(job, [("stage3c_curriculum_best", results)], complete=True)
            text = (job / "FINDINGS.md").read_text()
            self.assertIn("300/300 successful, 0 unsafe", text)
            self.assertIn("Status: complete", text)

    def test_job_sequences_training_evaluation_and_reports(self):
        root = Path(__file__).resolve().parent.parent
        recipe = root / "experiments" / "stage3c_half_rps_gamma09975_v0.json"
        evidence = root / "models" / "stage3c_half_rps_v0" / "evidence" / "holdout_sac_summary.csv"
        phases = []
        with TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            def fake_command(job, phase, command):
                phases.append(phase)
                write_status(job, "running", phase)
                if "--run-dir" in command:
                    run = Path(command[command.index("--run-dir") + 1])
                    final = run / "final"
                    final.mkdir(parents=True)
                    (final / "model.zip").write_bytes(b"test model")
                    (final / "replay_buffer.pkl").write_bytes(b"test replay")
                    (final / "state.json").write_text(json.dumps({"num_timesteps": 2}))
                else:
                    results = Path(command[command.index("--results-dir") + 1])
                    results.mkdir(parents=True)
                    (results / "summary.csv").write_bytes(evidence.read_bytes())

            original_load = reproduce.load_recipe
            def temporary_recipe(path):
                data = original_load(path)
                copied_recipe = temporary_root / "recipe.json"
                copied_recipe.write_bytes(path.read_bytes())
                data["_path"] = str(copied_recipe)
                for stage in data["stages"]:
                    stage["run_dir"] = str(temporary_root / "src" / "runs" / stage["id"])
                return data

            with patch("overnight.reproduce.load_recipe", side_effect=temporary_recipe), \
                 patch("overnight.reproduce.WORKSPACE", temporary_root), \
                 patch("overnight.WORKSPACE", temporary_root), \
                 patch("overnight.run_command", side_effect=fake_command):
                import shutil
                (temporary_root / "src").mkdir()
                for source in (root / "src").glob("*.py"):
                    shutil.copy2(source, temporary_root / "src" / source.name)
                reference = temporary_root / "models" / "stage3c_half_rps_v0" / "evidence"
                reference.mkdir(parents=True)
                shutil.copy2(evidence, reference / evidence.name)
                with patch("overnight.reproduce.working_tree_dirty", return_value=True):
                    job = temporary_root / "job"
                    run_job(recipe, job, 0, True, allow_dirty=True)
                self.assertEqual(len(phases), 7)
                self.assertTrue(all(phase.startswith("evaluate_") for phase in phases[3:]))
                self.assertEqual(json.loads((job / "status.json").read_text())["state"], "completed")
                self.assertIn("command TV +0.0%", (job / "FINDINGS.md").read_text())

    def test_dry_run_does_not_execute_or_create_output(self):
        root = Path(__file__).resolve().parent.parent
        with TemporaryDirectory() as directory:
            job = Path(directory) / "job"
            with patch("overnight.subprocess.run") as command:
                run_job(root / "experiments" / "stage3c_half_rps_gamma09975_v0.json", job, 87361, False)
            command.assert_not_called()
            self.assertFalse(job.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
