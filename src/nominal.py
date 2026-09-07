"""Train one nominal direct-current SAC controller and evaluate its best model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
import traceback

from evaluate import main as evaluate
from furuta_env import DEFAULT_CONFIG
from train import atomic_write_json, train


DEFAULT_JOB = Path(__file__).resolve().parent / "runs" / "nominal_sac_actor64_seed0_v0"
CONFIG = replace(
    DEFAULT_CONFIG,
    current_filter_cutoff_hz=0.0,
    current_filter_cutoff_randomization_hz=0.0,
    current_filter_cutoff_min_hz=0.0,
    current_filter_cutoff_max_hz=0.0,
    action_dead_time_max_samples=0,
    training_parameter_randomization=False,
    motor_torque_randomization=0.0,
    arm_mass_randomization=0.0,
    arm_inertia_randomization=0.0,
    pendulum_mass_randomization=0.0,
    pendulum_inertia_randomization=0.0,
    arm_damping_randomization=0.0,
    pendulum_damping_randomization=0.0,
)


def run(job: Path = DEFAULT_JOB, execute: bool = False) -> None:
    job = job.resolve()
    settings = {
        "seed": 0,
        "total_timesteps": 2_000_000,
        "gamma": 0.99,
        "actor_network": [64, 64],
        "critic_network": [64, 64],
        "evaluation_episodes": 50,
    }
    print(f"Nominal SAC: {settings}\nOutput: {job}", flush=True)
    if job.exists():
        raise FileExistsError(f"Choose a fresh job directory: {job}")
    if not execute:
        print("Dry run. Add --execute to train and evaluate.", flush=True)
        return
    job.mkdir(parents=True)
    phase = "training"

    def status(state: str, **details) -> None:
        atomic_write_json(job / "status.json", {
            "state": state,
            "phase": phase,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **details,
        })

    try:
        status("running")
        source = job / "source"
        source.mkdir()
        for path in Path(__file__).resolve().parent.glob("*.py"):
            shutil.copy2(path, source / path.name)
        atomic_write_json(job / "experiment.json", {
            "training": settings,
            "config": asdict(CONFIG),
            "evaluation": {"controller": "sac", "mode": "nominal", "episodes": 100, "base_seed": 80000},
            "selection": "Fixed nominal training validation; no selection on evaluation seeds.",
        })
        train(config=CONFIG, run_dir=job / "training", **settings)
        phase = "evaluation"
        status("running")
        evaluate(
            model_path=job / "training" / "best" / "best_model.zip",
            results_dir=job / "evaluation",
            controller="sac",
            mode="nominal",
            episodes=100,
            base_seed=80000,
        )
        with (job / "evaluation" / "summary.csv").open(newline="", encoding="utf-8") as handle:
            summary = next(csv.DictReader(handle))
        report = (
            "# Nominal SAC Result\n\n"
            f"Successes: {summary['success_count']}/{summary['episodes']}. "
            f"Unsafe episodes: {summary['unsafe_count']}. "
            f"Captures: {summary['capture_count']}.\n\n"
            "Fixed nominal plant, direct current, no delay, no LQR. "
            "The policy is selected using training validation, not evaluation seeds.\n\n"
            "Success requires the final second within 5 degrees of upright, "
            "0.25 rad absolute arm position, 0.5 rad/s arm speed, and 1 rad/s pendulum speed. "
            "Initial states still vary across the existing broad reset distribution.\n\n"
            "See evaluation/summary.txt and evaluation/sac/training_nominal for metrics, PNGs, and traces. "
            "One seed and a five-second test do not establish long-duration or hardware reliability.\n"
        )
        (job / "FINDINGS.md").write_text(report, encoding="utf-8")
        phase = "report"
        status("completed", success_count=int(summary["success_count"]), unsafe_count=int(summary["unsafe_count"]))
    except Exception:
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    run(arguments.job_dir, arguments.execute)
