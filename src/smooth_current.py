"""Run the weight-5 nominal SAC experiment and a matched current-jitter comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
from stable_baselines3 import SAC

from evaluate import run_episode, save_detailed_episode, write_csv
from furuta_env import FurutaConfig
from nominal import CONFIG, DEFAULT_JOB as BASELINE_JOB, run as run_nominal
from train import algorithm_settings, atomic_write_json, file_sha256


DEFAULT_JOB = BASELINE_JOB.parent / "nominal_sac_actor64_delta5_seed0_v0"
WEIGHT = 5.0
SEEDS = range(80000, 80100)
WINDOWS = ("full_episode", "first_second", "capture_window")
METRICS = ("change_rms_A", "total_variation_A", "hf_power_fraction")


def validate_baseline(baseline: Path) -> None:
    saved_config = json.loads((baseline / "training" / "config.json").read_text(encoding="utf-8"))
    if saved_config != asdict(CONFIG):
        raise ValueError("Baseline task differs from the nominal weight-3 configuration")
    settings = json.loads((baseline / "training" / "training.json").read_text(encoding="utf-8"))
    expected = algorithm_settings(CONFIG, total_timesteps=2_000_000, actor_network=[64, 64], critic_network=[64, 64])
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Baseline learner settings differ from the intended experiment")
    if not (baseline / "training" / "best" / "best_model.zip").is_file():
        raise FileNotFoundError("Baseline best model is missing")
    for name in ("train.py", "furuta_env.py", "furuta_model.py", "controllers.py", "evaluate.py"):
        if file_sha256(baseline / "source" / name) != file_sha256(Path(__file__).parent / name):
            raise ValueError(f"Core implementation differs from baseline snapshot: {name}")


def window_metrics(result: dict, config: FurutaConfig, window: str) -> dict:
    commands = np.asarray(
        [row["command_current_A"] for row in result["trace"][1::config.action_repeat]],
        dtype=np.float64,
    )
    period = config.sample_time * config.action_repeat
    times = np.arange(commands.size) * period
    if window == "full_episode":
        mask = np.ones(commands.size, dtype=bool)
    elif window == "first_second":
        mask = times < 1.0
    elif window == "capture_window":
        capture = result["capture_time"]
        mask = np.zeros(commands.size, dtype=bool) if capture is None else (
            (times >= max(0.0, capture - 0.2)) & (times < capture + 0.2)
        )
    else:
        raise ValueError(f"Unknown current window: {window}")
    changes = np.diff(commands, prepend=0.0)[mask]
    values = commands[mask]
    if not values.size:
        return {"command_samples": 0, **{key: None for key in METRICS}}
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(values.size, d=period)
    total_power = float(np.sum(power[1:]))
    return {
        "command_samples": int(values.size),
        "change_rms_A": float(np.sqrt(np.mean(changes ** 2))),
        "total_variation_A": float(np.sum(np.abs(changes))),
        "hf_power_fraction": (
            float(np.sum(power[frequencies >= 25.0]) / total_power)
            if values.size >= 4 and total_power > 0.0 else 0.0
        ),
    }


def compare(job: Path) -> None:
    rows = []
    episode_rows = []
    for label, model_path, config in (
        ("baseline", job / "reference" / "model.zip", CONFIG),
        ("candidate", job / "candidate" / "training" / "best" / "best_model.zip", replace(CONFIG, action_change_weight=WEIGHT)),
    ):
        model = SAC.load(model_path, device="cpu")
        for index, seed in enumerate(SEEDS, 1):
            result = run_episode(model, seed, False, "training", config, "sac")
            episode_rows.append({
                "model": label, "seed": seed, "success": int(result["success"]),
                "unsafe": int(result["unsafe"]), "finish_time_s": result["finish_time"],
            })
            for window in WINDOWS:
                rows.append({"model": label, "seed": seed, "window": window, **window_metrics(result, config, window)})
            if seed in (80000, 80015):
                save_detailed_episode(job / "comparison" / label, label, index, seed, result, config)
            if index % 10 == 0:
                print(f"Jitter comparison: {label} {index}/{len(SEEDS)}", flush=True)
    write_csv(job / "comparison" / "current_windows.csv", tuple(rows[0]), rows)
    write_csv(job / "comparison" / "episodes.csv", tuple(episode_rows[0]), episode_rows)
    summary = {}
    for label in ("baseline", "candidate"):
        episodes = [row for row in episode_rows if row["model"] == label]
        times = [row["finish_time_s"] for row in episodes if row["success"]]
        summary[label] = {
            "successes": sum(row["success"] for row in episodes),
            "unsafe": sum(row["unsafe"] for row in episodes),
            "mean_successful_finish_s": float(np.mean(times)) if times else None,
            "max_successful_finish_s": max(times) if times else None,
            "failed_seeds": [row["seed"] for row in episodes if not row["success"] or row["unsafe"]],
            "windows": {},
        }
        for window in WINDOWS:
            selected = [row for row in rows if row["model"] == label and row["window"] == window and row["command_samples"]]
            summary[label]["windows"][window] = {"episodes": len(selected)}
            for metric in METRICS:
                values = [row[metric] for row in selected]
                summary[label]["windows"][window][metric] = {
                    "mean": float(np.mean(values)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                    "max": max(values) if values else None,
                }
    atomic_write_json(job / "comparison.json", summary)
    lines = [
        "# Current-Jitter Experiment", "",
        "Fresh nominal SAC, action-change weight 3 -> 5. All other training settings unchanged.",
        "Both validation-selected policies evaluated on matched seeds 80000-80099; no holdout-based checkpoint selection.", "",
    ]
    for label, result in summary.items():
        lines.append(
            f"- {label}: {result['successes']}/{len(SEEDS)} successes, {result['unsafe']} unsafe; "
            f"mean/max successful finish time: {result['mean_successful_finish_s']} / {result['max_successful_finish_s']} s."
        )
        if result["failed_seeds"]:
            lines.append(f"- {label} failed seeds: {result['failed_seeds']}")
    lines.extend(["", "| Window | Metric | Baseline mean | Candidate mean | Change |", "|---|---|---:|---:|---:|"])
    for window in WINDOWS:
        for metric in METRICS:
            before = summary["baseline"]["windows"][window][metric]["mean"]
            after = summary["candidate"]["windows"][window][metric]["mean"]
            delta = f"{100.0 * (after / before - 1.0):+.1f}%" if before and after is not None else "n/a"
            lines.append(f"| {window} | {metric} | {before} | {after} | {delta} |")
    reliable = summary["candidate"]["successes"] == len(SEEDS) and summary["candidate"]["unsafe"] == 0
    lines.extend([
        "", "## Assessment", "",
        "Nominal reliability gate retained." if reliable else "Nominal reliability gate FAILED: do not replace the baseline based on smoother current.",
        "Lower RMS command changes and lower high-frequency power together support reduced jitter; total variation alone also reflects swing-up effort.",
        "Use comparison.json for p95/max values and comparison/current_windows.csv for paired episode metrics.",
        "Full-episode and first-second windows are fixed; capture_window spans 0.2 s before/after each policy's first mechanical capture and excludes episodes without capture.",
        "Window metrics use 200 Hz requested commands, including changes from the preceding command (initial command is compared with zero). High-frequency power is the fraction at >=25 Hz.",
        "Unsafe episodes are truncated and successful-only settling times exclude failures; check reliability and sample counts before interpreting averages.",
        "These seeds have already been inspected for the baseline: this is a diagnostic comparison, not a new untouched holdout or multi-seed confirmation.",
        "The baseline is preserved; no controller is automatically promoted. Hardware and long-duration stability remain untested.",
    ])
    (job / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(job: Path = DEFAULT_JOB, execute: bool = False) -> None:
    job = job.resolve()
    if job.exists():
        raise FileExistsError(f"Refusing existing job output: {job}")
    validate_baseline(BASELINE_JOB)
    print(f"Fresh weight-{WEIGHT:g} nominal SAC + automatic evaluation and jitter comparison\nOutput: {job}", flush=True)
    if not execute:
        print("Dry run. Add --execute to start.", flush=True)
        return
    job.mkdir(parents=True)
    phase = "preparation"

    def status(state: str, **details) -> None:
        atomic_write_json(job / "status.json", {
            "state": state, "phase": phase,
            "updated_utc": datetime.now(timezone.utc).isoformat(), **details,
        })

    try:
        status("running")
        reference = job / "reference"
        reference.mkdir()
        for origin, target in (
            (BASELINE_JOB / "training" / "best" / "best_model.zip", "model.zip"),
            (BASELINE_JOB / "training" / "config.json", "config.json"),
            (BASELINE_JOB / "training" / "training.json", "training.json"),
        ):
            shutil.copy2(origin, reference / target)
        atomic_write_json(job / "protocol.json", {
            "action_change_weight": WEIGHT, "baseline_action_change_weight": CONFIG.action_change_weight,
            "baseline_model_sha256": file_sha256(reference / "model.zip"),
            "seeds": list(SEEDS), "windows": list(WINDOWS),
            "reliability_gate": {"successes": 100, "unsafe": 0},
            "selection": "Training-validation best only; no automatic promotion.",
        })
        phase = "training_and_evaluation"
        status("running")
        run_nominal(job / "candidate", execute=True, action_change_weight=WEIGHT)
        phase = "jitter_comparison"
        status("running")
        compare(job)
        phase = "report"
        status("completed", report=str(job / "FINDINGS.md"))
    except Exception:
        failure = traceback.format_exc()
        status("failed", error=failure)
        (job / "ERROR.txt").write_text(failure, encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    run(arguments.job_dir, arguments.execute)
