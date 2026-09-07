"""Run an isolated recipe, evaluate its products, and save an unattended report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import reproduce


WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_RECIPE = WORKSPACE / "experiments" / "stage3c_half_rps_gamma09975_v0.json"
DEFAULT_JOB = WORKSPACE / "src" / "runs" / "overnight_gamma09975_seed0"
BASE_SEED = 70000
EPISODES = 100


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(job: Path, state: str, phase: str, **details) -> None:
    temporary = job / "status.tmp"
    temporary.write_text(
        json.dumps(
            {"state": state, "phase": phase, "updated_utc": timestamp(), **details},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(job / "status.json")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_report(job: Path, evaluations: list[tuple[str, Path]], complete: bool) -> None:
    lines = [
        "# Longer-Horizon Seed-0 Results",
        "",
        f"Status: {'complete' if complete else 'in progress'}. Updated: {timestamp()}",
        "",
        "Gamma 0.9975; five-second episodes; unchanged +100 success bonus.",
        "100 episodes per mode and controller; seeds 70000-70299.",
        "Best models are chosen by training validation, not these evaluation seeds.",
        "Final models are diagnostic comparisons, not holdout-selected replacements.",
        "",
        "| Model | Controller | Mode | Successes | Unsafe | Command TV (A) | Arm RMS speed (rad/s) | Pendulum RMS speed (rad/s) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    all_rows = []
    for label, results in evaluations:
        for row in read_rows(results / "summary.csv"):
            all_rows.append((label, row))
            lines.append(
                f"| {label} | {row['controller']} | {row['mode']} | "
                f"{row['success_count']}/{row['episodes']} | {row['unsafe_count']} | "
                f"{float(row['mean_swingup_current_total_variation_A']):.3f} | "
                f"{float(row['mean_omega1_rms_rad_s']):.3f} | "
                f"{float(row['mean_omega2_rms_rad_s']):.3f} |"
            )
    lines.extend(["", "## Findings", ""])
    for label, results in evaluations:
        rows = [row for model, row in all_rows if model == label]
        successes = sum(int(row["success_count"]) for row in rows)
        total = sum(int(row["episodes"]) for row in rows)
        unsafe = sum(int(row["unsafe_count"]) for row in rows)
        lines.append(f"- {label}: {successes}/{total} successful, {unsafe} unsafe.")
        failure_seeds = []
        for path in sorted(results.glob("*/*/episodes.csv")):
            for episode in read_rows(path):
                if int(episode["unsafe"]) or not int(episode["success"]):
                    failure_seeds.append(
                        f"{episode['controller']}/{episode['mode']} seed {episode['seed']}"
                        + (" (unsafe)" if int(episode["unsafe"]) else " (not settled)")
                    )
        if failure_seeds:
            lines.append("- Failure examples: " + "; ".join(failure_seeds[:12]) + ".")
            if len(failure_seeds) > 12:
                lines.append("- Additional failures are recorded in the episode CSV files.")
    baseline_path = job / "reference" / "holdout_sac_summary.csv"
    if baseline_path.exists():
        baseline = {row["mode"]: row for row in read_rows(baseline_path)}
        for label, row in all_rows:
            if label != "stage3c_curriculum_best" or row["controller"] != "sac":
                continue
            reference = baseline[row["mode"]]
            changes = []
            for field, name in (
                ("mean_swingup_current_total_variation_A", "command TV"),
                ("mean_omega1_rms_rad_s", "arm RMS speed"),
                ("mean_omega2_rms_rad_s", "pendulum RMS speed"),
            ):
                delta = 100.0 * (float(row[field]) / float(reference[field]) - 1.0)
                changes.append(f"{name} {delta:+.1f}%")
            lines.append(
                f"- Stage-3c best SAC, {row['mode']}, versus frozen 2.6M release: "
                + ", ".join(changes) + ". Lower is preferable only with retained reliability."
            )
    lines.extend([
        "",
        "## Interpretation Limits",
        "",
        "This is one training seed, not evidence of training-seed robustness or hardware safety.",
        "Stage-2 uncertainty settings differ from Stage-3c; their results are not matched-envelope comparisons.",
        "The historical controller used older validation settings and a different checkpoint-selection process.",
        "A fresh gamma-0.99 control under the same code is needed to isolate gamma's effect.",
        "The 50-episode validation setting is used throughout this new chain.",
        "Deterministic stress testing and hardware qualification are not part of this job.",
        "See status.json and phase logs for execution status; partial reports are not final results.",
    ])
    (job / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(job: Path, phase: str, command: list[str]) -> None:
    write_status(job, "running", phase, command=command, supervisor_pid=os.getpid())
    print(f"{timestamp()} Starting {phase}", flush=True)
    with (job / f"{phase}.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    print(f"{timestamp()} Finished {phase}", flush=True)


def run_job(recipe_path: Path, job: Path, seed: int, execute: bool, allow_dirty: bool = False) -> None:
    recipe = reproduce.load_recipe(recipe_path)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    for stage in recipe["stages"]:
        directory = reproduce.stage_run_dir(stage, seed)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing existing stage output: {directory}")
        print(subprocess.list2cmdline(reproduce.build_stage_command(recipe, stage, seed)))
    if job.exists():
        raise FileExistsError(f"Refusing existing job output: {job}")
    if not execute:
        print(f"Dry run: train all stages, then evaluate stage best models and the final model into {job}")
        return
    if reproduce.working_tree_dirty() and not allow_dirty:
        raise RuntimeError("Dirty working tree: use --allow-dirty for an explicitly noncanonical job.")
    job.mkdir(parents=True)
    evaluations = []
    try:
        write_status(job, "running", "snapshot", supervisor_pid=os.getpid())
        source = job / "source" / "src"
        source.mkdir(parents=True)
        for path in (WORKSPACE / "src").glob("*.py"):
            shutil.copy2(path, source / path.name)
        shutil.copy2(recipe_path, job / "recipe.json")
        reference = job / "reference"
        reference.mkdir()
        shutil.copy2(
            WORKSPACE / "models" / "stage3c_half_rps_v0" / "evidence" / "holdout_sac_summary.csv",
            reference / "holdout_sac_summary.csv",
        )
        provenance = {
            "started_utc": timestamp(), "seed": seed, "python": sys.executable,
            "recipe_sha256": recipe["_sha256"],
            "working_tree_dirty": reproduce.working_tree_dirty(),
            "source_sha256": {
                path.name: reproduce.file_sha256(path) for path in source.glob("*.py")
            },
            "evaluation_base_seed": BASE_SEED, "episodes_per_mode": EPISODES,
        }
        (job / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        specification = importlib.util.spec_from_file_location("frozen_reproduce", source / "reproduce.py")
        frozen = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(frozen)
        frozen.WORKSPACE = WORKSPACE
        write_report(job, evaluations, complete=False)
        for stage in recipe["stages"]:
            command = frozen.build_stage_command(recipe, stage, seed)
            command[1] = str(source / "train.py")
            command.insert(1, "-u")
            run_command(job, stage["id"], command)
            marker = frozen.stage_marker(recipe, stage, seed, command)
            (frozen.stage_run_dir(stage, seed) / "pipeline_stage.json").write_text(
                json.dumps(marker, indent=2) + "\n", encoding="utf-8",
            )
        candidates = [
            (stage["id"] + "_best", frozen.stage_run_dir(stage, seed) / "best" / "best_model.zip")
            for stage in recipe["stages"]
        ]
        final_stage = recipe["stages"][-1]
        candidates.append((
            final_stage["id"] + "_final",
            frozen.stage_run_dir(final_stage, seed) / "final" / "model.zip",
        ))
        for label, model in candidates:
            results = job / "evaluations" / label
            run_command(job, "evaluate_" + label, [
                sys.executable, "-u", str(source / "evaluate.py"),
                "--model", str(model), "--controller", "both",
                "--episodes", str(EPISODES), "--base-seed", str(BASE_SEED),
                "--results-dir", str(results),
            ])
            evaluations.append((label, results))
            write_report(job, evaluations, complete=False)
        write_report(job, evaluations, complete=True)
        write_status(job, "completed", "report", report=str(job / "FINDINGS.md"))
    except Exception:
        failure = traceback.format_exc()
        previous_status = json.loads((job / "status.json").read_text(encoding="utf-8"))
        write_status(job, "failed", previous_status["phase"], error=failure)
        write_report(job, evaluations, complete=False)
        with (job / "FINDINGS.md").open("a", encoding="utf-8") as report:
            report.write("\n## Execution Failure\n\n```text\n" + failure + "```\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    run_job(arguments.recipe.resolve(), arguments.job_dir.resolve(), arguments.seed, arguments.execute, arguments.allow_dirty)
