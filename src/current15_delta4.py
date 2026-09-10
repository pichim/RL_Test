"""Fresh current1.5/delta4 SAC training with a compact independent evaluation."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from stable_baselines3 import SAC

from current_penalty import DEFAULT_JOB as REFERENCE_JOB, TASK_CONFIG as REFERENCE_CONFIG, TRAINING, window_current
from evaluate import make_episode, run_prepared_episode, save_detailed_episode, write_csv
from feedback_swingup import summarize
from no_centering import score, status
from train import atomic_write_json, file_sha256, train


CONFIG = replace(REFERENCE_CONFIG, action_change_weight=4.0)
DEFAULT_JOB = REFERENCE_JOB.parent / "nominal_sac_current15_delta4_seed0_v0"
SEEDS = list(range(580000, 580020))


def preflight():
    saved = json.loads((REFERENCE_JOB / "protocol.json").read_text())
    if saved["config"] != asdict(REFERENCE_CONFIG) or saved["training"] != TRAINING:
        raise ValueError("Reference task/training differs")
    if not (REFERENCE_JOB / "completed.json").is_file():
        raise FileNotFoundError("Reference experiment is incomplete")
    for name in ("train.py", "furuta_env.py", "furuta_model.py", "evaluate.py"):
        if file_sha256(Path(__file__).with_name(name)) != saved["source_sha256"][name]:
            raise ValueError(f"Core source differs from reference: {name}")
    reference = REFERENCE_JOB / "reference/candidate.zip"
    hashes = json.loads((REFERENCE_JOB / "training_completed.json").read_text())["model_hashes"]
    if file_sha256(reference) != hashes["candidate"]:
        raise ValueError("Reference model hash differs")
    return reference


def training_job(job, execute=False):
    job = Path(job).resolve()
    if job.exists():
        raise FileExistsError(f"Refusing existing output: {job}")
    reference = preflight()
    print(f"Fresh seed0, 2M decisions, current1.5/delta4; final evaluation: {len(SEEDS)} paired 30s cases. {job}", flush=True)
    if not execute:
        print("Preflight passed; add --execute to train only.", flush=True)
        return
    job.mkdir(parents=True)
    try:
        status(job, "running", "preparation")
        source = job / "source"
        source.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, source / path.name)
        (job / "reference").mkdir()
        shutil.copy2(reference, job / "reference/original.zip")
        atomic_write_json(job / "protocol.json", {"config": asdict(CONFIG), "reference_config": asdict(REFERENCE_CONFIG),
            "training": TRAINING, "seeds": SEEDS, "evaluation_horizon_s": 30.0,
            "reference_sha256": file_sha256(reference), "reference_job": str(REFERENCE_JOB.resolve()),
            "single_change": {"action_change_weight": [3.0, 4.0]}, "settling_speed_gate": False,
            "selection": "50 fixed nominal validation cases every100k; success then reward. Best checkpoint, not final.",
            "evaluation": "20 fresh paired resets; exploratory only, no promotion. No correction wrapper or LQR.",
            "python": sys.executable,
            "source_sha256": {path.name: file_sha256(path) for path in source.glob("*.py")}})
        status(job, "running", "training")
        train(config=CONFIG, run_dir=job / "candidate/training", **TRAINING)
        shutil.copy2(job / "candidate/training/best/best_model.zip", job / "reference/modified.zip")
        atomic_write_json(job / "training_completed.json", {"state": "trained",
            "hashes": {label: file_sha256(job / "reference" / f"{label}.zip") for label in ("original", "modified")}})
        status(job, "trained", "awaiting_evaluation")
    except Exception:
        failure(job)
        raise


def failure(job):
    error = traceback.format_exc()
    atomic_write_json(job / "failed.json", {"error": error})
    status(job, "failed", "error", error=error)


def evaluation_job(job, execute=False):
    job = Path(job).resolve()
    trained = json.loads((job / "training_completed.json").read_text())
    protocol = json.loads((job / "protocol.json").read_text())
    if protocol["config"] != asdict(CONFIG) or protocol["reference_config"] != asdict(REFERENCE_CONFIG):
        raise ValueError("Task mismatch: run the frozen source")
    if protocol["seeds"] != SEEDS or protocol["evaluation_horizon_s"] != 30.0 or protocol["training"] != TRAINING:
        raise ValueError("Evaluation protocol differs")
    for name, expected in protocol["source_sha256"].items():
        if file_sha256(job / "source" / name) != expected:
            raise ValueError(f"Frozen source mismatch: {name}")
    for label, expected in trained["hashes"].items():
        if file_sha256(job / "reference" / f"{label}.zip") != expected:
            raise ValueError(f"Model mismatch: {label}")
    if (job / "evaluation_started.json").exists():
        raise FileExistsError("Evaluation already started; refusing overwrite")
    if not execute:
        print("Evaluation preflight passed; add --execute.", flush=True)
        return
    with (job / "evaluation_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"state": "started", "python": sys.executable}, handle)
    try:
        from analyze_damping import policy_analysis
        models = {label: SAC.load(job / "reference" / f"{label}.zip", device="cpu") for label in ("original", "modified")}
        episodes, currents, poles = [], [], {}
        for label, model in models.items():
            config = replace(REFERENCE_CONFIG if label == "original" else CONFIG, episode_time=30.0)
            for index, seed in enumerate(SEEDS, 1):
                plant = make_episode(seed, config, False, "training")
                initial = plant.x.tolist()
                result = run_prepared_episode(model, plant, config, "sac")
                metrics = score(result, config)
                episodes.append({"suite": "fresh20", "controller": label, "case": f"seed_{seed}", "seed": seed,
                    "initial_state": json.dumps(initial), **metrics,
                    "completed_horizon": int(len(result["trace"]) == 30001)})
                for window, start, end in (("first_5s", 0, 5), ("last_5s", 25, 30)):
                    current = window_current(result, config, start, end)
                    if current is not None:
                        currents.append({"suite": "fresh20", "controller": label, "case": f"seed_{seed}", "window": window, **current})
                destination = job / "evaluation" / label
                save_detailed_episode(destination, label, index, seed, result, config)
                write_csv(job / "evaluation/episodes.csv", tuple(episodes[0]), episodes)
                if currents:
                    write_csv(job / "evaluation/current_windows.csv", tuple(currents[0]), currents)
                status(job, "running", "evaluation", controller=label, completed=index, total=len(SEEDS))
                print(f"Evaluation {label}: {index}/{len(SEEDS)} unsafe={result['unsafe']}", flush=True)
            poles[label] = policy_analysis(model)
        summary = summarize(episodes, currents)
        atomic_write_json(job / "evaluation/summary.json", summary)
        atomic_write_json(job / "evaluation/poles.json", poles)
        lines = ["# Current1.5 / Delta4 Experiment", "",
            "Fresh seed0 2M training; only action-change weight changed 3 to 4. No local correction or LQR.",
            "Original = current1.5/delta3; modified = newly trained current1.5/delta4.",
            "20 paired 30s cases, seeds580000-580019. Small exploratory sample, not full qualification.",
            "No settling-speed or slow-decay nonregression gate. No automatic promotion.", "",
            "| Model | Success at 5s/end | Unsafe | Late losses | Mean finish (s) |",
            "|---|---:|---:|---:|---:|"]
        for label, values in summary["fresh20"].items():
            lines.append(f"| {label} | {values['centered_success_5s']}/{values['centered_success_end']} of 20 | {values['unsafe']} | "
                         f"{values['centered_lost_after_5s']} | {values['mean_finish_s']} |")
        lines += ["", "## Current Metrics", "", "| Window | Model | Pairs | RMS (A) | Change RMS (A) | TV (A) | HF RMS (A) |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for window in ("first_5s", "last_5s"):
            for label, values in summary["fresh20"].items():
                data = values["windows"][window]
                text = " | ".join("n/a" if data[key] is None else f"{data[key]:.6g}" for key in
                                  ("current_rms_A", "change_rms_A", "total_variation_A", "hf_rms_A"))
                lines.append(f"| {window} | {label} | {data['paired_cases']} | {text} |")
        lines += ["", "## All Discovered Local Poles", "", "| Model | Arm (deg) | Pole (1/s) | Magnitude | Damping |",
                  "|---|---:|---|---:|---:|"]
        for label, data in poles.items():
            for equilibrium in data["equilibria"]:
                for mode in equilibrium["modes"]:
                    if mode["continuous_imag"] < 0:
                        continue
                    real, imag = mode["continuous_real"], mode["continuous_imag"]
                    lines.append(f"| {label} | {equilibrium['arm_angle_deg']:.4f} | {real:.4f} +/- {imag:.4f}j | "
                                 f"{np.hypot(real, imag):.4f} | {mode['damping_ratio']:.4f} |")
        lines += ["", "Poles are principal-log equivalents of a5ms map, not global stability or hardware guarantees.",
                  "Root scan is non-exhaustive; inspect every root in evaluation/poles.json. No learned gain target is guaranteed.",
                  "Incomplete current windows are excluded pairwise; failures remain in counts.5s results are diagnostic, not a speed gate.",
                  "Inspect paired plots/traces in evaluation/. Reward curves differ in cost weights and are not directly comparable."]
        (job / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        atomic_write_json(job / "completed.json", {"state": "completed", "report": str(job / "REPORT.md")})
        status(job, "completed", "report")
    except Exception:
        failure(job)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only:
        evaluation_job(args.job_dir, args.execute)
    else:
        training_job(args.job_dir, args.execute)
