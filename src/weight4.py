"""Train nominal weight-4 SAC and compare frozen weights 3, 4, and 5."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
from stable_baselines3 import SAC

from evaluate import make_episode, run_prepared_episode, save_detailed_episode, write_csv
from furuta_env import wrap_angle
from nominal import CONFIG, DEFAULT_JOB as BASELINE_JOB, run as run_nominal
from smooth_current import DEFAULT_JOB as WEIGHT5_JOB, METRICS, validate_baseline, window_metrics
from train import algorithm_settings, atomic_write_json, file_sha256
from validate_smooth_current import current_metrics, horizon_metrics


DEFAULT_JOB = BASELINE_JOB.parent / "nominal_sac_actor64_delta4_seed0_v0"
LABELS = {"weight3": 3.0, "weight4": 4.0, "weight5": 5.0}
FRESH_SEEDS = range(280000, 280100)
CORE_FILES = ("train.py", "furuta_env.py", "furuta_model.py", "controllers.py", "evaluate.py")


def validate_references():
    validate_baseline(BASELINE_JOB)
    training = WEIGHT5_JOB / "candidate/training"
    saved = json.loads((training / "config.json").read_text(encoding="utf-8"))
    if saved != asdict(replace(CONFIG, action_change_weight=5.0)):
        raise ValueError("Weight-5 task differs from intended nominal configuration")
    settings = json.loads((training / "training.json").read_text(encoding="utf-8"))
    expected = algorithm_settings(CONFIG, total_timesteps=2_000_000, actor_network=[64, 64], critic_network=[64, 64])
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Weight-5 learner settings differ")
    for name in CORE_FILES:
        if file_sha256(WEIGHT5_JOB / "candidate/source" / name) != file_sha256(Path(__file__).parent / name):
            raise ValueError(f"Weight-5 core code differs: {name}")
    paths = {
        "weight3": BASELINE_JOB / "training/best/best_model.zip",
        "weight5": training / "best/best_model.zip",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def build_cases():
    cases = [{"suite": "diagnostic", "name": f"seed_{seed}", "seed": seed,
              "duration_s": 5.0, "state": None} for seed in range(80000, 80100)]
    center = make_episode(180051, CONFIG, False, "training").x.copy()
    for offsets in product((-1, 0, 1), repeat=4):
        state = center + np.asarray(offsets) * np.array([0.035, 0.035, 0.25, 0.25])
        state[1] = wrap_angle(state[1])
        if (abs(state[0]) > CONFIG.reset_theta1_half_range
                or abs(state[2]) > CONFIG.reset_omega1_half_range
                or abs(state[3]) > CONFIG.reset_omega2_half_range):
            continue
        cases.append({"suite": "neighborhood", "name": "offset_" + "_".join(map(str, offsets)),
                      "seed": 180051, "duration_s": 5.0, "state": state.tolist()})
    cases.extend({"suite": "fresh_long", "name": f"seed_{seed}", "seed": seed,
                  "duration_s": 30.0, "state": None} for seed in FRESH_SEEDS)
    return cases


def statistics(values):
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": max(values)} if values else None


def summarize(episodes, currents):
    summary = {}
    for suite in dict.fromkeys(row["suite"] for row in episodes):
        summary[suite] = {}
        for label in LABELS:
            selected = [row for row in episodes if row["suite"] == suite and row["model"] == label]
            summary[suite][label] = {
                "episodes": len(selected),
                **{key: sum(row[key] for row in selected) for key in (
                    "success_at_5s", "success_at_end", "unsafe", "balance_lost_after_5s",
                )},
                "finish_time_s": statistics([row["finish_time_s"] for row in selected if row["success_at_end"]]),
                "flagged_cases": [row["case"] for row in selected if not row["success_at_5s"] or not row["success_at_end"] or row["unsafe"] or row["balance_lost_after_5s"]],
                "windows": {},
            }
        windows = dict.fromkeys(row["window"] for row in currents if row["suite"] == suite)
        for window in windows:
            selected = [row for row in currents if row["suite"] == suite and row["window"] == window]
            shared = set.intersection(*({row["case"] for row in selected if row["model"] == label} for label in LABELS))
            for label in LABELS:
                available = [row for row in selected if row["model"] == label]
                paired = [row for row in available if row["case"] in shared]
                summary[suite][label]["windows"][window] = {
                    "available_episodes": len(available), "paired_episodes": len(paired),
                    "available": {metric: statistics([row[metric] for row in available]) for metric in METRICS},
                    "paired": {metric: statistics([row[metric] for row in paired]) for metric in METRICS},
                }
    return summary


def write_report(job, summary):
    lines = ["# Weight-4 Nominal SAC Comparison", "",
             "Fresh seed-0 training, 2M decisions, only action-change weight changed to 4. No evaluation-based checkpoint selection.", "",
             "| Suite | Model | Success at 5 s | Success at end | Unsafe | Lost balance after 5 s | Mean / max successful finish (s) |",
             "|---|---|---|---|---|---|---|"]
    reliable = True
    for suite, models in summary.items():
        for label, result in models.items():
            timing = result["finish_time_s"]
            times = f"{timing['mean']:.3f} / {timing['max']:.3f}" if timing else "n/a"
            lines.append(f"| {suite} | {label} | {result['success_at_5s']}/{result['episodes']} | {result['success_at_end']}/{result['episodes']} | {result['unsafe']} | {result['balance_lost_after_5s']} | {times} |")
            if label == "weight4" and result["flagged_cases"]:
                reliable = False
    lines.extend(["", "## Paired Current Metrics", "",
                  "Only cases with available windows in all three policies are paired; excluded unsafe/truncated cases still count against reliability.", "",
                  "| Suite / window | Model | Paired cases | Delta RMS mean / p95 / max (A) | HF fraction mean | TV mean (A) |",
                  "|---|---|---|---|---|---|"])
    for suite, models in summary.items():
        for label, result in models.items():
            for window, data in result["windows"].items():
                values = data["paired"]
                if values["change_rms_A"] is None:
                    continue
                rms = " / ".join(f"{values['change_rms_A'][key]:.6g}" for key in ("mean", "p95", "max"))
                lines.append(f"| {suite} / {window} | {label} | {data['paired_episodes']} | {rms} | {values['hf_power_fraction']['mean']:.6g} | {values['total_variation_A']['mean']:.6g} |")
    lines.extend(["", "## Assessment", "",
                  "Weight 4 passed the observed reliability gate." if reliable else "Weight 4 FAILED the observed reliability gate. Keep weight 3 as the reference; do not promote based on smoother commands.",
                  "Even a reliability pass is not automatic promotion: inspect paired RMS/HF metrics, p95/max tails, settling times, and neighborhood cases before deciding whether the tradeoff is worthwhile.",
                  "Diagnostic seeds 80000-80099 and the seed-180051 neighborhood are intentionally reused. Fresh seeds are 280000-280099, 30 s each. Neighborhood offsets are +/-0.035 rad per angle and +/-0.25 rad/s per velocity; combinations outside nominal reset support are omitted, not clipped.",
                  "First_5s and last_5s require complete windows. Capture windows follow each policy's capture time and exclude missing capture, but may be truncated by unsafe termination. HF fraction is relative >=25 Hz power, not absolute ripple amplitude. Finish statistics include successful cases only. Inspect summary.json and episodes.csv for all counts and flagged cases.",
                  "This is one training seed on the nominal plant, not hardware or parameter-robustness qualification. No model is automatically promoted."])
    (job / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(job, cases, progress):
    episodes = []
    currents = []
    for label, weight in LABELS.items():
        model = SAC.load(job / "reference" / f"{label}.zip", device="cpu")
        for index, case in enumerate(cases, 1):
            config = replace(CONFIG, episode_time=case["duration_s"], action_change_weight=weight)
            plant = make_episode(case["seed"], config, False, "training")
            if case["state"] is not None:
                plant.reset(np.asarray(case["state"], dtype=np.float64))
            result = run_prepared_episode(model, plant, config, "sac")
            horizon = horizon_metrics(result, config)
            row = {"model": label, "suite": case["suite"], "case": case["name"], "seed": case["seed"],
                   "success_at_5s": horizon["success_at_5s"], "success_at_end": int(result["success"]),
                   "unsafe": int(result["unsafe"]), "balance_lost_after_5s": horizon["balance_lost_after_5s"],
                   "finish_time_s": result["finish_time"] if result["success"] else None,
                   "duration_s": result["trace"][-1]["time_s"]}
            episodes.append(row)
            windows = {"first_5s": current_metrics(result, config, 0, 5),
                       "capture_window": window_metrics(result, config, "capture_window")}
            if case["duration_s"] == 30.0:
                windows["last_5s"] = current_metrics(result, config, 25, 30)
            for window, metrics in windows.items():
                if metrics is not None and metrics["command_samples"]:
                    currents.append({"model": label, "suite": case["suite"], "case": case["name"], "window": window, **metrics})
            if (index == 1 or case["seed"] in (80015, 280000) or case["name"] == "offset_0_0_0_0"
                    or not row["success_at_5s"] or not row["success_at_end"] or row["balance_lost_after_5s"]):
                save_detailed_episode(job / "comparison" / case["suite"] / label, label, index, case["seed"], result, config)
            if index % 10 == 0 or index == len(cases):
                write_csv(job / "comparison/episodes.csv", tuple(episodes[0]), episodes)
                if currents:
                    write_csv(job / "comparison/current_windows.csv", tuple(currents[0]), currents)
                progress(label, case["suite"], index, len(cases))
    summary = summarize(episodes, currents)
    atomic_write_json(job / "summary.json", summary)
    write_report(job, summary)


def run(job=DEFAULT_JOB, execute=False):
    job = job.resolve()
    if job.exists():
        raise FileExistsError(f"Refusing existing output: {job}")
    references = validate_references()
    cases = build_cases()
    print(f"Fresh weight-4 nominal SAC, 2M decisions, automatic three-policy evaluation.\nOutput: {job}\nComparison cases per policy: {len(cases)}", flush=True)
    if not execute:
        print("Dry run. Add --execute to start.", flush=True)
        return
    job.mkdir(parents=True)

    def status(state, phase, **details):
        atomic_write_json(job / "status.json", {"state": state, "phase": phase,
                          "updated_utc": datetime.now(timezone.utc).isoformat(), **details})

    try:
        status("running", "preparation")
        (job / "reference").mkdir()
        (job / "source").mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, job / "source" / path.name)
        for label, path in references.items():
            shutil.copy2(path, job / "reference" / f"{label}.zip")
        atomic_write_json(job / "protocol.json", {
            "training": {"weight": 4.0, "seed": 0, "decisions": 2_000_000, "gamma": 0.99, "actor": [64, 64], "critics": [64, 64]},
            "config": asdict(replace(CONFIG, action_change_weight=4.0)), "cases": cases,
            "reference_sha256": {label: file_sha256(job / "reference" / f"{label}.zip") for label in references},
            "selection": "Existing 50-episode training validation only; no evaluation-based checkpoint selection or automatic promotion.",
            "reliability_gate": "All weight-4 cases succeed by 5 s and at end, zero unsafe and zero loss of balance after 5 s.",
        })
        status("running", "training_and_nominal_evaluation")
        run_nominal(job / "candidate", execute=True, action_change_weight=4.0)
        shutil.copy2(job / "candidate/training/best/best_model.zip", job / "reference/weight4.zip")
        atomic_write_json(job / "model_hashes.json", {label: file_sha256(job / "reference" / f"{label}.zip") for label in LABELS})
        status("running", "comparison")

        def progress(label, suite, index, total):
            status("running", "comparison", model=label, suite=suite, cases_completed=index, cases_total=total)
            print(f"Comparison {label}: {index}/{total} ({suite})", flush=True)

        compare(job, cases, progress)
        status("completed", "report", report=str(job / "FINDINGS.md"))
    except Exception:
        failure = traceback.format_exc()
        status("failed", "error", error=failure)
        (job / "ERROR.txt").write_text(failure, encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args.job_dir, args.execute)
