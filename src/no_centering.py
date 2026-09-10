"""Train weight-4 nominal SAC without a preferred arm position."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
from uuid import uuid4

import numpy as np
from stable_baselines3 import SAC

from diagnose_centering import diagnose
from evaluate import make_episode, run_prepared_episode, save_detailed_episode, write_csv
from nominal import CONFIG, DEFAULT_JOB as BASELINE_JOB, run as run_nominal
from train import atomic_write_json, file_sha256
from validate_smooth_current import current_metrics
from weight4 import DEFAULT_JOB as WEIGHT4_JOB, build_cases, statistics


DEFAULT_JOB = BASELINE_JOB.parent / "nominal_sac_no_centering_delta4_seed0_v0"
TASK_CONFIG = replace(CONFIG, arm_angle_weight=0.0, balance_theta1=CONFIG.arm_angle_limit, action_change_weight=4.0)
LABELS = ("weight3_centered", "weight4_centered", "weight4_free")


def status(job, state, phase, **details):
    payload = {"state": state, "phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), **details}
    try:
        atomic_write_json(job / "status.json", payload)
    except PermissionError:
        atomic_write_json(job / f"status_event_{uuid4().hex}.json", payload)


def score(result, config):
    rows = result["trace"][1:]
    times = np.array([float(row["time_s"]) for row in rows])
    safe = np.array([not row["unsafe"] and abs(row["theta1_rad"]) <= config.arm_angle_limit for row in rows])
    free = safe & np.array([
        abs(row["upright_error_rad"]) <= config.balance_angle
        and abs(row["omega1_rad_s"]) <= config.balance_omega1
        and abs(row["omega2_rad_s"]) <= config.balance_omega2 for row in rows
    ])
    centered = free & np.array([abs(row["theta1_rad"]) <= CONFIG.balance_theta1 for row in rows])
    hold = int(round(config.balance_hold_time / config.sample_time))
    first_steps = int(round(5.0 / config.sample_time))
    result_metrics = {"unsafe": int(result["unsafe"]),
                      "minimum_travel_margin_rad": config.arm_angle_limit - max(abs(row["theta1_rad"]) for row in result["trace"])}
    for label, valid in (("free", free), ("centered", centered)):
        at_five = bool(len(valid) >= first_steps and np.all(safe[:first_steps]) and np.all(valid[first_steps - hold:first_steps]))
        at_end = bool(not result["unsafe"] and len(valid) >= hold and np.all(valid[-hold:]))
        bad = np.flatnonzero(~valid)
        start = float(times[bad[-1]]) if bad.size else 0.0
        result_metrics.update({
            f"{label}_success_5s": int(at_five), f"{label}_success_end": int(at_end),
            f"{label}_lost_after_5s": int(at_five and np.any(~valid[first_steps:])),
            f"{label}_finish_s": start + config.balance_hold_time if at_end else None,
        })
    return result_metrics


def experiment_cases():
    cases = build_cases()
    for case in cases:
        if case["suite"] == "fresh_long":
            case["seed"] += 100000
            case["name"] = f"seed_{case['seed']}"
    cases.extend({"suite": "slow_diagnostic", "seed": seed, "name": f"seed_{seed}",
                  "duration_s": 30.0, "state": None} for seed in (280022, 280029, 280076))
    return cases


def compare(job, cases):
    episodes = []
    currents = []
    for label in LABELS:
        model = SAC.load(job / "reference" / f"{label}.zip", device="cpu")
        for index, case in enumerate(cases, 1):
            config = replace(TASK_CONFIG, episode_time=case["duration_s"])
            plant = make_episode(case["seed"], config, False, "training")
            if case["state"] is not None:
                plant.reset(np.array(case["state"], dtype=np.float64))
            result = run_prepared_episode(model, plant, config, "sac")
            metrics = score(result, config)
            episodes.append({"model": label, "suite": case["suite"], "case": case["name"], "seed": case["seed"], **metrics})
            for window, start, end in (("first_5s", 0, 5), ("last_5s", 25, 30)):
                if end > case["duration_s"]:
                    continue
                current = current_metrics(result, config, start, end)
                if current is not None:
                    currents.append({"model": label, "suite": case["suite"], "case": case["name"], "window": window, **current})
            if (index == 1 or case["seed"] in (80015, 380000, 280022, 280029, 280076)
                    or case["name"] == "offset_0_0_0_0" or not metrics["free_success_5s"]
                    or not metrics["free_success_end"] or metrics["free_lost_after_5s"]):
                save_detailed_episode(job / "comparison" / case["suite"] / label, label, index, case["seed"], result, config)
            if index % 10 == 0 or index == len(cases):
                write_csv(job / "comparison/episodes.csv", tuple(episodes[0]), episodes)
                if currents:
                    write_csv(job / "comparison/current_windows.csv", tuple(currents[0]), currents)
                status(job, "running", "comparison", model=label, completed=index, total=len(cases))
                print(f"Comparison {label}: {index}/{len(cases)}", flush=True)
    summary = {}
    lines = ["# No-Centering Nominal SAC", "",
             "New task: zero arm-angle cost; success accepts any arm position inside unchanged travel limits. Velocity, upright, one-second hold, current, and safety limits unchanged. Weight 4, seed 0, 2M decisions. No automatic promotion.", "",
             "All frozen models are evaluated on identical nominal trajectories under both success definitions. Standard candidate/evaluation uses the NEW task and is not directly comparable with older centered success counts.", "",
             "| Suite | Model | Free success 5s / end | Centered success 5s / end | Unsafe | Free balance lost after 5s | Free finish mean / max (s) |", "|---|---|---|---|---|---|---|"]
    for suite in dict.fromkeys(row["suite"] for row in episodes):
        summary[suite] = {}
        for label in LABELS:
            selected = [row for row in episodes if row["suite"] == suite and row["model"] == label]
            counts = {key: sum(row[key] for row in selected) for key in (
                "unsafe", "free_success_5s", "free_success_end", "free_lost_after_5s",
                "centered_success_5s", "centered_success_end", "centered_lost_after_5s")}
            timing = statistics([row["free_finish_s"] for row in selected if row["free_success_end"]])
            data = {"episodes": len(selected), **counts, "free_finish_s": timing,
                    "minimum_travel_margin_rad": min(row["minimum_travel_margin_rad"] for row in selected),
                    "flagged_cases": [row["case"] for row in selected if not row["free_success_5s"] or not row["free_success_end"] or row["unsafe"] or row["free_lost_after_5s"]], "paired_current": {}}
            for window in ("first_5s", "last_5s"):
                available = [row for row in currents if row["suite"] == suite and row["window"] == window]
                shared = set.intersection(*({row["case"] for row in available if row["model"] == model_label} for model_label in LABELS))
                paired = [row for row in available if row["model"] == label and row["case"] in shared]
                data["paired_current"][window] = {"episodes": len(paired), **{metric: statistics([row[metric] for row in paired]) for metric in ("change_rms_A", "total_variation_A", "hf_power_fraction")}}
            summary[suite][label] = data
            timing_text = f"{timing['mean']:.3f} / {timing['max']:.3f}" if timing else "n/a"
            lines.append(f"| {suite} | {label} | {counts['free_success_5s']} / {counts['free_success_end']} of {len(selected)} | {counts['centered_success_5s']} / {counts['centered_success_end']} of {len(selected)} | {counts['unsafe']} | {counts['free_lost_after_5s']} | {timing_text} |")
    lines.extend(["", "## Paired First-Five-Second Current", "",
                  "| Suite | Model | Paired cases | Change RMS mean / p95 / max (A) | TV mean (A) | HF fraction mean |", "|---|---|---|---|---|---|"])
    for suite, models in summary.items():
        for label, data in models.items():
            current = data["paired_current"]["first_5s"]
            if not current["episodes"]:
                continue
            rms = " / ".join(f"{current['change_rms_A'][key]:.6g}" for key in ("mean", "p95", "max"))
            lines.append(f"| {suite} | {label} | {current['episodes']} | {rms} | {current['total_variation_A']['mean']:.6g} | {current['hf_power_fraction']['mean']:.6g} |")
    passed = all(not models["weight4_free"]["flagged_cases"] for models in summary.values())
    lines.extend(["", "Observed new-task reliability gate PASSED." if passed else "Observed new-task reliability gate FAILED; do not promote the candidate.",
                  "Paired current windows exclude incomplete windows in ANY model; unsafe cases still count as failures. Inspect all episode counts, travel margins, flagged cases, and last-5s metrics in summary.json. HF power fraction alone is not absolute jitter amplitude.",
                  "Fresh seeds 380000-380099 run for 30s; the remaining suites are reused diagnostics. Changed task, one training seed, nominal simulator only. No hardware safety guarantee. No new boundary penalty was introduced; a near-limit equilibrium can pass and should be assessed before deployment."])
    atomic_write_json(job / "summary.json", summary)
    (job / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(job=DEFAULT_JOB, execute=False):
    job = job.resolve()
    if job.exists():
        raise FileExistsError(f"Refusing existing output: {job}")
    references = {"weight3_centered": BASELINE_JOB / "training", "weight4_centered": WEIGHT4_JOB / "candidate/training"}
    for label, directory in references.items():
        weight = 3.0 if label == "weight3_centered" else 4.0
        if json.loads((directory / "config.json").read_text()) != asdict(replace(CONFIG, action_change_weight=weight)):
            raise ValueError(f"Reference configuration mismatch: {label}")
        if not (directory / "best/best_model.zip").is_file():
            raise FileNotFoundError(directory / "best/best_model.zip")
    cases = experiment_cases()
    print(f"No centering: fresh seed0 weight4 2M decisions; {len(cases)} comparison cases per model.\nOutput: {job}", flush=True)
    if not execute:
        print("Dry run. Add --execute.", flush=True)
        return
    job.mkdir(parents=True)
    try:
        status(job, "running", "preparation")
        (job / "source").mkdir()
        (job / "reference").mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, job / "source" / path.name)
        for label, directory in references.items():
            shutil.copy2(directory / "best/best_model.zip", job / "reference" / f"{label}.zip")
            shutil.copy2(directory / "config.json", job / "reference" / f"{label}_config.json")
        atomic_write_json(job / "protocol.json", {"config": asdict(TASK_CONFIG), "cases": cases,
            "changed_fields_from_weight4": {"arm_angle_weight": [1.5, 0.0], "balance_theta1": [0.25, CONFIG.arm_angle_limit]},
            "selection": "50 fixed training-validation episodes on NEW task, success then reward. No holdout selection or automatic promotion.",
            "reference_sha256": {label: file_sha256(job / "reference" / f"{label}.zip") for label in references},
            "saved_trajectory_diagnosis": diagnose()})
        status(job, "running", "training_and_nominal_evaluation")
        run_nominal(job / "candidate", execute=True, action_change_weight=4.0, task_config=TASK_CONFIG)
        shutil.copy2(job / "candidate/training/best/best_model.zip", job / "reference/weight4_free.zip")
        atomic_write_json(job / "model_hashes.json", {label: file_sha256(job / "reference" / f"{label}.zip") for label in LABELS})
        compare(job, cases)
        atomic_write_json(job / "completed.json", {"state": "completed", "report": str(job / "FINDINGS.md"), "updated_utc": datetime.now(timezone.utc).isoformat()})
        status(job, "completed", "report", report=str(job / "FINDINGS.md"))
    except Exception:
        failure = traceback.format_exc()
        atomic_write_json(job / "failed.json", {"error": failure})
        status(job, "failed", "error", error=failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    args = parser.parse_args()
    run(args.job_dir, args.execute)
