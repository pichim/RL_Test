"""Train and qualify one nominal SAC current-penalty experiment."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np
from stable_baselines3 import SAC

from evaluate import main as evaluate, make_episode, run_prepared_episode, save_detailed_episode, write_csv
from furuta_env import FurutaConfig
from no_centering import score, status
from nominal import CONFIG, DEFAULT_JOB as BASELINE_JOB
from train import algorithm_settings, atomic_write_json, file_sha256, train
from validate_smooth_current import current_metrics
from weight4 import build_cases, statistics


DEFAULT_JOB = BASELINE_JOB.parent / "nominal_sac_current15_delta3_seed0_v0"
ANALYSIS_PYTHON = BASELINE_JOB.parent / "lqr_study_env/Scripts/python.exe"
TASK_CONFIG = replace(CONFIG, current_weight=1.5)
LABELS = ("baseline", "candidate")
TRAINING = {"seed": 0, "total_timesteps": 2_000_000, "gamma": .99,
            "actor_network": [64, 64], "critic_network": [64, 64], "evaluation_episodes": 50}
CURRENT_METRICS = ("change_rms_A", "total_variation_A", "current_rms_A", "peak_current_A", "ac_rms_A", "hf_rms_A")


def cases():
    result = build_cases()
    for case in result:
        if case["suite"] == "fresh_long":
            case["seed"] += 200000
            case["name"] = f"seed_{case['seed']}"
    result.extend({"suite": "slow_diagnostic", "seed": seed, "name": f"seed_{seed}",
                   "duration_s": 30.0, "state": None} for seed in (280022, 280029, 280076))
    return result


def preflight():
    training = BASELINE_JOB / "training"
    saved = json.loads((training / "config.json").read_text())
    if FurutaConfig(**saved) != CONFIG:
        raise ValueError("Baseline task differs from nominal weight-3 configuration")
    settings = json.loads((training / "training.json").read_text())
    expected = algorithm_settings(CONFIG, total_timesteps=2_000_000, actor_network=[64, 64], critic_network=[64, 64])
    if any(settings.get(key) != value for key, value in expected.items()) or settings["seed"] != 0:
        raise ValueError("Baseline learner settings differ")
    baseline = training / "best/best_model.zip"
    if not baseline.is_file() or not ANALYSIS_PYTHON.is_file():
        raise FileNotFoundError("Baseline model or isolated analysis interpreter missing")
    changes = {name: [asdict(CONFIG)[name], value] for name, value in asdict(TASK_CONFIG).items()
               if asdict(CONFIG)[name] != value}
    if changes != {"current_weight": [1.0, 1.5]}:
        raise ValueError(f"Unexpected task changes: {changes}")
    source_comparison = {}
    for name in ("train.py", "furuta_env.py", "furuta_model.py", "controllers.py", "evaluate.py"):
        old = file_sha256(BASELINE_JOB / "source" / name)
        new = file_sha256(Path(__file__).with_name(name))
        source_comparison[name] = {"baseline_sha256": old, "current_sha256": new}
        if old != new and name != "furuta_env.py":
            raise ValueError(f"Unreviewed baseline core source difference: {name}")
    model = SAC.load(baseline, device="cpu")
    plant = make_episode(80000, CONFIG, False, "training")
    smoke = run_prepared_episode(model, plant, replace(CONFIG, episode_time=.02), "sac")
    if not smoke["trace"] or smoke["unsafe"]:
        raise RuntimeError("Baseline runtime smoke check failed")
    return {"changes": changes, "baseline_sha256": file_sha256(baseline), "core_source": source_comparison,
            "reviewed_environment_changes": "Nonnegative arm-angle validation and configurable current_weight with legacy default 1; focused reward tests required."}


def window_current(result, config, start, end):
    metrics = current_metrics(result, config, start, end)
    if metrics is None:
        return None
    first = int(round(start / config.sample_time))
    last = int(round(end / config.sample_time))
    commands = np.array([row["command_current_A"] for row in result["trace"][first + 1:last + 1:config.action_repeat]])
    ac_rms = float(np.std(commands))
    power = abs(np.fft.rfft(commands - np.mean(commands)))**2
    power[1:] *= 2
    if len(commands) % 2 == 0:
        power[-1] *= .5
    frequencies = np.fft.rfftfreq(len(commands), config.sample_time * config.action_repeat)
    hf_rms = float(np.sqrt(np.sum(power[frequencies >= 25.0])) / len(commands))
    return {**metrics, "current_rms_A": float(np.sqrt(np.mean(commands**2))),
            "peak_current_A": float(np.max(abs(commands))), "ac_rms_A": ac_rms,
            "hf_rms_A": hf_rms}


def summarize(episodes, currents):
    summary = {}
    for suite in dict.fromkeys(row["suite"] for row in episodes):
        summary[suite] = {}
        for label in LABELS:
            selected = [row for row in episodes if row["suite"] == suite and row["model"] == label]
            data = {"episodes": len(selected), **{key: sum(row[key] for row in selected) for key in (
                "unsafe", "centered_success_5s", "centered_success_end", "centered_lost_after_5s")},
                "finish_s": statistics([row["centered_finish_s"] for row in selected if row["centered_success_end"]]),
                "minimum_travel_margin_rad": min(row["minimum_travel_margin_rad"] for row in selected),
                "flagged_cases": [row["case"] for row in selected if not row["centered_success_5s"]
                                  or not row["centered_success_end"] or row["unsafe"] or row["centered_lost_after_5s"]],
                "windows": {}}
            for window in ("first_5s", "last_5s"):
                available = [row for row in currents if row["suite"] == suite and row["window"] == window]
                shared = set.intersection(*({row["case"] for row in available if row["model"] == name} for name in LABELS))
                paired = [row for row in available if row["model"] == label and row["case"] in shared]
                data["windows"][window] = {"paired_episodes": len(paired),
                    **{metric: statistics([row[metric] for row in paired]) for metric in CURRENT_METRICS}}
            summary[suite][label] = data
    return summary


def compare(job, protocol_cases):
    episodes, currents = [], []
    for label in LABELS:
        model = SAC.load(job / "reference" / f"{label}.zip", device="cpu")
        for index, case in enumerate(protocol_cases, 1):
            config = replace(CONFIG if label == "baseline" else TASK_CONFIG, episode_time=case["duration_s"])
            plant = make_episode(case["seed"], config, False, "training")
            if case["state"] is not None:
                plant.reset(np.asarray(case["state"], dtype=float))
            initial = plant.x.tolist()
            result = run_prepared_episode(model, plant, config, "sac")
            metrics = score(result, config)
            episodes.append({"model": label, "suite": case["suite"], "case": case["name"],
                             "seed": case["seed"], "initial_state": json.dumps(initial), **metrics})
            for window, start, end in (("first_5s", 0, 5), ("last_5s", 25, 30)):
                if end <= case["duration_s"]:
                    current = window_current(result, config, start, end)
                    if current is not None:
                        currents.append({"model": label, "suite": case["suite"], "case": case["name"], "window": window, **current})
            if (index <= 5 or case["seed"] in (80015, 480000, 280022, 280029, 280076)
                    or case["name"] == "offset_0_0_0_0" or not metrics["centered_success_5s"]
                    or not metrics["centered_success_end"] or metrics["centered_lost_after_5s"] or metrics["unsafe"]):
                save_detailed_episode(job / "comparison" / case["suite"] / label, label, index, case["seed"], result, config)
            if index % 10 == 0 or index == len(protocol_cases):
                write_csv(job / "comparison/episodes.csv", tuple(episodes[0]), episodes)
                if currents:
                    write_csv(job / "comparison/current_windows.csv", tuple(currents[0]), currents)
                status(job, "running", "comparison", model=label, completed=index, total=len(protocol_cases))
                print(f"Comparison {label}: {index}/{len(protocol_cases)}", flush=True)
    summary = summarize(episodes, currents)
    atomic_write_json(job / "summary.json", summary)
    return summary


def analyze(job):
    from analyze_damping import policy_analysis
    from lqr_weight_study import plant_matrices, pole_metrics
    from furuta_env import LQR_GAIN

    _, _, transition, action_input = plant_matrices()
    result = {"held_lqr_target": pole_metrics(transition - action_input @ (LQR_GAIN[None, :] / CONFIG.max_current), .005),
              "models": {}, "method": "Local 5ms map, previous action included; non-exhaustive stationary root scan, autograd/finite-difference check."}
    for label in LABELS:
        model = SAC.load(job / "reference" / f"{label}.zip", device="cpu")
        data = policy_analysis(model)
        for index, equilibrium in enumerate(data["equilibria"], 1):
            stable = all(np.hypot(mode["discrete_real"], mode["discrete_imag"]) < 1 for mode in equilibrium["modes"])
            equilibrium["stable"] = bool(stable)
            if stable and abs(equilibrium["arm_angle_rad"]) <= CONFIG.balance_theta1:
                for direction in (-1, 1):
                    config = replace(CONFIG, episode_time=10.0)
                    plant = make_episode(0, config, False, "training")
                    plant.reset(np.array([equilibrium["arm_angle_rad"] + direction * .02, np.pi, 0, 0]))
                    response = run_prepared_episode(model, plant, config, "sac")
                    save_detailed_episode(job / "poles" / label / f"offset_{direction}", label, index, 0, response, config)
        result["models"][label] = {"sha256": file_sha256(job / "reference" / f"{label}.zip"), **data}
    atomic_write_json(job / "poles/analysis.json", result)


def review(job, summary):
    poles = json.loads((job / "poles/analysis.json").read_text())
    suitable = []
    for equilibrium in poles["models"]["candidate"]["equilibria"]:
        if not equilibrium["stable"] or abs(equilibrium["arm_angle_rad"]) > CONFIG.balance_theta1:
            continue
        slow = max(equilibrium["modes"], key=lambda mode: mode["continuous_real"])
        oscillatory = [mode for mode in equilibrium["modes"] if abs(mode["continuous_imag"]) > 1e-6]
        slow_pair = max(oscillatory, key=lambda mode: mode["continuous_real"]) if oscillatory else None
        if (-slow["continuous_real"] >= .9 * 3.8712781982
                and (slow_pair is None or slow_pair["damping_ratio"] >= .7)):
            suitable.append(equilibrium["arm_angle_deg"])
    reliable = all(not suite["candidate"]["flagged_cases"] for suite in summary.values())
    fresh = summary["fresh_long"]
    windows = {label: fresh[label]["windows"]["first_5s"] for label in LABELS}
    paired = windows["candidate"]["paired_episodes"] == 100
    smooth = paired and all(windows["candidate"][metric]["mean"] <= windows["baseline"][metric]["mean"]
                            for metric in ("change_rms_A", "total_variation_A", "hf_rms_A"))
    timing = all(fresh[label]["finish_s"] is not None for label in LABELS)
    timing = timing and fresh["candidate"]["finish_s"]["mean"] <= fresh["baseline"]["finish_s"]["mean"] * 1.1
    gates = {"all_case_reliability": reliable, "local_pole_screen": bool(suitable),
             "fresh_current_nonregression": bool(smooth), "fresh_mean_finish_within_10_percent": bool(timing),
             "suitable_equilibrium_angles_deg": suitable, "automatic_promotion": False}
    atomic_write_json(job / "review.json", gates)
    lines = ["# Current-Penalty 1.5 SAC Experiment", "",
             "Fresh seed 0, 2M decisions; current coefficient 1 -> 1.5 only. Delta-action coefficient 3, centered task and limits unchanged.",
             "Best checkpoint selected on training validation only. No automatic promotion. One training seed; nominal simulation is not a hardware guarantee.", "",
             "## Predeclared Screens", "", "```json", json.dumps(gates, indent=2), "```", "",
             "Pole screen requires a stable centered equilibrium, slowest decay at least 90% of weight-3 reference, and slowest oscillatory pair damping >=0.7.",
             "Root scan is non-exhaustive; a passing local screen is not global stability. Review every root and small-signal trace in poles/.", "",
             "| Suite | Model | Success 5s/end | Unsafe | Late losses | Finish mean/max (s) |", "|---|---|---|---:|---:|---|"]
    for suite, models in summary.items():
        for label, data in models.items():
            timing_data = data["finish_s"]
            text = f"{timing_data['mean']:.3f}/{timing_data['max']:.3f}" if timing_data else "n/a"
            lines.append(f"| {suite} | {label} | {data['centered_success_5s']}/{data['centered_success_end']} of {data['episodes']} | {data['unsafe']} | {data['centered_lost_after_5s']} | {text} |")
    lines += ["", "## Fresh Paired Current Windows", "",
              "| Window | Model | Paired cases | Delta RMS (A) | TV (A) | Current RMS (A) | HF RMS (A) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for window in ("first_5s", "last_5s"):
        for label in LABELS:
            data = fresh[label]["windows"][window]
            values = [f"{data[metric]['mean']:.6g}" if data[metric] else "n/a"
                      for metric in ("change_rms_A", "total_variation_A", "current_rms_A", "hf_rms_A")]
            lines.append(f"| {window} | {label} | {data['paired_episodes']} | " + " | ".join(values) + " |")
    lines += ["", "## Local Poles", "",
              "All discovered stationary roots are included; +/- rows represent conjugate pairs.", "",
              "| Model | Arm equilibrium (deg) | Stable | Pole (1/s) | omega0 (rad/s) | Damping |",
              "|---|---:|---|---|---:|---:|"]
    for label, data in poles["models"].items():
        for equilibrium in data["equilibria"]:
            for mode in equilibrium["modes"]:
                if mode["continuous_imag"] < 0:
                    continue
                real, imag = mode["continuous_real"], mode["continuous_imag"]
                text = f"{real:.3f} +/- {imag:.3f}j" if imag else f"{real:.3f}"
                lines.append(f"| {label} | {equilibrium['arm_angle_deg']:.4f} | {equilibrium['stable']} | {text} | {np.hypot(real, imag):.3f} | {mode['damping_ratio']:.3f} |")
    lines += ["", "Current windows exclude incomplete windows in either model; failures remain in reliability counts.",
              "Inspect summary.json for tail metrics and travel margins; comparison CSVs retain paired initial states and all failures.",
              "HF RMS is the absolute >=25Hz unwindowed FFT-band RMS with Parseval/Nyquist weighting, not a relative power fraction.",
              "Diagnostic and neighborhood cases are reused; only seeds480000-480099 are fresh. Training reward is not compared across different cost weights."]
    (job / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gates


def evaluate_job(job=DEFAULT_JOB, execute=False):
    job = Path(job).resolve()
    trained = json.loads((job / "training_completed.json").read_text())
    protocol = json.loads((job / "protocol.json").read_text())
    if protocol["config"] != asdict(TASK_CONFIG) or protocol["training"] != TRAINING:
        raise ValueError("Saved protocol differs; use the saved source/current_penalty.py")
    analysis_python = Path(protocol["analysis_python"])
    if not analysis_python.is_file():
        raise FileNotFoundError(f"Saved analysis interpreter missing: {analysis_python}")
    for name, expected in protocol["source_sha256"].items():
        if file_sha256(job / "source" / name) != expected:
            raise ValueError(f"Saved source hash differs: {name}")
    for label in LABELS:
        if file_sha256(job / "reference" / f"{label}.zip") != trained["model_hashes"][label]:
            raise ValueError(f"Saved {label} model hash differs")
    if (job / "evaluation_started.json").exists():
        raise FileExistsError(f"Evaluation already started; refusing to overwrite: {job}")
    if not execute:
        print("Evaluation preflight passed. Dry run; add --execute.", flush=True)
        return
    with (job / "evaluation_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"updated_utc": datetime.now(timezone.utc).isoformat()}, handle)
    try:
        status(job, "running", "nominal_evaluation")
        evaluate(model_path=job / "reference/candidate.zip", config_path=job / "candidate/training/config.json",
             results_dir=job / "candidate/evaluation",
                 controller="sac", mode="nominal", episodes=100, base_seed=80000)
        status(job, "running", "comparison")
        summary = compare(job, protocol["cases"])
        status(job, "running", "pole_analysis")
        with (job / "pole_analysis.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(analysis_python), "-u", str(job / "source/current_penalty.py"),
                            "--analyze", "--job-dir", str(job)], stdout=log, stderr=subprocess.STDOUT, check=True)
        gates = review(job, summary)
        atomic_write_json(job / "completed.json", {"state": "completed", "review": gates,
            "report": str(job / "FINDINGS.md"), "updated_utc": datetime.now(timezone.utc).isoformat()})
        status(job, "completed", "report")
    except Exception:
        failure = traceback.format_exc()
        atomic_write_json(job / "failed.json", {"error": failure})
        status(job, "failed", "error", error=failure)
        raise


def run(job=DEFAULT_JOB, execute=False, train_only=False):
    job = Path(job).resolve()
    if job.exists():
        raise FileExistsError(f"Refusing existing output: {job}")
    checks = preflight()
    protocol_cases = cases()
    print(f"Current1.5/delta3, fresh seed0 2M; {len(protocol_cases)} paired cases per model. Output: {job}", flush=True)
    if not execute:
        print("Preflight passed. Dry run; add --execute.", flush=True)
        return
    job.mkdir(parents=True)
    try:
        status(job, "running", "preparation")
        (job / "source").mkdir()
        (job / "reference").mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, job / "source" / path.name)
        shutil.copy2(BASELINE_JOB / "training/best/best_model.zip", job / "reference/baseline.zip")
        atomic_write_json(job / "protocol.json", {"config": asdict(TASK_CONFIG), "training": TRAINING,
            "cases": protocol_cases, "preflight": checks, "analysis_python": str(ANALYSIS_PYTHON.resolve()),
            "source_sha256": {path.name: file_sha256(path) for path in (job / "source").glob("*.py")},
            "selection": "50 nominal training-validation episodes every100k, success then reward; no holdout selection.",
            "screens": {"all_cases_success_5s_and_end_no_unsafe_or_late_loss": True, "slow_decay_min": .9 * 3.8712781982,
                        "slow_pair_damping_min": .7, "fresh_mean_finish_ratio_max": 1.1,
                        "fresh_first5_delta_rms_tv_hf_rms_ratio_max": 1.0, "automatic_promotion": False}})
        status(job, "running", "training")
        train(config=TASK_CONFIG, run_dir=job / "candidate/training", **TRAINING)
        candidate = job / "candidate/training/best/best_model.zip"
        shutil.copy2(candidate, job / "reference/candidate.zip")
        hashes = {label: file_sha256(job / "reference" / f"{label}.zip") for label in LABELS}
        atomic_write_json(job / "model_hashes.json", hashes)
        atomic_write_json(job / "training_completed.json", {"state": "trained", "model_hashes": hashes,
            "updated_utc": datetime.now(timezone.utc).isoformat()})
        status(job, "trained", "awaiting_evaluation")
        if not train_only:
            evaluate_job(job, execute=True)
    except Exception:
        failure = traceback.format_exc()
        atomic_write_json(job / "failed.json", {"error": failure})
        status(job, "failed", "error", error=failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--analyze", action="store_true")
    mode.add_argument("--train-only", action="store_true", help="Train and freeze the selected model; do not evaluate")
    mode.add_argument("--evaluate-only", action="store_true", help="Evaluate a completed training job without retraining")
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    args = parser.parse_args()
    if args.analyze:
        analyze(args.job_dir.resolve())
    elif args.evaluate_only:
        evaluate_job(args.job_dir, args.execute)
    else:
        run(args.job_dir, args.execute, train_only=args.train_only)
