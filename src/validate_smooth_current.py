"""Evaluate fixed nominal policies on fresh 30-second matched episodes."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse
import csv
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
from stable_baselines3 import SAC

from evaluate import run_episode, save_detailed_episode, write_csv
from smooth_current import CONFIG, DEFAULT_JOB, METRICS, WEIGHT, window_metrics
from train import atomic_write_json, file_sha256


OUTPUT = DEFAULT_JOB.parent / "nominal_sac_delta5_long_validation_v0"
SEEDS = range(180000, 180100)


def horizon_metrics(result, config, horizon=5.0):
    steps = int(round(horizon / config.sample_time))
    hold = int(round(config.balance_hold_time / config.sample_time))
    trace = result["trace"]
    prefix = trace[1:steps + 1]
    success = bool(len(prefix) == steps and not any(row["unsafe"] for row in prefix)
                   and all(row["balanced"] for row in prefix[-hold:]))
    later = trace[steps + 1:]
    return {
        "success_at_5s": int(success),
        "unsafe_by_5s": int(any(row["unsafe"] for row in prefix)),
        "balance_lost_after_5s": int(success and any(not row["balanced"] or row["unsafe"] for row in later)),
        "completed_30s": int(len(trace) - 1 == int(round(config.episode_time / config.sample_time))),
    }


def current_metrics(result, config, start, end):
    first = int(round(start / config.sample_time))
    last = int(round(end / config.sample_time))
    trace = result["trace"]
    if len(trace) <= last:
        return None
    sliced = {"trace": trace[first:last + 1]}
    metrics = window_metrics(sliced, config, "full_episode")
    if first:
        commands = np.asarray([row["command_current_A"] for row in sliced["trace"][1::config.action_repeat]])
        changes = np.diff(commands, prepend=trace[first]["command_current_A"])
        metrics["change_rms_A"] = float(np.sqrt(np.mean(changes ** 2)))
        metrics["total_variation_A"] = float(np.sum(np.abs(changes)))
    return metrics


def run(job=OUTPUT):
    job = job.resolve()
    if job.exists():
        raise FileExistsError(f"Refusing existing output: {job}")
    paths = {
        "baseline": DEFAULT_JOB / "reference/model.zip",
        "candidate": DEFAULT_JOB / "candidate/training/best/best_model.zip",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with (DEFAULT_JOB / "comparison/current_windows.csv").open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    worst = sorted({int(max(
        (row for row in old_rows if row["model"] == label and row["window"] == window),
        key=lambda row: float(row[metric]),
    )["seed"]) for label in paths for window in ("full_episode", "capture_window") for metric in METRICS})
    job.mkdir(parents=True)

    def status(state, **details):
        atomic_write_json(job / "status.json", {
            "state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), **details,
        })

    try:
        status("running", phase="preparation")
        for label, path in paths.items():
            shutil.copy2(path, job / f"{label}.zip")
        source = job / "source"
        source.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, source / path.name)
        atomic_write_json(job / "protocol.json", {
            "seeds": list(SEEDS), "episode_time_s": 30.0, "worst_diagnostic_seeds": worst,
            "config": asdict(CONFIG), "candidate_action_change_weight": WEIGHT,
            "model_sha256": {label: file_sha256(path) for label, path in paths.items()},
            "selection": "Frozen validation-selected policies; no training or checkpoint selection.",
            "windows": ["first_5s", "last_5s"],
        })
        episodes = []
        currents = []
        for label in paths:
            model = SAC.load(job / f"{label}.zip", device="cpu")
            config = replace(CONFIG, episode_time=30.0, action_change_weight=WEIGHT if label == "candidate" else CONFIG.action_change_weight)
            for index, seed in enumerate(SEEDS, 1):
                result = run_episode(model, seed, False, "training", config, "sac")
                episodes.append({
                    "model": label, "seed": seed, **horizon_metrics(result, config),
                    "success_at_30s": int(result["success"]), "unsafe": int(result["unsafe"]),
                    "finish_time_s": result["finish_time"] if result["success"] else None,
                })
                for window, start, end in (("first_5s", 0, 5), ("last_5s", 25, 30)):
                    metrics = current_metrics(result, config, start, end)
                    if metrics is not None:
                        currents.append({"model": label, "seed": seed, "window": window, **metrics})
                if index == 1 or not result["success"] or episodes[-1]["balance_lost_after_5s"]:
                    save_detailed_episode(job / "fresh" / label, label, index, seed, result, config)
                if index % 10 == 0:
                    status("running", phase="fresh_long_episodes", model=label, episodes=index)
                    print(f"{label}: {index}/{len(SEEDS)} long episodes", flush=True)
            for seed in worst:
                short_config = replace(config, episode_time=5.0)
                result = run_episode(model, seed, False, "training", short_config, "sac")
                save_detailed_episode(job / "worst_diagnostic" / label, label, seed, seed, result, short_config)
        write_csv(job / "episodes.csv", tuple(episodes[0]), episodes)
        write_csv(job / "current_windows.csv", tuple(currents[0]), currents)
        summary = {}
        lines = ["# Fresh Nominal Long-Duration Validation", "", "100 matched seeds 180000-180099, 30 seconds each, frozen SAC policies. No plant randomization, filtering, or delay.", ""]
        for label in paths:
            selected = [row for row in episodes if row["model"] == label]
            counts = {key: sum(row[key] for row in selected) for key in (
                "success_at_5s", "unsafe_by_5s", "success_at_30s", "unsafe", "balance_lost_after_5s", "completed_30s",
            )}
            times = [row["finish_time_s"] for row in selected if row["success_at_30s"]]
            counts["mean_finish_s"] = float(np.mean(times)) if times else None
            counts["max_finish_s"] = max(times) if times else None
            counts["failed_seeds"] = [row["seed"] for row in selected if not row["success_at_5s"] or not row["success_at_30s"] or row["unsafe"] or row["balance_lost_after_5s"]]
            summary[label] = {"episodes": counts, "windows": {}}
            lines.append(f"- {label}: {counts['success_at_5s']}/100 success at 5 s, {counts['success_at_30s']}/100 at 30 s; {counts['unsafe']} unsafe; {counts['balance_lost_after_5s']} lost balance after passing at 5 s. Mean/max final successful finish: {counts['mean_finish_s']} / {counts['max_finish_s']} s. Flagged seeds: {counts['failed_seeds']}")
            for window in ("first_5s", "last_5s"):
                selected_current = [row for row in currents if row["model"] == label and row["window"] == window]
                statistics = {"episodes": len(selected_current)}
                for metric in METRICS:
                    values = [row[metric] for row in selected_current]
                    statistics[metric] = {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)), "max": max(values)} if values else None
                summary[label]["windows"][window] = statistics
        lines.extend(["", "| Window | Metric | Baseline mean / p95 / max | Candidate mean / p95 / max |", "|---|---|---|---|"])
        for window in ("first_5s", "last_5s"):
            for metric in METRICS:
                cells = [" / ".join(f"{value:.6g}" for value in summary[label]["windows"][window][metric].values()) if summary[label]["windows"][window][metric] else "n/a" for label in paths]
                lines.append(f"| {window} | {metric} | {cells[0]} | {cells[1]} |")
        lines.extend(["", f"Paired old diagnostic worst-case plots are in worst_diagnostic/ for seeds {worst}.", "Current statistics require complete windows; check counts and unsafe episodes in summary.json. Late-window changes include the preceding command. HF fraction is relative power at >=25 Hz, not absolute ripple amplitude. Final success alone does not prove uninterrupted balance: the post-5s loss count tests that separately.", "This is a single-training-seed nominal simulation check, not hardware or parameter-robustness qualification. Models remain unchanged; no automatic promotion."])
        atomic_write_json(job / "summary.json", summary)
        (job / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        status("completed", report=str(job / "FINDINGS.md"))
    except Exception:
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        run()
    else:
        print(f"100 fresh matched 30-second episodes per policy; output {OUTPUT}. Add --execute.")
