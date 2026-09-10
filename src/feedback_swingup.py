"""Evaluate the frozen local correction on paired full swing-up cases."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import traceback

import numpy as np

from current_penalty import window_current
from evaluate import make_episode, run_prepared_episode, save_detailed_episode, write_csv
from local_feedback_probe import DEFAULT_JOB, DEFAULT_SENSITIVITY, FACTOR, INNER, OUTER, prepare
from no_centering import score, status
from train import atomic_write_json, file_sha256


OUTPUT = DEFAULT_JOB.parent / "current15_feedback_swingup_v0"
FROZEN_PROBE = DEFAULT_JOB.parent / "current15_local_feedback_probe_v0"
METRICS = ("current_rms_A", "change_rms_A", "total_variation_A", "hf_rms_A")


def summarize(episodes, currents):
    result = {}
    for suite in dict.fromkeys(row["suite"] for row in episodes):
        result[suite] = {}
        for label in ("original", "modified"):
            selected = [row for row in episodes if row["suite"] == suite and row["controller"] == label]
            finishes = [row["centered_finish_s"] for row in selected if row["centered_finish_s"] is not None]
            result[suite][label] = {"cases": len(selected),
                **{key: sum(row[key] for row in selected) for key in
                   ("unsafe", "centered_success_5s", "centered_success_end", "centered_lost_after_5s", "completed_horizon")},
                "mean_finish_s": float(np.mean(finishes)) if finishes else None, "windows": {}}
        for window in ("first_5s", "last_5s"):
            paired = set.intersection(*[{row["case"] for row in currents if row["suite"] == suite
                                         and row["controller"] == label and row["window"] == window}
                                        for label in ("original", "modified")])
            for label in ("original", "modified"):
                selected = [row for row in currents if row["suite"] == suite and row["controller"] == label
                            and row["window"] == window and row["case"] in paired]
                result[suite][label]["windows"][window] = {"paired_cases": len(paired),
                    **{key: float(np.mean([row[key] for row in selected])) if selected else None for key in METRICS}}
    return result


def transition_metrics(trace):
    active = np.array([abs(row["correction_action"]) > 1e-8 for row in trace])
    corrections = np.array([row["correction_action"] for row in trace])
    return {"correction_active_steps": int(np.sum(active)),
            "correction_activity_transitions": int(np.sum(active[1:] != active[:-1])),
            "max_correction_action": float(np.max(abs(corrections))),
            "max_correction_step_action": float(np.max(abs(np.diff(corrections, prepend=0))))}


def run(output=OUTPUT, execute=False):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    frozen = json.loads((FROZEN_PROBE / "protocol.json").read_text())
    if file_sha256(Path(__file__).with_name("local_feedback_probe.py")) != frozen["source_sha256"]["local_feedback_probe.py"]:
        raise ValueError("Local correction source differs from the completed probe")
    original, modified, config, checks = prepare(DEFAULT_JOB, DEFAULT_SENSITIVITY)
    planned = json.loads((DEFAULT_JOB / "protocol.json").read_text())["cases"]
    if len(planned) != 284 or len({(case["suite"], case["name"]) for case in planned}) != 284:
        raise ValueError("Expected 284 unique saved cases")
    print(f"Preflight passed: {len(planned)} paired cases, all extended to 30s; correction frozen.", flush=True)
    if not execute:
        print("Dry run; add --execute.", flush=True)
        return
    output.mkdir(parents=True)
    try:
        source = output / "source"
        source.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, source / path.name)
        shutil.copy2(DEFAULT_JOB / "reference/candidate.zip", output / "candidate.zip")
        shutil.copy2(DEFAULT_SENSITIVITY, output / "sensitivity.json")
        atomic_write_json(output / "protocol.json", {"cases": planned, "config": asdict(config), "checks": checks,
            "original_case_horizons_retained_as_metadata_only": True, "evaluation_horizon_s": 30,
            "velocity_derivative_factor": FACTOR, "inner_bounds": INNER.tolist(), "outer_bounds": OUTER.tolist(),
            "selection": "Reused diagnostic cases, not unseen holdout. No controller tuning during evaluation.",
            "settling_speed_gate": False, "automatic_promotion": False,
            "source_sha256": {path.name: file_sha256(path) for path in source.glob("*.py")}})
        episodes, currents = [], []
        for index, case in enumerate(planned, 1):
            for label, policy in (("original", original), ("modified", modified)):
                plant = make_episode(case["seed"], config, False, "training")
                if case["state"] is not None:
                    plant.reset(np.asarray(case["state"], dtype=float))
                initial = plant.x.tolist()
                result = run_prepared_episode(policy, plant, config, "sac")
                measures = score(result, config)
                rows = []
                for previous, current in zip(result["trace"][::config.action_repeat], result["trace"][1::config.action_repeat]):
                    state = np.array([previous["theta1_rad"], previous["upright_error_rad"],
                                      previous["omega1_rad_s"], previous["omega2_rad_s"]])
                    from local_feedback_probe import correction
                    delta = float(correction(state, checks["equilibrium_arm_rad"], checks["original_gain"][3])) if label == "modified" else 0.0
                    rows.append({"time_s": previous["time_s"], "correction_action": delta,
                                 "command_current_A": current["command_current_A"]})
                episodes.append({"suite": case["suite"], "case": case["name"], "controller": label,
                    "seed": case["seed"], "initial_state": json.dumps(initial), **measures,
                    "completed_horizon": int(len(result["trace"]) == 30001), **transition_metrics(rows)})
                for window, start, end in (("first_5s", 0, 5), ("last_5s", 25, 30)):
                    metrics = window_current(result, config, start, end)
                    if metrics is not None:
                        currents.append({"suite": case["suite"], "case": case["name"], "controller": label, "window": window, **metrics})
                destination = output / "traces" / case["suite"] / case["name"] / label
                write_csv(destination / "feedback.csv", tuple(rows[0]), rows)
                if index <= 5 or not measures["centered_success_end"] or measures["unsafe"] or measures["centered_lost_after_5s"]:
                    save_detailed_episode(destination, label, index, case["seed"], result, config)
                else:
                    write_csv(destination / "trajectory.csv", tuple(result["trace"][0]), result["trace"])
            write_csv(output / "episodes.csv", tuple(episodes[0]), episodes)
            write_csv(output / "current_windows.csv", tuple(currents[0]), currents)
            status(output, "running", "comparison", completed_cases=index, total_cases=len(planned))
            print(f"{index}/{len(planned)} {case['suite']} {case['name']}: "
                  f"final balance original/modified={episodes[-2]['centered_success_end']}/{episodes[-1]['centered_success_end']}", flush=True)
        summary = summarize(episodes, currents)
        atomic_write_json(output / "summary.json", summary)
        lines = ["# Frozen Feedback Full Swing-Up Comparison", "",
                 "284 reused paired cases, each evaluated for 30 seconds. No speed gate, retraining, or promotion.",
                 "Original means the current1.5 actor, not the older current1 baseline.", "",
                 "| Suite | Controller | Balance at 5s/end | Unsafe | Late losses | Mean finish (s) |",
                 "|---|---|---:|---:|---:|---:|"]
        for suite, controllers in summary.items():
            for label, values in controllers.items():
                finish = values["mean_finish_s"]
                lines.append(f"| {suite} | {label} | {values['centered_success_5s']}/{values['centered_success_end']} of {values['cases']} | "
                             f"{values['unsafe']} | {values['centered_lost_after_5s']} | {finish if finish is not None else 'n/a'} |")
        lines += ["", "See summary.json for paired first/last5s current means by suite, and episodes.csv for all cases.",
                  "Every trajectory and pre-clipping correction is retained in traces/. Correction activity transitions",
                  "are threshold crossings at 1e-8 normalized action, not a fitted oscillation or damping measure.",
                  "Five-second failures are diagnostic only; final balance, safety and losses remain important.",
                  "Slower motion is accepted, but balance must be reached within the finite 30-second horizon.",
                  "Reused cases cannot serve as fresh evidence for later tuning. No hardware qualification is implied."]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        atomic_write_json(output / "completed.json", {"state": "completed", "report": str(output / "REPORT.md")})
        status(output, "completed", "report")
    except Exception:
        error = traceback.format_exc()
        atomic_write_json(output / "failed.json", {"error": error})
        status(output, "failed", "error", error=error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args.output, args.execute)
