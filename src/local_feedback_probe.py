"""Compare SAC with a tapered local velocity-feedback correction in simulation."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from stable_baselines3 import SAC
import torch

from analyze_damping import actor_function, jacobian, plant_step
from current_penalty import DEFAULT_JOB, TASK_CONFIG, window_current
from evaluate import make_episode, run_prepared_episode, save_detailed_episode, write_csv
from no_centering import score, status
from pole_sensitivity import closed_loop, match_poles
from train import atomic_write_json, file_sha256


DEFAULT_SENSITIVITY = DEFAULT_JOB.parent / "current15_pole_sensitivity_v1/analysis.json"
DEFAULT_OUTPUT = DEFAULT_JOB.parent / "current15_local_feedback_probe_v0"
INNER = np.array([.05, .05, .5, .5])
OUTER = np.array([.15, .15, 1.5, 1.5])
FACTOR = .75
DURATION = 30.0


def correction(state, equilibrium, derivative):
    offsets = np.asarray(state)[..., :4] - np.array([equilibrium, 0, 0, 0])
    blend = np.clip((np.abs(offsets) - INNER) / (OUTER - INNER), 0, 1)
    weight = np.prod(1 - blend * blend * (3 - 2 * blend), axis=-1)
    return weight * (FACTOR - 1) * derivative * offsets[..., 3]


class LocalFeedbackPolicy:
    def __init__(self, model, equilibrium, derivative):
        self.model = model
        self.equilibrium = equilibrium
        self.derivative = derivative

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("This probe supports deterministic evaluation only")
        observation = np.asarray(observation)
        if observation.shape[-1] != 7:
            raise ValueError("Expected the seven-state SAC observation")
        action, hidden = self.model.predict(observation, deterministic=True)
        state = np.stack((np.arctan2(observation[..., 0], observation[..., 1]),
                          np.arctan2(observation[..., 3], observation[..., 4]),
                          observation[..., 2] * TASK_CONFIG.max_abs_omega1,
                          observation[..., 5] * TASK_CONFIG.max_abs_omega2), axis=-1)
        delta = np.asarray(correction(state, self.equilibrium, self.derivative))[..., None]
        return np.clip(np.asarray(action) + delta, -1, 1), hidden


def cases(equilibrium):
    planned = [{"name": "equilibrium", "state": [equilibrium, np.pi, 0, 0]}]
    for axis, name in enumerate(("arm_angle", "pendulum_angle", "arm_velocity", "pendulum_velocity")):
        amplitudes = (.02, .10, .16) if axis < 2 else (.2, 1.0, 1.6)
        for amplitude in amplitudes:
            for sign in (-1, 1):
                state = np.array([equilibrium, np.pi, 0, 0])
                state[axis] += sign * amplitude
                planned.append({"name": f"{name}_{sign * amplitude:+g}", "state": state.tolist()})
    return planned


def prepare(job, sensitivity_path):
    sensitivity = json.loads(Path(sensitivity_path).read_text())
    protocol = json.loads((job / "protocol.json").read_text())
    if protocol["config"] != asdict(TASK_CONFIG):
        raise ValueError("Saved task configuration differs from this probe")
    for name in ("furuta_model.py", "furuta_env.py", "evaluate.py", "analyze_damping.py"):
        if file_sha256(Path(__file__).with_name(name)) != protocol["source_sha256"][name]:
            raise ValueError(f"Source differs from completed experiment: {name}")
    model_path = job / "reference/candidate.zip"
    if file_sha256(model_path) != sensitivity["model_sha256"]:
        raise ValueError("Model hash differs from sensitivity study")
    config = replace(TASK_CONFIG, episode_time=DURATION)
    model = SAC.load(model_path, device="cpu")
    equilibrium = float(np.deg2rad(sensitivity["equilibrium_arm_deg"]))
    gain = np.array(sensitivity["original_gain"])
    action = actor_function(model)
    point = np.array([equilibrium, 0, 0, 0, 0])
    actual_gain = torch.autograd.functional.jacobian(action, torch.tensor(point)).detach().numpy()
    np.testing.assert_allclose(actual_gain, gain, atol=1e-8, rtol=1e-7)
    changed = gain.copy()
    changed[3] *= FACTOR

    def modified_step(state):
        command = float(action(torch.tensor(state)).detach()) + float(correction(state, equilibrium, gain[3]))
        command = float(np.clip(command, -1, 1))
        return np.r_[plant_step(state[:4], command), command]

    matrix = jacobian(modified_step, point, 1e-6)
    expected = closed_loop(np.array(sensitivity["plant_a"]), np.array(sensitivity["plant_b"]), changed)
    np.testing.assert_allclose(matrix, expected, atol=1e-5, rtol=1e-6)
    selected = next(row for row in sensitivity["cases"] if row["label"] == "omega2 x0.75")
    expected_poles = np.array([complex(*value) for value in selected["discrete_poles"]])
    poles = np.linalg.eigvals(matrix).astype(complex)
    np.testing.assert_allclose(match_poles(expected_poles, poles), expected_poles, atol=1e-5, rtol=1e-5)
    checks = {"model_sha256": file_sha256(model_path), "sensitivity_sha256": file_sha256(Path(sensitivity_path)),
              "equilibrium_arm_rad": equilibrium, "original_gain": gain.tolist(), "modified_gain": changed.tolist(),
              "local_jacobian_max_error": float(np.max(abs(matrix - expected))),
              "expected_continuous_poles": selected["continuous_poles"]}
    return model, LocalFeedbackPolicy(model, equilibrium, gain[3]), config, checks


def report(output, episodes, currents):
    summary = {}
    for label in ("original", "modified"):
        selected = [row for row in episodes if row["controller"] == label]
        summary[label] = {"cases": len(selected),
            "unsafe": sum(row["unsafe"] for row in selected),
            "final_balance": sum(row["completed_horizon"] and row["centered_success_end"] for row in selected),
            "lost_after_5s": sum(row["centered_lost_after_5s"] for row in selected)}
    windows = {}
    for window in ("first_5s", "last_5s"):
        paired = set.intersection(*[{row["case"] for row in currents if row["controller"] == label and row["window"] == window}
                                    for label in ("original", "modified")])
        windows[window] = {"paired_cases": len(paired)}
        for label in ("original", "modified"):
            selected = [row for row in currents if row["controller"] == label and row["window"] == window and row["case"] in paired]
            windows[window][label] = {key: float(np.mean([row[key] for row in selected])) if selected else None
                                      for key in ("current_rms_A", "change_rms_A", "total_variation_A", "hf_rms_A")}
    atomic_write_json(output / "summary.json", {"controllers": summary, "current_windows": windows,
        "settling_speed_gate": False, "automatic_promotion": False})
    lines = ["# Nonlinear Local Feedback Probe", "",
             "Original frozen SAC versus SAC plus a smoothly tapered local pendulum-velocity correction.",
             "This is a modified controller, not a retrained policy or a new reward-weight experiment.",
             "No speed nonregression gate or automatic promotion is applied. The horizon is finite (30 s).", "",
             "| Controller | Final balance | Unsafe | Balance lost after 5 s |",
             "|---|---:|---:|---:|"]
    for label, values in summary.items():
        lines.append(f"| {label} | {values['final_balance']}/{values['cases']} | {values['unsafe']} | {values['lost_after_5s']} |")
    lines += ["", "## Paired Mean Current Metrics", "",
              "| Window | Controller | Pairs | RMS (A) | Change RMS (A) | TV (A) | HF RMS (A) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for window, values in windows.items():
        for label in ("original", "modified"):
            text = " | ".join("n/a" if value is None else f"{value:.6g}" for value in values[label].values())
            lines.append(f"| {window} | {label} | {values['paired_cases']} | {text} |")
    lines += ["", "Inspect episodes.csv for finish times, per-case safety, and final equilibrium offsets.",
              "Finish times use the existing balance tolerances, not strict return to the exact expansion point.",
              "The plots/ directory contains paired state/current plots and full traces for every case.",
              "Smaller current metrics do not alone prove increased nonlinear damping; inspect ringdown traces.",
              "Early-terminated windows are excluded from BOTH controllers' paired current means, not from safety counts.",
              "These are local and taper-boundary perturbations, not a swing-up or hardware qualification.",
              "Starting outside the taper reproduces the original action only until the trajectory enters its support."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(job=DEFAULT_JOB, sensitivity=DEFAULT_SENSITIVITY, output=DEFAULT_OUTPUT, execute=False):
    job, sensitivity, output = Path(job).resolve(), Path(sensitivity).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    model, modified, config, checks = prepare(job, sensitivity)
    planned = cases(checks["equilibrium_arm_rad"])
    print(f"Preflight passed: {len(planned)} paired cases, {DURATION:g}s each. Output: {output}", flush=True)
    if not execute:
        print("Dry run only; add --execute to simulate. No training is performed.", flush=True)
        return
    output.mkdir(parents=True)
    try:
        status(output, "running", "preparation")
        source = output / "source"
        source.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, source / path.name)
        shutil.copy2(job / "reference/candidate.zip", output / "candidate.zip")
        shutil.copy2(sensitivity, output / "sensitivity.json")
        atomic_write_json(output / "protocol.json", {"config": asdict(config), "checks": checks, "cases": planned,
            "velocity_derivative_factor": FACTOR, "inner_bounds": INNER.tolist(), "outer_bounds": OUTER.tolist(),
            "gate_coordinates": ["arm_offset_rad", "upright_error_rad", "omega1_rad_s", "omega2_rad_s"],
            "python": sys.executable, "torch_version": torch.__version__, "numpy_version": np.__version__,
            "initial_previous_action": 0, "settling_speed_gate": False,
            "source_sha256": {path.name: file_sha256(path) for path in source.glob("*.py")}})
        episodes, currents = [], []
        for index, case in enumerate(planned, 1):
            for label, policy in (("original", model), ("modified", modified)):
                plant = make_episode(0, config, False, "training")
                plant.reset(np.array(case["state"]))
                result = run_prepared_episode(policy, plant, config, "sac")
                final = result["trace"][-1]
                episodes.append({"controller": label, "case": case["name"], "initial_state": json.dumps(case["state"]),
                    **score(result, config), "completed_horizon": int(len(result["trace"]) == int(round(DURATION / config.sample_time)) + 1),
                    "final_arm_offset_rad": final["theta1_rad"] - checks["equilibrium_arm_rad"],
                    "final_upright_error_rad": final["upright_error_rad"]})
                for name, start, end in (("first_5s", 0, 5), ("last_5s", 25, 30)):
                    metrics = window_current(result, config, start, end)
                    if metrics is not None:
                        currents.append({"controller": label, "case": case["name"], "window": name, **metrics})
                save_detailed_episode(output / "plots" / case["name"] / label, label, index, 0, result, config)
                write_csv(output / "episodes.csv", tuple(episodes[0]), episodes)
                if currents:
                    write_csv(output / "current_windows.csv", tuple(currents[0]), currents)
                status(output, "running", "simulation", case=case["name"], controller=label, completed_cases=index - 1)
                print(f"{index}/{len(planned)} {case['name']} {label}: unsafe={result['unsafe']}", flush=True)
        report(output, episodes, currents)
        atomic_write_json(output / "completed.json", {"state": "completed", "report": str(output / "REPORT.md")})
        status(output, "completed", "report")
    except Exception:
        failure = traceback.format_exc()
        atomic_write_json(output / "failed.json", {"error": failure})
        status(output, "failed", "error", error=failure)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_SENSITIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run(args.job_dir, args.sensitivity, args.output, args.execute)
