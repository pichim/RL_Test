"""Evaluate one neural policy and save compact Furuta rollout plots."""

import csv
from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
import json
from math import isfinite

from config import EVALUATION, RUNS, EncoderSettings, Task, configure_runtime

configure_runtime()

from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from furuta_env import (
    Encoder, FurutaSwingUpEnv, arm_position_error, balanced_state_ready, mechanical_capture_ready,
    observation_from_plant, policy_observation, sample_matlab_initial_state, unsafe, upright_error, wrap_angle,
)
from furuta_model import FurutaPendulum
from guided_sac import LQRController, atomic_write_json, teacher_bounds


def task_directory(model_path):
    for directory in (model_path.parent, model_path.parent.parent):
        if (directory / "config.json").is_file():
            return directory
    raise FileNotFoundError(f"No task configuration found beside {model_path}")


def load_task(model_path):
    path = task_directory(model_path) / "config.json"
    return Task(**json.loads(path.read_text(encoding="utf-8")))


def load_encoder(model_path):
    path = task_directory(model_path) / "encoder.json"
    return EncoderSettings(**json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None


def nominal_start(seed, task):
    state_seed, _ = np.random.SeedSequence(seed).spawn(2)
    state = sample_matlab_initial_state(np.random.default_rng(state_seed), task)
    state[1] = wrap_angle(state[1])
    return state


def rollout(model, task, initial_state, previous_action=0.0, *, duration=None,
            encoder_settings=None, encoder_seed=0):
    """Score the final uninterrupted hold across the full fixed-duration episode."""
    horizon = task.episode_time if duration is None else duration
    if not isfinite(horizon) or horizon <= 0:
        raise ValueError("Rollout duration must be finite and positive")
    plant = FurutaPendulum(task.sample_time)
    plant.reset(initial_state)
    encoder = None if encoder_settings is None else Encoder(encoder_settings, task)
    if encoder is not None:
        encoder.reset(plant.x, encoder_seed)
    held_action = previous_action
    command = held_action * task.max_current
    capture = 0.0 if mechanical_capture_ready(plant.x, task) else None
    initial_capture = capture is not None
    first_balance = None
    balance_steps = 0
    violation = False
    currents, changes = [], []
    trace = []

    def record(time):
        theta1, _, omega1, omega2 = map(float, plant.x)
        trace.append({
            "time_s": time,
            "arm_angle_rad": theta1,
            "arm_orientation_rad": arm_position_error(theta1, 0.0),
            "upright_error_rad": upright_error(plant.x),
            "arm_speed_rad_s": omega1,
            "pendulum_speed_rad_s": omega2,
            "current_A": command,
        })

    record(0.0)
    for step in range(round(horizon / task.sample_time)):
        if step % task.action_repeat == 0:
            observation = policy_observation(plant, task, held_action, encoder)
            action, _ = model.predict(observation, deterministic=True)
            held_action = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            next_command = held_action * task.max_current
            changes.append(next_command - command)
            command = next_command
        last_finite_state = plant.x.copy()
        try:
            plant.step(command)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            plant.x = last_finite_state
            violation = True
        plant.x[1] = wrap_angle(plant.x[1])
        if encoder is not None and not violation:
            encoder.advance(plant.x, step + 1)
        time = (step + 1) * task.sample_time
        violation = violation or unsafe(plant.x, task)
        if capture is None and mechanical_capture_ready(plant.x, task):
            capture = time
        record(time)
        currents.append(command)
        if violation:
            break
        if balanced_state_ready(plant.x, 0.0, task):
            balance_steps += 1
            if first_balance is None:
                first_balance = time
        else:
            balance_steps = 0

    success = not violation and balance_steps >= round(task.balance_hold_time / task.sample_time)
    balance_start = max(0.0, trace[-1]["time_s"] - balance_steps * task.sample_time) if success else None
    change_rms = rms(changes)
    return {
        "balanced": success,
        "unsafe": violation,
        "initial_capture": initial_capture,
        "capture_s": capture,
        "first_balance_s": first_balance,
        "final_balance_start_s": balance_start,
        "finish_s": balance_start + task.balance_hold_time if success else None,
        "current_rms_A": rms(currents),
        "current_change_rms_A": change_rms,
        "current_slew_rms_A_per_s": change_rms / task.control_period,
    }, trace


def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if values else 0.0


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_episode(path, trace, outcome, current_limit):
    """Three aligned panels for angles, angular velocities, and current."""
    time = [row["time_s"] for row in trace]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    signals = (
        (("arm_orientation_rad", "Arm orientation"), ("upright_error_rad", "Upright error")),
        (("arm_speed_rad_s", "Arm"), ("pendulum_speed_rad_s", "Pendulum")),
        (("current_A", "Current command"),),
    )
    for axis, series, ylabel in zip(axes, signals, ("Angle [rad]", "Speed [rad/s]", "Current [A]")):
        for key, label in series:
            values = np.array([row[key] for row in trace])
            if axis is axes[0]:
                # A wrap is a change of coordinates, not a physical jump.
                values[1:][np.abs(np.diff(values)) > np.pi] = np.nan
            axis.plot(time, values, label=label)
        for event, label, style in (
            ("capture_s", "Capture", ":"), ("final_balance_start_s", "Final balance", "--"),
        ):
            if outcome[event] is not None:
                axis.axvline(outcome[event], color="0.35", linestyle=style,
                             label=label if axis is axes[0] else None)
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right", fontsize="small")
        axis.grid(alpha=0.25)
    for limit in (-current_limit, current_limit):
        axes[-1].axhline(limit, color="0.5", linestyle=":")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(rows):
    finishes = [row["finish_s"] for row in rows if row["balanced"]]
    capture_times = [row["capture_s"] for row in rows if row["capture_s"] is not None]
    return {
        "starts": len(rows),
        "balanced": len(finishes),
        "unsafe": sum(row["unsafe"] for row in rows),
        "captured": len(capture_times),
        "mean_capture_s": float(np.mean(capture_times)) if capture_times else None,
        "mean_finish_s": float(np.mean(finishes)) if finishes else None,
        "mean_current_change_rms_A": float(np.mean([row["current_change_rms_A"] for row in rows])),
        "mean_current_slew_rms_A_per_s": float(np.mean([row["current_slew_rms_A_per_s"] for row in rows])),
    }


def upright_benchmark(model, config, encoder_settings=None):
    """Measure held-out LQR command agreement and nonlinear local recovery."""
    teacher = LQRController(config.max_current)
    bounds = teacher_bounds(config)
    upright = np.array([0.0, np.pi, 0.0, 0.0])
    recovered_actor = recovered_teacher = 0
    unsafe_actor = unsafe_teacher = 0
    env = FurutaSwingUpEnv(config, encoder_settings=encoder_settings)
    try:
        for i, point in enumerate(product((-0.85, 0.0, 0.85), repeat=4)):
            initial = upright + bounds * point
            previous = (-0.75, 0.0, 0.75)[i % 3]
            for use_teacher in (False, True):
                _, info = env.reset(seed=i if encoder_settings is not None else None, options={"initial_state": initial})
                env.last_normalized_action = previous
                observation = env.observation()
                done = False
                while not done:
                    if use_teacher:
                        action = [teacher.current(info["state"]) / config.max_current]
                    else:
                        action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                if use_teacher:
                    recovered_teacher += int(info["is_success"])
                    unsafe_teacher += int(info["unsafe"])
                else:
                    recovered_actor += int(info["is_success"])
                    unsafe_actor += int(info["unsafe"])

        observations = []
        targets = []
        for point in product((-1.0, -0.6, 0.0, 0.6, 1.0), repeat=4):
            env.plant.reset(upright + bounds * point)
            for previous in (-0.75, -0.25, 0.25, 0.75):
                observations.append(observation_from_plant(env.plant, config, previous))
                targets.append(teacher.current(env.plant.x))
        probes = np.asarray(observations)
        observation = probes if encoder_settings is None else {"measured": probes, "clean": probes}
        action, _ = model.predict(observation, deterministic=True)
        command_errors = np.abs(action.ravel() * config.max_current - targets)
    finally:
        env.close()
    return {
        "recovery_starts": 3**len(bounds),
        "actor_recovered": recovered_actor,
        "actor_unsafe": unsafe_actor,
        "lqr_recovered": recovered_teacher,
        "lqr_unsafe": unsafe_teacher,
        "current_probes": len(command_errors),
        "mean_abs_current_error_A": float(np.mean(command_errors)),
        "p95_abs_current_error_A": float(np.quantile(command_errors, 0.95)),
        "max_abs_current_error_A": float(np.max(command_errors)),
    }


def capture_panel(model, task, path, encoder_settings=None):
    results = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            initial = np.array([float(row[key]) for key in (
                "theta1_rad", "theta2_rad", "omega1_rad_s", "omega2_rad_s",
            )])
            previous = float(row["previous_command_A"]) / task.max_current
            outcome, _ = rollout(
                model, task, initial, previous_action=previous, duration=float(row["time_remaining_s"]),
                encoder_settings=encoder_settings, encoder_seed=int(row["seed"]),
            )
            results.append({"seed": int(row["seed"]), **outcome})
    if not results:
        raise ValueError(f"Capture panel is empty: {path}")
    return {**summarize(results), "episodes": results}


def _evaluate_rate(model, model_path, task, directory, settings, capture_starts):
    encoder = load_encoder(model_path)
    if isinstance(model.observation_space, spaces.Dict) != (encoder is not None):
        raise ValueError("Keep encoder.json with the task config for a measured-input checkpoint")
    if encoder is not None:
        encoder.plant_steps(task)
    directory.mkdir(parents=True)
    atomic_write_json(directory / "config.json", asdict(task))
    if encoder is not None:
        atomic_write_json(directory / "encoder.json", asdict(encoder))
    rows = []
    plotted_failure = False
    for index, seed in enumerate(range(settings.base_seed, settings.base_seed + settings.episodes), 1):
        outcome, trace = rollout(model, task, nominal_start(seed, task), encoder_settings=encoder, encoder_seed=seed)
        rows.append({"seed": seed, **outcome})
        if index <= settings.plot_episodes or (not outcome["balanced"] and not plotted_failure):
            stem = directory / f"episode_{index:03d}"
            write_csv(stem.with_suffix(".csv"), trace)
            plot_episode(stem.with_suffix(".png"), trace, outcome, task.max_current)
            plotted_failure |= not outcome["balanced"]

    summary = summarize(rows)
    write_csv(directory / "episodes.csv", rows)
    atomic_write_json(directory / "summary.json", {
        "model": str(model_path.resolve()), "base_seed": settings.base_seed, **summary,
    })
    atomic_write_json(directory / "upright.json", upright_benchmark(model, task, encoder))
    atomic_write_json(directory / "capture.json", capture_panel(model, task, capture_starts, encoder))
    return summary


def _capture_fixture(settings, capture_starts):
    if settings.episodes < 1 or settings.base_seed < 0 or settings.plot_episodes < 0:
        raise ValueError("Invalid evaluation count or seed")
    panel = settings.capture_starts if capture_starts is None else capture_starts
    if not panel.is_file():
        raise FileNotFoundError(panel)
    return panel


def evaluate(model_path, directory, settings=EVALUATION, *, capture_starts=None, action_repeat=None):
    """Evaluate one checkpoint at its saved rate or with a different current hold."""
    panel = _capture_fixture(settings, capture_starts)
    task = load_task(model_path)
    if action_repeat is not None:
        task = replace(task, action_repeat=action_repeat)
    model = SAC.load(model_path, device="cpu")
    return _evaluate_rate(model, model_path, task, directory, settings, panel)


def evaluate_rates(model_path, directory, settings=EVALUATION, *, capture_starts=None):
    """Score the same checkpoint and starting states at both control rates."""
    panel = _capture_fixture(settings, capture_starts)
    task = load_task(model_path)
    tasks = {}
    for rate, name in ((200, "evaluation"), (500, "evaluation_500hz")):
        repeat = task.steps_at_rate(rate)
        tasks[rate] = (directory / name, replace(task, action_repeat=repeat))

    model = SAC.load(model_path, device="cpu")
    for rate, (output, rate_task) in tasks.items():
        print(f"Evaluating at {rate} Hz\n{output}", flush=True)
        summary = _evaluate_rate(model, model_path, rate_task, output, settings, panel)
        print(summary, flush=True)


if __name__ == "__main__":
    output = RUNS / datetime.now().strftime("evaluation_%Y%m%d_%H%M%S_%f")
    evaluate_rates(EVALUATION.model, output)
