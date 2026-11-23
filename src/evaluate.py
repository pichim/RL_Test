"""Evaluate one neural policy and save compact Furuta rollout plots."""

import csv
from dataclasses import replace
from datetime import datetime
from itertools import product
import json

from config import EVALUATION, RUNS, Task, configure_runtime

configure_runtime()

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from furuta_env import (
    FurutaSwingUpEnv, arm_position_error, balanced_state_ready, mechanical_capture_ready,
    observation_from_plant, sample_matlab_initial_state, unsafe, upright_error, wrap_angle,
)
from furuta_model import FurutaPendulum
from guided_sac import LQRController, atomic_write_json, teacher_bounds


def load_task(model_path):
    for directory in (model_path.parent, model_path.parent.parent):
        path = directory / "config.json"
        if path.is_file():
            return Task(**json.loads(path.read_text(encoding="utf-8")))
    raise FileNotFoundError(f"No task configuration found beside {model_path}")


def nominal_start(seed, task):
    state_seed, _ = np.random.SeedSequence(seed).spawn(2)
    state = sample_matlab_initial_state(np.random.default_rng(state_seed), task)
    state[1] = wrap_angle(state[1])
    return state


def rollout(model, task, initial_state, previous_action=0.0):
    """Score the final uninterrupted hold across the full fixed-duration episode."""
    plant = FurutaPendulum(task.sample_time)
    plant.reset(initial_state)
    held_action = previous_action
    command = 0.0
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
    for step in range(round(task.episode_time / task.sample_time)):
        if step % task.action_repeat == 0:
            observation = observation_from_plant(plant, task, held_action)
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
    return {
        "balanced": success,
        "unsafe": violation,
        "initial_capture": initial_capture,
        "capture_s": capture,
        "first_balance_s": first_balance,
        "final_balance_start_s": balance_start,
        "finish_s": balance_start + task.balance_hold_time if success else None,
        "current_rms_A": rms(currents),
        "current_change_rms_A": rms(changes),
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
    }


def upright_benchmark(model_path, config):
    """Measure held-out LQR command agreement and nonlinear local recovery."""
    model = SAC.load(model_path, device="cpu")
    teacher = LQRController(config.max_current)
    bounds = teacher_bounds(config)
    upright = np.array([0.0, np.pi, 0.0, 0.0])
    recovered_actor = recovered_teacher = 0
    unsafe_actor = unsafe_teacher = 0
    env = FurutaSwingUpEnv(config)
    try:
        for i, point in enumerate(product((-0.85, 0.0, 0.85), repeat=4)):
            initial = upright + bounds * point
            previous = (-0.75, 0.0, 0.75)[i % 3]
            for use_teacher in (False, True):
                _, info = env.reset(options={"initial_state": initial})
                env.last_normalized_action = previous
                observation = observation_from_plant(env.plant, config, previous)
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
        action, _ = model.predict(np.asarray(observations), deterministic=True)
        command_errors = np.abs(action.ravel() * config.max_current - targets)
    finally:
        env.close()
    return {
        "recovery_starts": 81,
        "actor_recovered": recovered_actor,
        "actor_unsafe": unsafe_actor,
        "lqr_recovered": recovered_teacher,
        "lqr_unsafe": unsafe_teacher,
        "current_probes": len(command_errors),
        "mean_abs_current_error_A": float(np.mean(command_errors)),
        "p95_abs_current_error_A": float(np.quantile(command_errors, 0.95)),
        "max_abs_current_error_A": float(np.max(command_errors)),
    }


def capture_panel(model, task, path):
    results = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            initial = np.array([float(row[key]) for key in (
                "theta1_rad", "theta2_rad", "omega1_rad_s", "omega2_rad_s",
            )])
            remaining = replace(task, episode_time=float(row["time_remaining_s"]))
            previous = float(row["previous_command_A"]) / task.max_current
            outcome, _ = rollout(model, remaining, initial, previous_action=previous)
            results.append({"seed": int(row["seed"]), **outcome})
    if not results:
        raise ValueError(f"Capture panel is empty: {path}")
    return {**summarize(results), "episodes": results}


def evaluate(model_path, directory, settings=EVALUATION, *, capture_starts=None):
    if settings.episodes < 1 or settings.base_seed < 0 or settings.plot_episodes < 0:
        raise ValueError("Invalid evaluation count or seed")
    task = load_task(model_path)
    model = SAC.load(model_path, device="cpu")
    directory.mkdir(parents=True)
    rows = []
    plotted_failure = False
    for index, seed in enumerate(range(settings.base_seed, settings.base_seed + settings.episodes), 1):
        outcome, trace = rollout(model, task, nominal_start(seed, task))
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
    atomic_write_json(directory / "upright.json", upright_benchmark(model_path, task))
    panel = settings.capture_starts if capture_starts is None else capture_starts
    atomic_write_json(directory / "capture.json", capture_panel(model, task, panel))
    return summary


if __name__ == "__main__":
    output = RUNS / datetime.now().strftime("evaluation_%Y%m%d_%H%M%S_%f")
    print(evaluate(EVALUATION.model, output))
    print(output)
