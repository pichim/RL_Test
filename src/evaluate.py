"""Evaluate direct-current SAC alone and with optional LQR handover."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from controllers import HybridSupervisor, LQRController, Mode
from furuta_env import (
    DEFAULT_CONFIG,
    TASK_VERSION,
    FurutaConfig,
    balanced_state_ready,
    capture_ready,
    current_filter_cutoff_range_hz,
    lqr_handover_current,
    mechanical_capture_ready,
    observation_from_plant,
    sample_action_dead_time_samples,
    sample_current_filter_cutoff_hz,
    sample_matlab_initial_state,
    sample_parameter_scales,
    unsafe,
    upright_error,
    wrap_angle,
)
from furuta_model import FurutaPendulum


DEFAULT_EPISODES = 100
DEFAULT_BASE_SEED = 10_000
FIRST_DETAILED_EPISODES = 5

ROOT = Path(__file__).resolve().parent
SELECTED_DIR = ROOT.parent / "models" / "stage3c_half_rps_v0"
RUN_DIR = SELECTED_DIR
MODEL_PATH = SELECTED_DIR / "model.zip"
RESULTS_DIR = ROOT / "runs" / "evaluation_stage3c_half_rps_v0"

TRACE_FIELDS = (
    "time_s",
    "theta1_rad",
    "theta1_reference_rad",
    "theta1_relative_rad",
    "theta2_rad",
    "upright_error_rad",
    "omega1_rad_s",
    "omega2_rad_s",
    "normalized_action",
    "command_current_A",
    "plant_command_current_A",
    "current_A",
    "current_slew_A_s",
    "candidate_lqr_current_A",
    "mode",
    "mechanical_capture_ready",
    "handover_ready",
    "balanced",
    "unsafe",
)

EPISODE_FIELDS = (
    "controller",
    "mode",
    "episode",
    "seed",
    "initial_capture",
    "success",
    "genuine_swingup",
    "unsafe",
    "capture_time_s",
    "handover_time_s",
    "capture_to_handover_s",
    "first_balance_time_s",
    "final_balance_start_time_s",
    "finish_time_s",
    "supervisor_switches",
    "lqr_fraction",
    "swingup_current_rms_A",
    "swingup_mean_abs_current_A",
    "swingup_saturation_fraction",
    "swingup_applied_current_change_rms_A",
    "swingup_applied_current_slew_rms_A_s",
    "maximum_swingup_applied_current_slew_A_s",
    "swingup_applied_current_total_variation_A",
    "swingup_current_change_rms_A",
    "swingup_current_slew_rms_A_s",
    "maximum_swingup_current_slew_A_s",
    "swingup_current_total_variation_A",
    "swingup_current_second_difference_rms_A",
    "swingup_current_reversals",
    "swingup_high_frequency_power_fraction",
    "handover_current_jump_A",
    "max_abs_omega1_rad_s",
    "max_abs_omega2_rad_s",
    "max_abs_current_A",
    "arm_excursion_rad",
    "omega1_rms_rad_s",
    "omega2_rms_rad_s",
    "post_capture_arm_excursion_rad",
    "post_capture_omega1_rms_rad_s",
    "final_arm_error_rms_rad",
    "final_omega1_rms_rad_s",
    "final_max_abs_omega1_rad_s",
    "minimum_capture_metric",
    "minimum_abs_upright_error_deg",
    "initial_theta1_rad",
    "initial_theta2_rad",
    "initial_omega1_rad_s",
    "initial_omega2_rad_s",
    "motor_torque_scale",
    "arm_mass_scale",
    "arm_inertia_scale",
    "pendulum_mass_scale",
    "pendulum_inertia_scale",
    "arm_damping_scale",
    "pendulum_damping_scale",
    "current_filter_cutoff_hz",
    "action_dead_time_samples",
)

SUMMARY_FIELDS = (
    "controller",
    "mode",
    "episodes",
    "initial_capture_count",
    "success_count",
    "success_percent",
    "genuine_swingup_count",
    "unsafe_count",
    "capture_count",
    "mean_capture_time_s",
    "mean_handover_time_s",
    "mean_capture_to_handover_s",
    "mean_first_balance_time_s",
    "mean_final_balance_start_time_s",
    "mean_successful_finish_time_s",
    "mean_supervisor_switches",
    "mean_lqr_fraction",
    "mean_swingup_current_rms_A",
    "mean_swingup_abs_current_A",
    "mean_swingup_saturation_fraction",
    "mean_swingup_applied_current_change_rms_A",
    "mean_swingup_applied_current_slew_rms_A_s",
    "maximum_swingup_applied_current_slew_A_s",
    "mean_swingup_applied_current_total_variation_A",
    "mean_swingup_current_change_rms_A",
    "p95_swingup_current_change_rms_A",
    "maximum_swingup_current_change_rms_A",
    "mean_swingup_current_slew_rms_A_s",
    "maximum_swingup_current_slew_A_s",
    "mean_swingup_current_total_variation_A",
    "p95_swingup_current_total_variation_A",
    "maximum_swingup_current_total_variation_A",
    "mean_swingup_current_second_difference_rms_A",
    "mean_swingup_current_reversals",
    "mean_swingup_high_frequency_power_fraction",
    "mean_handover_current_jump_A",
    "worst_abs_omega1_rad_s",
    "worst_abs_omega2_rad_s",
    "maximum_current_A",
    "mean_arm_excursion_rad",
    "p95_arm_excursion_rad",
    "maximum_arm_excursion_rad",
    "mean_omega1_rms_rad_s",
    "p95_omega1_rms_rad_s",
    "maximum_omega1_rms_rad_s",
    "mean_omega2_rms_rad_s",
    "p95_omega2_rms_rad_s",
    "maximum_omega2_rms_rad_s",
    "mean_post_capture_arm_excursion_rad",
    "p95_post_capture_arm_excursion_rad",
    "maximum_post_capture_arm_excursion_rad",
    "mean_post_capture_omega1_rms_rad_s",
    "p95_post_capture_omega1_rms_rad_s",
    "maximum_post_capture_omega1_rms_rad_s",
    "mean_final_arm_error_rms_rad",
    "p95_final_arm_error_rms_rad",
    "maximum_final_arm_error_rms_rad",
    "mean_final_omega1_rms_rad_s",
    "p95_final_omega1_rms_rad_s",
    "maximum_final_omega1_rms_rad_s",
    "worst_final_abs_omega1_rad_s",
    "mean_minimum_capture_metric",
    "best_minimum_capture_metric",
    "mean_minimum_abs_upright_error_deg",
    "mean_current_filter_cutoff_hz",
    "minimum_current_filter_cutoff_hz",
    "maximum_current_filter_cutoff_hz",
    "mean_action_dead_time_samples",
    "maximum_action_dead_time_samples",
)


def load_run_config(
    model_path: Path,
    config_path: Path | None,
) -> tuple[FurutaConfig, Path]:
    """Load the exact task configuration saved beside a model."""
    candidates = [config_path] if config_path is not None else [
        model_path.parent / "config.json",
        model_path.parent.parent / "config.json",
        model_path.parent.parent.parent / "config.json",
    ]
    found = next(
        (path for path in candidates if path is not None and path.is_file()),
        None,
    )
    if found is None:
        raise FileNotFoundError(
            "No training config found beside the model. Pass --config explicitly."
        )
    data = json.loads(found.read_text(encoding="utf-8"))
    if data.get("task_version") != TASK_VERSION:
        raise ValueError(
            "The model configuration does not use the current task version: "
            f"expected {TASK_VERSION!r}, got {data.get('task_version')!r}."
        )
    try:
        config = FurutaConfig(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The saved configuration does not match the current task schema."
        ) from error
    return config, found


def balanced(
    state: np.ndarray,
    theta1_reference: float,
    config: FurutaConfig = DEFAULT_CONFIG,
) -> bool:
    """Return the tight final balance condition after LQR handover."""
    return balanced_state_ready(state, theta1_reference, config)


def make_episode(
    seed: int,
    config: FurutaConfig,
    randomized: bool,
    reset_mode: str,
) -> FurutaPendulum:
    """Create one repeatable nominal or robustness-evaluation plant."""
    state_seed, parameter_seed = np.random.SeedSequence(seed).spawn(2)
    state_rng = np.random.default_rng(state_seed)
    parameter_rng = np.random.default_rng(parameter_seed)
    plant = FurutaPendulum(config.sample_time)
    scales = (
        sample_parameter_scales(parameter_rng, config)
        if randomized
        else np.ones(7, dtype=np.float64)
    )
    plant.set_parameter_scales(*scales)
    plant.current_filter_cutoff_hz = (
        sample_current_filter_cutoff_hz(parameter_rng, config)
        if randomized
        else config.current_filter_cutoff_hz
    )
    plant.action_dead_time_samples = (
        sample_action_dead_time_samples(parameter_rng, config)
        if randomized
        else 0
    )
    if reset_mode == "training":
        state = sample_matlab_initial_state(
            state_rng,
            config,
            1.0 if randomized else 0.0,
        )
    elif reset_mode == "downward_rest":
        # A distinct, hardware-relevant start condition near the hanging
        # equilibrium. The arbitrary-angle distribution is covered by
        # ``training`` above.
        state = np.array(
            [
                state_rng.uniform(-np.pi / 4.0, np.pi / 4.0),
                state_rng.uniform(-np.pi / 4.0, np.pi / 4.0),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unknown reset mode: {reset_mode}")
    plant.reset(state)
    plant.x[1] = wrap_angle(plant.x[1])
    return plant


def capture_metric(state: np.ndarray, config: FurutaConfig) -> float:
    """Return 1 at the mechanical capture-box boundary."""
    _, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    return float(
        max(
            abs(upright_error(state)) / config.capture_angle,
            abs(omega1) / config.capture_omega1,
            abs(omega2) / config.capture_omega2,
        )
    )


def root_mean_square(values: np.ndarray) -> float:
    """Return the RMS of a nonempty vector, or zero for an empty vector."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2))) if values.size else 0.0


def trajectory_shape_metrics(
    trace: list[dict],
    capture_time: float | None,
    final_window_steps: int,
) -> dict:
    """Measure arm behavior after capture and over the final balance window."""
    if final_window_steps <= 0:
        raise ValueError("final_window_steps must be positive")
    time = np.asarray([row["time_s"] for row in trace], dtype=np.float64)
    theta1 = np.asarray([row["theta1_rad"] for row in trace], dtype=np.float64)
    arm_error = np.asarray(
        [row["theta1_relative_rad"] for row in trace],
        dtype=np.float64,
    )
    omega1 = np.asarray([row["omega1_rad_s"] for row in trace], dtype=np.float64)

    # Include both endpoints of the requested physical interval.
    final_count = min(final_window_steps + 1, len(trace))
    final_slice = slice(len(trace) - final_count, len(trace))
    final_omega1 = omega1[final_slice]

    if capture_time is None:
        post_capture_arm_excursion = None
        post_capture_omega1_rms = None
    else:
        capture_index = int(np.searchsorted(time, capture_time, side="left"))
        post_capture_theta1 = theta1[capture_index:]
        post_capture_omega1 = omega1[capture_index:]
        post_capture_arm_excursion = (
            float(np.ptp(post_capture_theta1))
            if post_capture_theta1.size
            else 0.0
        )
        post_capture_omega1_rms = root_mean_square(post_capture_omega1)

    return {
        "post_capture_arm_excursion": post_capture_arm_excursion,
        "post_capture_omega1_rms": post_capture_omega1_rms,
        "final_arm_error_rms": root_mean_square(arm_error[final_slice]),
        "final_omega1_rms": root_mean_square(final_omega1),
        "final_max_abs_omega1": (
            float(np.max(np.abs(final_omega1))) if final_omega1.size else 0.0
        ),
    }


def trace_row(
    plant: FurutaPendulum,
    time_s: float,
    normalized_action: float,
    command_current: float,
    plant_command_current: float,
    current: float,
    current_slew: float,
    mode: Mode,
    is_unsafe: bool,
    config: FurutaConfig,
    theta1_reference: float,
) -> dict:
    """Create one diagnostic plant sample."""
    state = plant.x
    return {
        "time_s": float(time_s),
        "theta1_rad": float(state[0]),
        "theta1_reference_rad": float(theta1_reference),
        "theta1_relative_rad": float(state[0] - theta1_reference),
        "theta2_rad": float(state[1]),
        "upright_error_rad": float(upright_error(state)),
        "omega1_rad_s": float(state[2]),
        "omega2_rad_s": float(state[3]),
        "normalized_action": float(normalized_action),
        "command_current_A": float(command_current),
        "plant_command_current_A": float(plant_command_current),
        "current_A": float(current),
        "current_slew_A_s": float(current_slew),
        "candidate_lqr_current_A": lqr_handover_current(
            state,
            config.max_current,
        ),
        "mode": mode.name.lower(),
        "mechanical_capture_ready": int(mechanical_capture_ready(state, config)),
        "handover_ready": int(
            mode is Mode.SWING_UP and capture_ready(state, current, config)
        ),
        "balanced": int(balanced(state, theta1_reference, config)),
        "unsafe": int(is_unsafe),
    }


def run_episode(
    model: SAC,
    seed: int,
    randomized: bool,
    reset_mode: str,
    config: FurutaConfig = DEFAULT_CONFIG,
    controller: str = "hybrid",
) -> dict:
    """Run one SAC-only or hybrid episode and retain its trace."""
    plant = make_episode(seed, config, randomized, reset_mode)
    return run_prepared_episode(model, plant, config, controller)


def run_prepared_episode(
    model: SAC,
    plant: FurutaPendulum,
    config: FurutaConfig = DEFAULT_CONFIG,
    controller: str = "hybrid",
) -> dict:
    """Run one explicitly prepared deterministic plant and retain its trace."""
    if controller not in {"sac", "hybrid"}:
        raise ValueError(f"unknown controller: {controller}")
    use_lqr = controller == "hybrid"
    initial_state = plant.x.copy()
    initial_capture = mechanical_capture_ready(initial_state, config)
    parameter_scales = plant.parameter_scales.copy()
    current_filter_cutoff_hz = float(plant.current_filter_cutoff_hz)
    action_dead_time_samples = int(plant.action_dead_time_samples)

    lqr = LQRController(config.max_current)
    supervisor = HybridSupervisor(config)
    # SAC is trained against absolute theta1 = 0. The hybrid LQR instead uses
    # the arm angle at handover as its local reference.
    theta1_reference = 0.0
    lqr.set_reference(initial_state)

    held_action = 0.0
    action_steps_left = 0
    command_current = 0.0
    plant_command_current = 0.0
    delayed_current_commands = [0.0] * action_dead_time_samples
    applied_current = 0.0
    balance_steps = 0
    capture_time = 0.0 if initial_capture else None
    handover_time = None
    first_balance_time = None
    handover_current_jump = None
    switches = 0
    previous_mode = supervisor.mode
    max_omega1 = abs(float(plant.x[2]))
    max_omega2 = abs(float(plant.x[3]))
    max_current = 0.0
    min_capture_metric = capture_metric(plant.x, config)
    min_abs_upright_error = abs(upright_error(plant.x))
    is_unsafe = False
    finish_time = None
    lqr_steps = 0
    swingup_currents: list[float] = []
    swingup_applied_changes: list[float] = []
    swingup_applied_slews: list[float] = []
    swingup_current_changes: list[float] = []
    swingup_slews: list[float] = []
    command_currents: list[float] = []
    theta1_samples: list[float] = [float(plant.x[0])]
    omega1_samples: list[float] = [float(plant.x[2])]
    omega2_samples: list[float] = [float(plant.x[3])]
    policy_period = config.sample_time * config.action_repeat

    trace = [
        trace_row(
            plant,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            supervisor.mode,
            False,
            config,
            theta1_reference,
        )
    ]
    total_steps = int(round(config.episode_time / config.sample_time))
    required_balance_steps = int(
        round(config.balance_hold_time / config.sample_time)
    )

    for step in range(total_steps):
        mode = (
            supervisor.update(plant.x, applied_current)
            if use_lqr
            else Mode.SWING_UP
        )
        if mode is not previous_mode:
            switches += 1
            previous_mode = mode
            if mode is Mode.LQR:
                handover_time = step * config.sample_time
                theta1_reference = float(plant.x[0])
                lqr.set_reference(plant.x)
                handover_current_jump = abs(lqr.current(plant.x) - applied_current)
            else:
                held_action = float(
                    np.clip(applied_current / config.max_current, -1.0, 1.0)
                )
                command_current = applied_current
                plant_command_current = applied_current
                delayed_current_commands = [
                    applied_current
                ] * action_dead_time_samples
                action_steps_left = 0

        if mode is Mode.LQR:
            normalized_action = np.nan
            current_slew = np.nan
            lqr_steps += 1
        else:
            current_slew = 0.0
            if action_steps_left <= 0:
                observation = observation_from_plant(
                    plant,
                    config,
                    previous_action=held_action,
                )
                action, _ = model.predict(observation, deterministic=True)
                normalized_action = float(
                    np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)
                )
                next_current = normalized_action * config.max_current
                current_change = next_current - command_current
                current_slew = current_change / policy_period
                swingup_current_changes.append(current_change)
                swingup_slews.append(current_slew)
                command_currents.append(next_current)
                held_action = normalized_action
                command_current = next_current
                if action_dead_time_samples > 0:
                    delayed_current_commands.append(command_current)
                    plant_command_current = delayed_current_commands.pop(0)
                else:
                    plant_command_current = command_current
                action_steps_left = config.action_repeat
            normalized_action = held_action
            action_steps_left -= 1

        last_finite_state = plant.x.copy()
        last_finite_current = applied_current
        try:
            if mode is Mode.LQR:
                # The hardware current controller runs at 10 kHz, much faster
                # than SAC's 200 Hz policy. RK4-stage feedback approximates
                # this effectively continuous LQR path on the 1 kHz plant grid.
                # The 100 Hz uncertainty model belongs only to the RL path.
                plant.step_feedback(lqr.current)
                applied_current = lqr.current(plant.x)
                command_current = lqr.current(plant.x)
                plant_command_current = command_current
            else:
                _, applied_current = plant.step_filtered_current(
                    plant_command_current,
                    applied_current,
                    current_filter_cutoff_hz,
                )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            plant.x = last_finite_state
            applied_current = last_finite_current
            is_unsafe = True

        applied_current_slew = (
            (applied_current - last_finite_current) / config.sample_time
        )
        if mode is Mode.SWING_UP:
            swingup_currents.append(applied_current)
            swingup_applied_changes.append(
                applied_current - last_finite_current
            )
            swingup_applied_slews.append(applied_current_slew)

        if np.isfinite(plant.x[1]):
            plant.x[1] = wrap_angle(plant.x[1])
        current_time = (step + 1) * config.sample_time
        theta1_samples.append(float(plant.x[0]))
        omega1_samples.append(float(plant.x[2]))
        omega2_samples.append(float(plant.x[3]))
        max_omega1 = max(max_omega1, abs(float(plant.x[2])))
        max_omega2 = max(max_omega2, abs(float(plant.x[3])))
        max_current = max(max_current, abs(float(applied_current)))
        min_capture_metric = min(min_capture_metric, capture_metric(plant.x, config))
        min_abs_upright_error = min(
            min_abs_upright_error,
            abs(upright_error(plant.x)),
        )
        is_unsafe = bool(is_unsafe or unsafe(plant.x, config))
        if capture_time is None and mechanical_capture_ready(plant.x, config):
            capture_time = current_time
        is_balanced = bool(
            not is_unsafe and balanced(plant.x, theta1_reference, config)
        )
        trace.append(
            trace_row(
                plant,
                current_time,
                normalized_action,
                command_current,
                plant_command_current,
                applied_current,
                applied_current_slew,
                mode,
                is_unsafe,
                config,
                theta1_reference,
            )
        )

        if is_unsafe:
            finish_time = current_time
            break
        if is_balanced:
            balance_steps += 1
            if first_balance_time is None:
                first_balance_time = current_time
        else:
            balance_steps = 0

    # A successful controller is still balanced at the end, rather than one
    # that merely passed through upright at some earlier instant.
    success = bool(not is_unsafe and balance_steps >= required_balance_steps)
    final_balance_start_time = None
    if success:
        final_time = float(trace[-1]["time_s"])
        final_balance_start_time = max(
            0.0,
            final_time - balance_steps * config.sample_time,
        )
        finish_time = final_balance_start_time + config.balance_hold_time

    executed_steps = max(len(trace) - 1, 1)
    currents = np.asarray(swingup_currents, dtype=np.float64)
    applied_changes = np.asarray(swingup_applied_changes, dtype=np.float64)
    applied_slews = np.asarray(swingup_applied_slews, dtype=np.float64)
    changes = np.asarray(swingup_current_changes, dtype=np.float64)
    slews = np.asarray(swingup_slews, dtype=np.float64)
    commands = np.asarray(command_currents, dtype=np.float64)
    second_differences = np.diff(commands, n=2)
    nonzero_signs = np.sign(commands[np.abs(commands) > 1e-9])
    current_reversals = int(np.sum(np.diff(nonzero_signs) != 0))
    if commands.size >= 4 and np.any(commands != np.mean(commands)):
        centered = commands - np.mean(commands)
        frequencies = np.fft.rfftfreq(commands.size, d=policy_period)
        power = np.abs(np.fft.rfft(centered)) ** 2
        total_power = float(np.sum(power[1:]))
        high_frequency_fraction = (
            float(np.sum(power[frequencies >= 25.0]) / total_power)
            if total_power > 0.0
            else 0.0
        )
    else:
        high_frequency_fraction = 0.0
    theta1_array = np.asarray(theta1_samples, dtype=np.float64)
    omega1_array = np.asarray(omega1_samples, dtype=np.float64)
    omega2_array = np.asarray(omega2_samples, dtype=np.float64)
    shape_metrics = trajectory_shape_metrics(
        trace,
        capture_time,
        required_balance_steps,
    )
    return {
        "initial_capture": initial_capture,
        "success": success,
        "genuine_swingup": bool(success and not initial_capture),
        "unsafe": is_unsafe,
        "capture_time": capture_time,
        "handover_time": handover_time,
        "capture_to_handover": (
            None
            if capture_time is None or handover_time is None
            else max(0.0, handover_time - capture_time)
        ),
        "first_balance_time": first_balance_time,
        "final_balance_start_time": final_balance_start_time,
        "finish_time": finish_time,
        "switches": switches,
        "lqr_fraction": lqr_steps / executed_steps,
        "swingup_current_rms": root_mean_square(currents),
        "swingup_mean_abs_current": (
            float(np.mean(np.abs(currents))) if currents.size else 0.0
        ),
        "swingup_saturation_fraction": (
            float(np.mean(np.abs(currents) >= 0.95 * config.max_current))
            if currents.size
            else 0.0
        ),
        "swingup_applied_current_change_rms": root_mean_square(
            applied_changes
        ),
        "swingup_applied_current_slew_rms": root_mean_square(applied_slews),
        "maximum_swingup_applied_current_slew": (
            float(np.max(np.abs(applied_slews)))
            if applied_slews.size
            else 0.0
        ),
        "swingup_applied_current_total_variation": (
            float(np.sum(np.abs(applied_changes)))
            if applied_changes.size
            else 0.0
        ),
        "swingup_current_change_rms": root_mean_square(changes),
        "swingup_current_slew_rms": root_mean_square(slews),
        "maximum_swingup_current_slew": (
            float(np.max(np.abs(slews))) if slews.size else 0.0
        ),
        "swingup_current_total_variation": (
            float(np.sum(np.abs(changes))) if changes.size else 0.0
        ),
        "swingup_current_second_difference_rms": root_mean_square(
            second_differences
        ),
        "swingup_current_reversals": current_reversals,
        "swingup_high_frequency_power_fraction": high_frequency_fraction,
        "handover_current_jump": handover_current_jump,
        "max_omega1": max_omega1,
        "max_omega2": max_omega2,
        "max_current": max_current,
        "arm_excursion": float(np.ptp(theta1_array)),
        "omega1_rms": root_mean_square(omega1_array),
        "omega2_rms": root_mean_square(omega2_array),
        **shape_metrics,
        "min_capture_metric": min_capture_metric,
        "min_abs_upright_error": min_abs_upright_error,
        "initial_state": initial_state,
        "parameter_scales": parameter_scales,
        "current_filter_cutoff_hz": current_filter_cutoff_hz,
        "action_dead_time_samples": action_dead_time_samples,
        "trace": trace,
    }


def optional_float(value: float | None) -> str | float:
    return "" if value is None else float(value)


def episode_row(
    controller: str, name: str, episode: int, seed: int, result: dict
) -> dict:
    initial = result["initial_state"]
    scales = result["parameter_scales"]
    return {
        "controller": controller,
        "mode": name,
        "episode": episode,
        "seed": seed,
        "initial_capture": int(result["initial_capture"]),
        "success": int(result["success"]),
        "genuine_swingup": int(result["genuine_swingup"]),
        "unsafe": int(result["unsafe"]),
        "capture_time_s": optional_float(result["capture_time"]),
        "handover_time_s": optional_float(result["handover_time"]),
        "capture_to_handover_s": optional_float(result["capture_to_handover"]),
        "first_balance_time_s": optional_float(result["first_balance_time"]),
        "final_balance_start_time_s": optional_float(
            result["final_balance_start_time"]
        ),
        "finish_time_s": optional_float(result["finish_time"]),
        "supervisor_switches": result["switches"],
        "lqr_fraction": result["lqr_fraction"],
        "swingup_current_rms_A": result["swingup_current_rms"],
        "swingup_mean_abs_current_A": result["swingup_mean_abs_current"],
        "swingup_saturation_fraction": result["swingup_saturation_fraction"],
        "swingup_applied_current_change_rms_A": result[
            "swingup_applied_current_change_rms"
        ],
        "swingup_applied_current_slew_rms_A_s": result[
            "swingup_applied_current_slew_rms"
        ],
        "maximum_swingup_applied_current_slew_A_s": result[
            "maximum_swingup_applied_current_slew"
        ],
        "swingup_applied_current_total_variation_A": result[
            "swingup_applied_current_total_variation"
        ],
        "swingup_current_change_rms_A": result["swingup_current_change_rms"],
        "swingup_current_slew_rms_A_s": result["swingup_current_slew_rms"],
        "maximum_swingup_current_slew_A_s": result[
            "maximum_swingup_current_slew"
        ],
        "swingup_current_total_variation_A": result[
            "swingup_current_total_variation"
        ],
        "swingup_current_second_difference_rms_A": result[
            "swingup_current_second_difference_rms"
        ],
        "swingup_current_reversals": result["swingup_current_reversals"],
        "swingup_high_frequency_power_fraction": result[
            "swingup_high_frequency_power_fraction"
        ],
        "handover_current_jump_A": optional_float(result["handover_current_jump"]),
        "max_abs_omega1_rad_s": result["max_omega1"],
        "max_abs_omega2_rad_s": result["max_omega2"],
        "max_abs_current_A": result["max_current"],
        "arm_excursion_rad": result["arm_excursion"],
        "omega1_rms_rad_s": result["omega1_rms"],
        "omega2_rms_rad_s": result["omega2_rms"],
        "post_capture_arm_excursion_rad": optional_float(
            result["post_capture_arm_excursion"]
        ),
        "post_capture_omega1_rms_rad_s": optional_float(
            result["post_capture_omega1_rms"]
        ),
        "final_arm_error_rms_rad": result["final_arm_error_rms"],
        "final_omega1_rms_rad_s": result["final_omega1_rms"],
        "final_max_abs_omega1_rad_s": result["final_max_abs_omega1"],
        "minimum_capture_metric": result["min_capture_metric"],
        "minimum_abs_upright_error_deg": np.rad2deg(
            result["min_abs_upright_error"]
        ),
        "initial_theta1_rad": initial[0],
        "initial_theta2_rad": initial[1],
        "initial_omega1_rad_s": initial[2],
        "initial_omega2_rad_s": initial[3],
        "motor_torque_scale": scales[0],
        "arm_mass_scale": scales[1],
        "arm_inertia_scale": scales[2],
        "pendulum_mass_scale": scales[3],
        "pendulum_inertia_scale": scales[4],
        "arm_damping_scale": scales[5],
        "pendulum_damping_scale": scales[6],
        "current_filter_cutoff_hz": result["current_filter_cutoff_hz"],
        "action_dead_time_samples": result["action_dead_time_samples"],
    }


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_available(results: list[dict], key: str) -> float | None:
    values = [result[key] for result in results if result[key] is not None]
    return None if not values else float(np.mean(values))


def percentile_available(
    results: list[dict], key: str, percentile: float
) -> float | None:
    """Return a percentile of the available episode values."""
    values = [result[key] for result in results if result[key] is not None]
    return None if not values else float(np.percentile(values, percentile))


def maximum_available(results: list[dict], key: str) -> float | None:
    """Return the maximum of the available episode values."""
    values = [result[key] for result in results if result[key] is not None]
    return None if not values else float(np.max(values))


def summary_row(controller: str, name: str, results: list[dict]) -> dict:
    number = len(results)
    successful_finish_times = [
        result["finish_time"] for result in results if result["success"]
    ]
    return {
        "controller": controller,
        "mode": name,
        "episodes": number,
        "initial_capture_count": sum(r["initial_capture"] for r in results),
        "success_count": sum(r["success"] for r in results),
        "success_percent": 100.0 * sum(r["success"] for r in results) / number,
        "genuine_swingup_count": sum(r["genuine_swingup"] for r in results),
        "unsafe_count": sum(r["unsafe"] for r in results),
        "capture_count": sum(r["capture_time"] is not None for r in results),
        "mean_capture_time_s": optional_float(mean_available(results, "capture_time")),
        "mean_handover_time_s": optional_float(
            mean_available(results, "handover_time")
        ),
        "mean_capture_to_handover_s": optional_float(
            mean_available(results, "capture_to_handover")
        ),
        "mean_first_balance_time_s": optional_float(
            mean_available(results, "first_balance_time")
        ),
        "mean_final_balance_start_time_s": optional_float(
            mean_available(results, "final_balance_start_time")
        ),
        "mean_successful_finish_time_s": (
            float(np.mean(successful_finish_times))
            if successful_finish_times
            else ""
        ),
        "mean_supervisor_switches": float(np.mean([r["switches"] for r in results])),
        "mean_lqr_fraction": float(np.mean([r["lqr_fraction"] for r in results])),
        "mean_swingup_current_rms_A": float(
            np.mean([r["swingup_current_rms"] for r in results])
        ),
        "mean_swingup_abs_current_A": float(
            np.mean([r["swingup_mean_abs_current"] for r in results])
        ),
        "mean_swingup_saturation_fraction": float(
            np.mean([r["swingup_saturation_fraction"] for r in results])
        ),
        "mean_swingup_applied_current_change_rms_A": float(
            np.mean([r["swingup_applied_current_change_rms"] for r in results])
        ),
        "mean_swingup_applied_current_slew_rms_A_s": float(
            np.mean([r["swingup_applied_current_slew_rms"] for r in results])
        ),
        "maximum_swingup_applied_current_slew_A_s": float(
            np.max([r["maximum_swingup_applied_current_slew"] for r in results])
        ),
        "mean_swingup_applied_current_total_variation_A": float(
            np.mean(
                [r["swingup_applied_current_total_variation"] for r in results]
            )
        ),
        "mean_swingup_current_change_rms_A": float(
            np.mean([r["swingup_current_change_rms"] for r in results])
        ),
        "p95_swingup_current_change_rms_A": percentile_available(
            results, "swingup_current_change_rms", 95.0
        ),
        "maximum_swingup_current_change_rms_A": maximum_available(
            results, "swingup_current_change_rms"
        ),
        "mean_swingup_current_slew_rms_A_s": float(
            np.mean([r["swingup_current_slew_rms"] for r in results])
        ),
        "maximum_swingup_current_slew_A_s": float(
            np.max([r["maximum_swingup_current_slew"] for r in results])
        ),
        "mean_swingup_current_total_variation_A": float(
            np.mean([r["swingup_current_total_variation"] for r in results])
        ),
        "p95_swingup_current_total_variation_A": percentile_available(
            results, "swingup_current_total_variation", 95.0
        ),
        "maximum_swingup_current_total_variation_A": maximum_available(
            results, "swingup_current_total_variation"
        ),
        "mean_swingup_current_second_difference_rms_A": float(
            np.mean(
                [r["swingup_current_second_difference_rms"] for r in results]
            )
        ),
        "mean_swingup_current_reversals": float(
            np.mean([r["swingup_current_reversals"] for r in results])
        ),
        "mean_swingup_high_frequency_power_fraction": float(
            np.mean([r["swingup_high_frequency_power_fraction"] for r in results])
        ),
        "mean_handover_current_jump_A": optional_float(
            mean_available(results, "handover_current_jump")
        ),
        "worst_abs_omega1_rad_s": float(np.max([r["max_omega1"] for r in results])),
        "worst_abs_omega2_rad_s": float(np.max([r["max_omega2"] for r in results])),
        "maximum_current_A": float(np.max([r["max_current"] for r in results])),
        "mean_arm_excursion_rad": float(
            np.mean([r["arm_excursion"] for r in results])
        ),
        "p95_arm_excursion_rad": percentile_available(
            results, "arm_excursion", 95.0
        ),
        "maximum_arm_excursion_rad": maximum_available(
            results, "arm_excursion"
        ),
        "mean_omega1_rms_rad_s": float(
            np.mean([r["omega1_rms"] for r in results])
        ),
        "p95_omega1_rms_rad_s": percentile_available(
            results, "omega1_rms", 95.0
        ),
        "maximum_omega1_rms_rad_s": maximum_available(results, "omega1_rms"),
        "mean_omega2_rms_rad_s": float(
            np.mean([r["omega2_rms"] for r in results])
        ),
        "p95_omega2_rms_rad_s": percentile_available(
            results, "omega2_rms", 95.0
        ),
        "maximum_omega2_rms_rad_s": maximum_available(results, "omega2_rms"),
        "mean_post_capture_arm_excursion_rad": optional_float(
            mean_available(results, "post_capture_arm_excursion")
        ),
        "p95_post_capture_arm_excursion_rad": optional_float(
            percentile_available(results, "post_capture_arm_excursion", 95.0)
        ),
        "maximum_post_capture_arm_excursion_rad": optional_float(
            maximum_available(results, "post_capture_arm_excursion")
        ),
        "mean_post_capture_omega1_rms_rad_s": optional_float(
            mean_available(results, "post_capture_omega1_rms")
        ),
        "p95_post_capture_omega1_rms_rad_s": optional_float(
            percentile_available(results, "post_capture_omega1_rms", 95.0)
        ),
        "maximum_post_capture_omega1_rms_rad_s": optional_float(
            maximum_available(results, "post_capture_omega1_rms")
        ),
        "mean_final_arm_error_rms_rad": float(
            np.mean([r["final_arm_error_rms"] for r in results])
        ),
        "p95_final_arm_error_rms_rad": percentile_available(
            results, "final_arm_error_rms", 95.0
        ),
        "maximum_final_arm_error_rms_rad": maximum_available(
            results, "final_arm_error_rms"
        ),
        "mean_final_omega1_rms_rad_s": float(
            np.mean([r["final_omega1_rms"] for r in results])
        ),
        "p95_final_omega1_rms_rad_s": percentile_available(
            results, "final_omega1_rms", 95.0
        ),
        "maximum_final_omega1_rms_rad_s": maximum_available(
            results, "final_omega1_rms"
        ),
        "worst_final_abs_omega1_rad_s": float(
            np.max([r["final_max_abs_omega1"] for r in results])
        ),
        "mean_minimum_capture_metric": float(
            np.mean([r["min_capture_metric"] for r in results])
        ),
        "best_minimum_capture_metric": float(
            np.min([r["min_capture_metric"] for r in results])
        ),
        "mean_minimum_abs_upright_error_deg": float(
            np.rad2deg(np.mean([r["min_abs_upright_error"] for r in results]))
        ),
        "mean_current_filter_cutoff_hz": float(
            np.mean([r["current_filter_cutoff_hz"] for r in results])
        ),
        "minimum_current_filter_cutoff_hz": float(
            np.min([r["current_filter_cutoff_hz"] for r in results])
        ),
        "maximum_current_filter_cutoff_hz": float(
            np.max([r["current_filter_cutoff_hz"] for r in results])
        ),
        "mean_action_dead_time_samples": float(
            np.mean([r["action_dead_time_samples"] for r in results])
        ),
        "maximum_action_dead_time_samples": int(
            np.max([r["action_dead_time_samples"] for r in results])
        ),
    }


def available(value: str | float, unit: str = "") -> str:
    return "n/a" if value == "" else f"{float(value):.3f}{unit}"


def summarize_text(row: dict) -> list[str]:
    name = f"{row['controller']} / {row['mode']}"
    return [
        name,
        "-" * len(name),
        f"Initial capture states: {row['initial_capture_count']}/{row['episodes']}",
        f"Balanced: {row['success_count']}/{row['episodes']} "
        f"({row['success_percent']:.1f}%)",
        f"Genuine swing-ups: {row['genuine_swingup_count']}/{row['episodes']}",
        f"Unsafe: {row['unsafe_count']}/{row['episodes']}",
        f"Entered mechanical capture box: {row['capture_count']}/{row['episodes']}",
        f"Mean capture time: {available(row['mean_capture_time_s'], ' s')}",
        f"Mean handover time: {available(row['mean_handover_time_s'], ' s')}",
        "Mean capture-to-handover delay: "
        f"{available(row['mean_capture_to_handover_s'], ' s')}",
        "Mean start of final uninterrupted balance: "
        f"{available(row['mean_final_balance_start_time_s'], ' s')}",
        "Mean successful finish time: "
        f"{available(row['mean_successful_finish_time_s'], ' s')}",
        f"Mean supervisor switches: {row['mean_supervisor_switches']:.2f}",
        f"Mean LQR fraction: {100.0 * row['mean_lqr_fraction']:.1f}%",
        f"Mean swing-up current RMS: {row['mean_swingup_current_rms_A']:.4f} A",
        f"Mean swing-up |current|: {row['mean_swingup_abs_current_A']:.4f} A",
        "Mean swing-up saturation fraction: "
        f"{100.0 * row['mean_swingup_saturation_fraction']:.1f}%",
        "Mean applied-current change RMS (1 kHz): "
        f"{row['mean_swingup_applied_current_change_rms_A']:.4f} A",
        "Mean applied-current slew RMS (1 kHz): "
        f"{row['mean_swingup_applied_current_slew_rms_A_s']:.3f} A/s",
        "Maximum applied-current slew (1 kHz): "
        f"{row['maximum_swingup_applied_current_slew_A_s']:.3f} A/s",
        "Mean applied-current total variation: "
        f"{row['mean_swingup_applied_current_total_variation_A']:.3f} A",
        "Mean current-command change RMS: "
        f"{row['mean_swingup_current_change_rms_A']:.4f} A",
        "Current-command change RMS p95/max: "
        f"{row['p95_swingup_current_change_rms_A']:.4f} / "
        f"{row['maximum_swingup_current_change_rms_A']:.4f} A",
        "Mean current-command slew RMS: "
        f"{row['mean_swingup_current_slew_rms_A_s']:.3f} A/s",
        "Maximum current-command slew: "
        f"{row['maximum_swingup_current_slew_A_s']:.3f} A/s",
        "Mean current-command total variation: "
        f"{row['mean_swingup_current_total_variation_A']:.3f} A",
        "Current-command total variation p95/max: "
        f"{row['p95_swingup_current_total_variation_A']:.3f} / "
        f"{row['maximum_swingup_current_total_variation_A']:.3f} A",
        "Mean current-command second-difference RMS: "
        f"{row['mean_swingup_current_second_difference_rms_A']:.4f} A",
        "Mean current-command reversals: "
        f"{row['mean_swingup_current_reversals']:.2f}",
        "Mean high-frequency command power: "
        f"{100.0 * row['mean_swingup_high_frequency_power_fraction']:.1f}%",
        "Mean handover current jump: "
        f"{available(row['mean_handover_current_jump_A'], ' A')}",
        f"Worst |omega1|: {row['worst_abs_omega1_rad_s']:.3f} rad/s",
        f"Worst |omega2|: {row['worst_abs_omega2_rad_s']:.3f} rad/s",
        f"Maximum current: {row['maximum_current_A']:.3f} A",
        f"Mean arm excursion: {row['mean_arm_excursion_rad']:.3f} rad",
        "Arm excursion p95/max: "
        f"{row['p95_arm_excursion_rad']:.3f} / "
        f"{row['maximum_arm_excursion_rad']:.3f} rad",
        f"Mean omega1 RMS: {row['mean_omega1_rms_rad_s']:.3f} rad/s",
        "Omega1 RMS p95/max: "
        f"{row['p95_omega1_rms_rad_s']:.3f} / "
        f"{row['maximum_omega1_rms_rad_s']:.3f} rad/s",
        f"Mean omega2 RMS: {row['mean_omega2_rms_rad_s']:.3f} rad/s",
        "Omega2 RMS p95/max: "
        f"{row['p95_omega2_rms_rad_s']:.3f} / "
        f"{row['maximum_omega2_rms_rad_s']:.3f} rad/s",
        "Mean post-capture arm excursion: "
        f"{available(row['mean_post_capture_arm_excursion_rad'], ' rad')}",
        "Post-capture arm excursion p95/max: "
        f"{available(row['p95_post_capture_arm_excursion_rad'], ' rad')} / "
        f"{available(row['maximum_post_capture_arm_excursion_rad'], ' rad')}",
        "Mean post-capture omega1 RMS: "
        f"{available(row['mean_post_capture_omega1_rms_rad_s'], ' rad/s')}",
        "Mean final-window arm-error RMS: "
        f"{row['mean_final_arm_error_rms_rad']:.4f} rad",
        "Final-window arm-error RMS p95/max: "
        f"{row['p95_final_arm_error_rms_rad']:.4f} / "
        f"{row['maximum_final_arm_error_rms_rad']:.4f} rad",
        "Mean final-window omega1 RMS: "
        f"{row['mean_final_omega1_rms_rad_s']:.4f} rad/s",
        "Final-window omega1 RMS p95/max: "
        f"{row['p95_final_omega1_rms_rad_s']:.4f} / "
        f"{row['maximum_final_omega1_rms_rad_s']:.4f} rad/s",
        "Worst final-window |omega1|: "
        f"{row['worst_final_abs_omega1_rad_s']:.3f} rad/s",
        f"Mean closest capture-box metric: {row['mean_minimum_capture_metric']:.3f}",
        f"Best closest capture-box metric: {row['best_minimum_capture_metric']:.3f}",
        "Mean minimum |upright error|: "
        f"{row['mean_minimum_abs_upright_error_deg']:.2f} deg",
        "Current-filter cutoff mean/min/max: "
        f"{row['mean_current_filter_cutoff_hz']:.2f} / "
        f"{row['minimum_current_filter_cutoff_hz']:.2f} / "
        f"{row['maximum_current_filter_cutoff_hz']:.2f} Hz",
        "Action dead time mean/max: "
        f"{row['mean_action_dead_time_samples']:.2f} / "
        f"{row['maximum_action_dead_time_samples']} policy samples",
    ]


def selected_episode_indices(results: list[dict]) -> list[int]:
    selected = list(range(min(FIRST_DETAILED_EPISODES, len(results))))
    for candidate in (
        next((i for i, result in enumerate(results) if not result["success"]), None),
        next((i for i, result in enumerate(results) if result["unsafe"]), None),
        max(
            range(len(results)),
            key=lambda i: results[i]["swingup_current_total_variation"],
            default=None,
        ),
        max(
            range(len(results)),
            key=lambda i: results[i]["arm_excursion"],
            default=None,
        ),
    ):
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    return selected


def shade_lqr_regions(axes, time: np.ndarray, lqr_mode: np.ndarray) -> None:
    if len(time) < 2:
        return
    edges = np.diff(np.concatenate(([0], lqr_mode.astype(np.int8), [0])))
    sample_time = float(np.median(np.diff(time)))
    for start, stop in zip(
        np.flatnonzero(edges == 1),
        np.flatnonzero(edges == -1),
        strict=True,
    ):
        right = time[min(stop - 1, len(time) - 1)] + sample_time
        for axis in axes:
            axis.axvspan(time[start], right, color="0.90", zorder=0)


def plot_episode(
    path: Path,
    name: str,
    episode: int,
    seed: int,
    result: dict,
    config: FurutaConfig,
) -> None:
    trace = result["trace"]
    time = np.array([row["time_s"] for row in trace])
    theta1 = np.rad2deg([row["theta1_rad"] for row in trace])
    upright = np.rad2deg([row["upright_error_rad"] for row in trace])
    omega1 = np.array([row["omega1_rad_s"] for row in trace])
    omega2 = np.array([row["omega2_rad_s"] for row in trace])
    command_current = np.array([row["command_current_A"] for row in trace])
    plant_command_current = np.array(
        [row["plant_command_current_A"] for row in trace]
    )
    current = np.array([row["current_A"] for row in trace])
    lqr_mode = np.array([row["mode"] == "lqr" for row in trace])
    capture = np.array([row["mechanical_capture_ready"] for row in trace])
    balance = np.array([row["balanced"] for row in trace])

    figure, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    shade_lqr_regions(axes, time, lqr_mode)
    axes[0].plot(time, theta1, label=r"absolute arm $\theta_1$")
    axes[0].plot(time, upright, label="upright error")
    axes[0].axhline(np.rad2deg(config.arm_angle_limit), color="0.5", ls="--")
    axes[0].axhline(-np.rad2deg(config.arm_angle_limit), color="0.5", ls="--")
    axes[0].set_ylabel("angle [deg]")
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].plot(time, omega1, label=r"$\omega_1$")
    axes[1].plot(time, omega2, label=r"$\omega_2$")
    axes[1].set_ylabel("velocity [rad/s]")
    axes[1].legend(loc="upper right", ncol=2)
    axes[2].step(
        time,
        command_current,
        where="post",
        color="0.55",
        alpha=0.8,
        label="requested current",
    )
    axes[2].step(
        time,
        plant_command_current,
        where="post",
        color="C0",
        alpha=0.75,
        label="delayed plant command",
    )
    axes[2].plot(time, current, color="C3", label="applied current")
    axes[2].axhline(config.max_current, color="0.5", ls="--")
    axes[2].axhline(-config.max_current, color="0.5", ls="--")
    axes[2].set_ylabel("current [A]")
    axes[2].legend(loc="upper right", ncol=3)
    axes[3].step(time, lqr_mode.astype(float), where="post", label="LQR active")
    axes[3].step(time, 0.65 * capture, where="post", label="capture box")
    axes[3].step(time, 0.30 * balance, where="post", label="balanced")
    axes[3].set_yticks([0.0, 0.3, 0.65, 1.0])
    axes[3].set_yticklabels(["SAC", "balanced", "capture", "LQR"])
    axes[3].set_ylim(-0.08, 1.08)
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="upper right", ncol=3)
    for axis in axes:
        axis.grid(True, alpha=0.3)
        if result["capture_time"] is not None:
            axis.axvline(result["capture_time"], color="C4", ls=":")
        if result["first_balance_time"] is not None:
            axis.axvline(result["first_balance_time"], color="C2", ls=":")
    scales = result["parameter_scales"]
    figure.suptitle(
        f"{name} episode {episode}, seed={seed}: balanced={result['success']}, "
        f"unsafe={result['unsafe']}\nparameter scales = "
        f"[{scales[0]:.3f}, {scales[1]:.3f}, {scales[2]:.3f}, "
        f"{scales[3]:.3f}, {scales[4]:.3f}, {scales[5]:.3f}, "
        f"{scales[6]:.3f}], filter = "
        f"{result['current_filter_cutoff_hz']:.1f} Hz"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_detailed_episode(
    directory: Path,
    name: str,
    episode: int,
    seed: int,
    result: dict,
    config: FurutaConfig,
) -> None:
    stem = f"episode_{episode:03d}"
    write_csv(directory / f"{stem}.csv", TRACE_FIELDS, result["trace"])
    plot_episode(
        directory / f"{stem}.png",
        name,
        episode,
        seed,
        result,
        config,
    )


def evaluation_modes(mode: str) -> tuple[tuple[str, bool, str], ...]:
    """Return the requested evaluation scenarios."""
    all_modes = (
        ("training_nominal", False, "training"),
        ("training_randomized", True, "training"),
        ("downward_rest_randomized", True, "downward_rest"),
    )
    if mode == "all":
        return all_modes
    if mode == "nominal":
        return all_modes[:1]
    raise ValueError(f"unknown evaluation mode: {mode}")


def main(
    model_path: Path = MODEL_PATH,
    results_dir: Path = RESULTS_DIR,
    config_path: Path | None = None,
    controller: str = "both",
    overwrite: bool = False,
    episodes: int = DEFAULT_EPISODES,
    base_seed: int = DEFAULT_BASE_SEED,
    randomized_arm_velocity_half_range_rps: float | None = None,
    mode: str = "all",
) -> None:
    if controller not in {"sac", "hybrid", "both"}:
        raise ValueError(f"unknown controller: {controller}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if base_seed < 0:
        raise ValueError("base seed must be nonnegative")
    modes = evaluation_modes(mode)
    if not model_path.is_file():
        raise FileNotFoundError(f"No trained SAC model found: {model_path}")
    config, resolved_config_path = load_run_config(model_path, config_path)
    if randomized_arm_velocity_half_range_rps is not None:
        arm_velocity_bound = (
            float(randomized_arm_velocity_half_range_rps) * 2.0 * np.pi
        )
        if (
            not np.isfinite(arm_velocity_bound)
            or arm_velocity_bound < config.reset_omega1_half_range
            or arm_velocity_bound >= config.max_abs_omega1
        ):
            raise ValueError(
                "randomized arm-velocity half-range must be finite and convert "
                "to a bound in [reset_omega1_half_range, max_abs_omega1)"
            )
        config = replace(
            config,
            randomized_reset_omega1_half_range=arm_velocity_bound,
        )
    run_dir = resolved_config_path.parent
    results_dir = results_dir.expanduser().resolve()
    protected_directories = {
        Path(results_dir.anchor),
        ROOT.resolve(),
        ROOT.parent.resolve(),
        (ROOT / "runs").resolve(),
        run_dir.resolve(),
    }
    if results_dir in protected_directories or results_dir.is_relative_to(
        run_dir.resolve()
    ):
        raise ValueError(f"Refusing to replace protected directory: {results_dir}")
    results_archive = results_dir.parent / f"{results_dir.name}_results.zip"
    if (results_dir.exists() or results_archive.exists()) and not overwrite:
        raise FileExistsError(
            "Evaluation output already exists. Choose another --results-dir "
            "or pass --overwrite to replace this specific result set."
        )
    if results_dir.exists():
        if not results_dir.is_dir():
            raise FileExistsError(f"Evaluation path is not a directory: {results_dir}")
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model = SAC.load(model_path, device="cpu")
    control_rate = 1.0 / (config.sample_time * config.action_repeat)
    minimum_cutoff, maximum_cutoff = current_filter_cutoff_range_hz(config)
    text_output = [
        f"Model: {model_path}",
        f"Configuration: {resolved_config_path}",
        f"SAC rate: {control_rate:g} Hz",
        (
            "SAC current path: continuous first-order response centered at "
            f"{config.current_filter_cutoff_hz:g} Hz"
            + (
                " with randomized evaluation range "
                f"{minimum_cutoff:g}--{maximum_cutoff:g} Hz"
                if minimum_cutoff != maximum_cutoff
                else ""
            )
            if config.current_filter_cutoff_hz > 0.0
            else "SAC current path: direct-current model"
        ),
        "LQR current path: direct continuous RK4-stage feedback",
        (
            "SAC action dead time: randomized over 0--"
            f"{config.action_dead_time_max_samples} policy samples "
            f"(0--{1000.0 * config.action_dead_time_max_samples / control_rate:g} ms)"
            if config.action_dead_time_max_samples > 0
            else "SAC action dead time: disabled"
        ),
        f"Evaluation controllers: {controller}",
        f"Episodes per mode: {episodes}",
        f"Base seed: {base_seed}",
        f"Evaluation mode: {mode}",
        (
            "Randomized arm reset velocity: +/-"
            f"{config.randomized_reset_omega1_half_range / (2.0 * np.pi):g} "
            "rotations/s"
        ),
    ]
    aggregate_rows = []
    controllers = ("sac", "hybrid") if controller == "both" else (controller,)
    for active_controller in controllers:
        for mode_index, (name, randomized, reset_mode) in enumerate(modes):
            mode_directory = results_dir / active_controller / name
            mode_directory.mkdir(parents=True, exist_ok=True)
            seeds = [
                base_seed + mode_index * episodes + episode
                for episode in range(episodes)
            ]
            results = [
                run_episode(
                    model,
                    seed,
                    randomized,
                    reset_mode,
                    config,
                    active_controller,
                )
                for seed in seeds
            ]
            rows = [
                episode_row(
                    active_controller,
                    name,
                    episode + 1,
                    seed,
                    result,
                )
                for episode, (seed, result) in enumerate(
                    zip(seeds, results, strict=True)
                )
            ]
            write_csv(mode_directory / "episodes.csv", EPISODE_FIELDS, rows)
            for index in selected_episode_indices(results):
                save_detailed_episode(
                    mode_directory,
                    f"{active_controller}/{name}",
                    index + 1,
                    seeds[index],
                    results[index],
                    config,
                )
            aggregate = summary_row(active_controller, name, results)
            aggregate_rows.append(aggregate)
            text_output.extend(["", *summarize_text(aggregate)])

    write_csv(results_dir / "summary.csv", SUMMARY_FIELDS, aggregate_rows)
    summary = "\n".join(text_output) + "\n"
    (results_dir / "summary.txt").write_text(summary, encoding="utf-8")
    (results_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    (results_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "episodes_per_mode": episodes,
                "base_seed": base_seed,
                "mode": mode,
                "seed_ranges": {
                    name: [
                        base_seed + mode_index * episodes,
                        base_seed + (mode_index + 1) * episodes - 1,
                    ]
                    for mode_index, (name, _, _) in enumerate(modes)
                },
                "controller": controller,
                "evaluation_overrides": {
                    "randomized_arm_velocity_half_range_rps": (
                        randomized_arm_velocity_half_range_rps
                    )
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = results_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    shutil.copy2(model_path, artifacts / "model.zip")
    for training_file in run_dir.glob("training*"):
        if training_file.is_file():
            shutil.copy2(training_file, artifacts / training_file.name)
    shutil.make_archive(
        str(results_archive.with_suffix("")),
        "zip",
        root_dir=results_dir,
    )
    print(summary)
    print(f"results saved under: {results_dir}")
    print(f"shareable archive: {results_archive}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"SAC model or checkpoint (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"new output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing result directory/archive",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="training config.json (normally discovered beside the model)",
    )
    parser.add_argument(
        "--controller",
        choices=("sac", "hybrid", "both"),
        default="both",
        help="evaluate SAC alone, the SAC-to-LQR hybrid, or both (default)",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "nominal"),
        default="all",
        help="evaluate all scenarios or only the nominal training reset",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"episodes per evaluation mode (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=(
            "first nominal seed; later modes use consecutive non-overlapping "
            f"ranges (default: {DEFAULT_BASE_SEED})"
        ),
    )
    parser.add_argument(
        "--randomized-arm-velocity-half-range-rps",
        type=float,
        default=None,
        help=(
            "evaluation-only arm reset velocity half-range in rotations/s; "
            "the saved training configuration remains unchanged"
        ),
    )
    arguments = parser.parse_args()
    if arguments.episodes <= 0:
        parser.error("--episodes must be positive")
    if arguments.base_seed < 0:
        parser.error("--base-seed must be nonnegative")
    if (
        arguments.randomized_arm_velocity_half_range_rps is not None
        and not np.isfinite(arguments.randomized_arm_velocity_half_range_rps)
    ):
        parser.error(
            "--randomized-arm-velocity-half-range-rps must be finite"
        )
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(
        arguments.model,
        arguments.results_dir,
        arguments.config,
        arguments.controller,
        arguments.overwrite,
        arguments.episodes,
        arguments.base_seed,
        arguments.randomized_arm_velocity_half_range_rps,
        arguments.mode,
    )
