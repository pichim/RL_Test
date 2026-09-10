"""Read-only policy and trace diagnostics for nominal SAC damping."""

from copy import deepcopy
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks, periodogram
import torch
from stable_baselines3 import SAC

from evaluate import run_prepared_episode, save_detailed_episode
from furuta_model import FurutaPendulum
from no_centering import DEFAULT_JOB, TASK_CONFIG
from train import atomic_write_json, file_sha256


OUTPUT = DEFAULT_JOB / "damping_analysis"


def jacobian(function, point, step):
    columns = []
    for index in range(len(point)):
        offset = np.zeros_like(point)
        offset[index] = step
        columns.append((function(point + offset) - function(point - offset)) / (2 * step))
    return np.column_stack(columns)


def spectral_metrics(values, period):
    values = np.asarray(values, dtype=float)
    frequencies, power = periodogram(values, fs=1 / period, window="hann", detrend="constant", scaling="density")
    spacing = frequencies[1] - frequencies[0]
    return {
        "ac_rms_A": float(np.std(values)),
        "peak_to_peak_A": float(np.ptp(values)),
        "dominant_frequency_Hz": float(frequencies[1 + np.argmax(power[1:])]),
        "band_rms_A": {name: float(np.sqrt(np.sum(power[(frequencies >= lower) & (frequencies < upper)]) * spacing))
                       for name, lower, upper in (("0.2_to_3Hz", .2, 3), ("3_to_25Hz", 3, 25), ("25_to_100Hz", 25, 100.0001))},
    }


def actor_function(model):
    actor = deepcopy(model.actor).to(dtype=torch.float64)

    def action(state):
        theta, error, omega1, omega2, previous = state.unbind()
        observation = torch.stack((theta.sin(), theta.cos(), (omega1 / 30).clamp(-1, 1),
                                   error.sin(), error.cos(), (omega2 / 30).clamp(-1, 1), previous.clamp(-1, 1)))
        return actor.mu(actor.latent_pi(observation)).tanh().squeeze()

    return action


def plant_step(state, action):
    plant = FurutaPendulum(TASK_CONFIG.sample_time)
    plant.reset(np.array([state[0], np.pi + state[1], state[2], state[3]]))
    for _ in range(TASK_CONFIG.action_repeat):
        plant.step(float(action) * TASK_CONFIG.max_current)
    result = plant.x.copy()
    result[1] -= np.pi
    return result


def modes(matrix):
    period = TASK_CONFIG.sample_time * TASK_CONFIG.action_repeat
    result = []
    for pole in np.linalg.eigvals(matrix).astype(complex):
        continuous = np.log(pole) / period
        result.append({"discrete_real": float(pole.real), "discrete_imag": float(pole.imag),
                       "continuous_real": float(continuous.real), "continuous_imag": float(continuous.imag),
                       "damped_frequency_Hz": float(abs(continuous.imag) / (2 * np.pi)),
                       "damping_ratio": float(-continuous.real / abs(continuous)),
                       "envelope_time_constant_s": float(-1 / continuous.real) if continuous.real < 0 else None})
    return result


def policy_analysis(model):
    action = actor_function(model)

    def evaluate_action(state):
        return float(action(torch.as_tensor(state, dtype=torch.float64)).detach())

    def full_step(state):
        command = evaluate_action(state)
        return np.r_[plant_step(state[:4], command), command]

    def stationary_action(angle):
        return evaluate_action(np.array([angle, 0, 0, 0, 0]))

    angles = np.linspace(-TASK_CONFIG.arm_angle_limit, TASK_CONFIG.arm_angle_limit, 401)
    roots = []
    for lower, upper in zip(angles[:-1], angles[1:]):
        if stationary_action(lower) * stationary_action(upper) < 0:
            root = brentq(stationary_action, lower, upper, xtol=1e-12)
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(root)
    equilibria = []
    for angle in roots:
        state = np.array([angle, 0, 0, 0, 0], dtype=float)
        gain = torch.autograd.functional.jacobian(action, torch.tensor(state, dtype=torch.float64)).detach().numpy()
        plant_a = jacobian(lambda physical: plant_step(physical, 0), state[:4], 1e-5)
        plant_b = jacobian(lambda command: plant_step(state[:4], command[0]), np.array([0.0]), 1e-5)
        matrix = np.zeros((5, 5))
        matrix[:4, :4] = plant_a + plant_b @ gain[None, :4]
        matrix[:4, 4] = plant_b[:, 0] * gain[4]
        matrix[4, :] = gain
        numerical = jacobian(full_step, state, 1e-5)
        discrepancy = float(np.max(np.abs(matrix - numerical)))
        if discrepancy > 1e-5:
            raise AssertionError(f"Analytic/finite-difference Jacobians differ: {discrepancy}")
        observation = np.array([np.sin(angle), np.cos(angle), 0, 0, 1, 0, 0], dtype=np.float32)
        sb3_action = float(model.predict(observation, deterministic=True)[0][0])
        equilibria.append({"arm_angle_rad": angle, "arm_angle_deg": float(np.rad2deg(angle)),
                           "action_gain_order": ["theta1", "upright_error", "omega1", "omega2", "previous_action"],
                           "normalized_action_jacobian": gain.tolist(), "modes": modes(matrix),
                           "finite_difference_check_max_abs_error": discrepancy,
                           "float32_equilibrium_current_residual_A": .5 * sb3_action})
    return {"equilibria": equilibria, "stationary_upright_current_A": {
        str(degrees): .5 * stationary_action(np.deg2rad(degrees)) for degrees in (-90, -45, 0, 30, 45, 90)
    }}


def trace_analysis(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    commands = rows[1::TASK_CONFIG.action_repeat]
    times = np.arange(len(commands)) * .005
    currents = np.array([float(row["command_current_A"]) for row in commands])
    arm = np.array([float(row["theta1_rad"]) for row in commands])
    velocity = np.array([float(row["omega1_rad_s"]) for row in commands])
    result = {"path": str(path.relative_to(DEFAULT_JOB)), "final_arm_deg": float(np.rad2deg(arm[-1])), "windows": {}}
    for start, end in ((1, 3), (3, 5), (5, 10), (25, 30)):
        if float(rows[-1]["time_s"]) < end:
            continue
        selected = (times >= start) & (times < end)
        result["windows"][f"{start}_{end}s"] = {
            **spectral_metrics(currents[selected], .005),
            "arm_peak_to_peak_deg": float(np.rad2deg(np.ptp(arm[selected]))),
            "arm_speed_rms_rad_s": float(np.sqrt(np.mean(velocity[selected] ** 2))),
        }
    peaks, _ = find_peaks(arm, distance=100, prominence=np.deg2rad(.1))
    result["arm_positive_peaks_after_1s"] = [{"time_s": float(times[index]), "angle_deg": float(np.rad2deg(arm[index]))}
                                             for index in peaks if times[index] > 1]
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir()
    result = {"method": "Float64 copy of frozen float32 actor, ReLU local derivatives; sampled 5ms map with previous normalized action as fifth state. Finite-difference cross-check. Equilibrium poles are local only, not global or hardware guarantees.", "models": {}, "traces": []}
    for label in ("weight3_centered", "weight4_centered", "weight4_free"):
        path = DEFAULT_JOB / "reference" / f"{label}.zip"
        model = SAC.load(path, device="cpu")
        result["models"][label] = {"sha256": file_sha256(path), **policy_analysis(model)}
        print(label, json.dumps(result["models"][label], indent=2), flush=True)
        if label == "weight4_free":
            for index, equilibrium in enumerate(result["models"][label]["equilibria"], 1):
                plant = FurutaPendulum(.001)
                plant.reset(np.array([equilibrium["arm_angle_rad"] + .02, np.pi, 0, 0]))
                config = replace(TASK_CONFIG, episode_time=10.0)
                response = run_prepared_episode(model, plant, config, "sac")
                save_detailed_episode(OUTPUT / "small_signal", "free_actor_arm_perturbation_0.02rad", index, 0, response, config)
    paths = list((DEFAULT_JOB / "candidate/evaluation/sac/training_nominal").glob("episode_*.csv"))
    paths.extend((DEFAULT_JOB / "comparison/fresh_long/weight4_free").glob("episode_*.csv"))
    for path in paths:
        result["traces"].append(trace_analysis(path))
    atomic_write_json(OUTPUT / "analysis.json", result)
    shutil_source = Path(__file__).read_text(encoding="utf-8")
    (OUTPUT / "analyze_damping_snapshot.py").write_text(shutil_source, encoding="utf-8")
    print(f"Saved {OUTPUT / 'analysis.json'}")


if __name__ == "__main__":
    main()
