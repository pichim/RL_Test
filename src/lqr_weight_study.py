"""Local reward-weight study; does not train or modify deployed controllers."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import scipy
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import linear_sum_assignment

from analyze_damping import jacobian
from furuta_env import LQR_GAIN
from furuta_model import FurutaPendulum
from nominal import CONFIG


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs" / "lqr_weight_study_v0"
BASE_WEIGHTS = {
    "arm_angle": CONFIG.arm_angle_weight,
    "pendulum_angle": 1.0,
    "arm_velocity": CONFIG.arm_velocity_weight,
    "pendulum_velocity": CONFIG.velocity_weight,
    "current": 1.0,
    "current_change": CONFIG.action_change_weight,
}


def plant_matrices():
    plant = FurutaPendulum(CONFIG.sample_time)
    equilibrium = np.array([0.0, np.pi, 0.0, 0.0])
    continuous = jacobian(lambda state: plant.state_derivative(state, 0.0), equilibrium, 1e-5)
    current_input = jacobian(
        lambda current: plant.state_derivative(equilibrium, float(current[0])),
        np.zeros(1), 1e-5,
    )
    generator = np.zeros((5, 5))
    generator[:4, :4] = continuous
    generator[:4, 4:] = current_input * CONFIG.max_current
    transition = expm(generator * CONFIG.sample_time * CONFIG.action_repeat)
    return continuous, current_input, transition[:4, :4], transition[:4, 4:]


def augmented_problem(transition, action_input, weights):
    """Return dynamics and endpoint-state cost z'Qz + 2z'Nu + u'Ru."""
    state_cost = np.diag([weights[name] for name in (
        "arm_angle", "pendulum_angle", "arm_velocity", "pendulum_velocity",
    )])
    dynamics = np.zeros((5, 5))
    dynamics[:4, :4] = transition
    control = np.vstack((action_input, [[1.0]]))
    endpoint = dynamics[:4]
    quadratic = endpoint.T @ state_cost @ endpoint
    quadratic[4, 4] += weights["current_change"]
    cross = endpoint.T @ state_cost @ action_input
    cross[4, 0] -= weights["current_change"]
    effort = action_input.T @ state_cost @ action_input
    effort[0, 0] += weights["current"] + weights["current_change"]
    return dynamics, control, quadratic, cross, effort


def solve_problem(problem, gamma):
    dynamics, control, quadratic, cross, effort = problem
    value = solve_discrete_are(
        np.sqrt(gamma) * dynamics, np.sqrt(gamma) * control,
        quadratic, effort, s=cross,
    )
    gain = np.linalg.solve(
        effort + gamma * control.T @ value @ control,
        cross.T + gamma * control.T @ value @ dynamics,
    )
    closed = dynamics - control @ gain
    policy_cost = quadratic - cross @ gain - gain.T @ cross.T + gain.T @ effort @ gain
    residual = value - policy_cost - gamma * closed.T @ value @ closed
    relative_residual = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(value)))
    if relative_residual > 1e-8:
        raise RuntimeError(f"Riccati residual too large: {relative_residual}")
    return gain, closed, value, relative_residual


def pole_metrics(matrix, period):
    result = []
    for eigenvalue in sorted(np.linalg.eigvals(matrix).astype(complex), key=lambda value: -abs(value)):
        pole = np.log(eigenvalue) / period
        result.append({
            "discrete_real": float(eigenvalue.real), "discrete_imag": float(eigenvalue.imag),
            "discrete_abs": float(abs(eigenvalue)),
            "real": float(pole.real), "imag": float(pole.imag),
            "omega0": float(abs(pole)),
            "damping": float(-pole.real / abs(pole)) if abs(pole) else None,
        })
    return result


def pole_distance(candidate, target):
    """Match four target poles; report rather than hide the remaining fifth pole."""
    candidate_poles = np.array([complex(mode["real"], mode["imag"]) for mode in candidate])
    target_poles = np.array([complex(mode["real"], mode["imag"]) for mode in target])
    distances = abs(target_poles[:, None] - candidate_poles[None, :]) / abs(target_poles[:, None])
    target_indices, candidate_indices = linear_sum_assignment(distances)
    extra = sorted(set(range(len(candidate))) - set(candidate_indices.tolist()))
    return float(np.mean(distances[target_indices, candidate_indices])), [candidate[index] for index in extra]


def scenarios():
    yield "baseline", BASE_WEIGHTS.copy()
    for name in BASE_WEIGHTS:
        for factor in (0.5, 1.5, 2.0):
            yield f"{name}_x{factor:g}", {**BASE_WEIGHTS, name: BASE_WEIGHTS[name] * factor}


def model_checks(transition, action_input):
    plant = FurutaPendulum(CONFIG.sample_time)

    def held_step(augmented):
        plant.reset(augmented[:4] + np.array([0.0, np.pi, 0.0, 0.0]))
        for _ in range(CONFIG.action_repeat):
            plant.step(float(augmented[4]) * CONFIG.max_current)
        return plant.x - np.array([0.0, np.pi, 0.0, 0.0])

    actual = jacobian(held_step, np.zeros(5), 1e-5)
    error = float(np.max(abs(actual - np.column_stack((transition, action_input)))))
    if error > 1e-6:
        raise RuntimeError(f"ZOH/RK4 crosscheck failed: {error}")
    return {"zoh_vs_rk4_jacobian_max_error": error}


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    continuous, current_input, transition, action_input = plant_matrices()
    period = CONFIG.sample_time * CONFIG.action_repeat
    target = pole_metrics(transition - action_input @ (LQR_GAIN[None, :] / CONFIG.max_current), period)
    records = []
    for gamma in (1.0, 0.99):
        for name, weights in scenarios():
            gain, closed, _, residual = solve_problem(augmented_problem(transition, action_input, weights), gamma)
            modes = pole_metrics(closed, period)
            distance, unmatched = pole_distance(modes, target)
            records.append({
                "scenario": name, "gamma": gamma, "weights": weights,
                "normalized_action_gain": gain.tolist(), "modes": modes,
                "stable": all(mode["discrete_abs"] < 1.0 for mode in modes),
                "target_distance": distance, "unmatched_modes": unmatched,
                "relative_riccati_residual": residual,
            })
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "numpy": np.__version__, "scipy": scipy.__version__},
        "config": asdict(CONFIG), "base_weights": BASE_WEIGHTS, "period": period,
        "state_order": ["arm_angle", "upright_error", "arm_velocity", "pendulum_velocity", "previous_normalized_action"],
        "checks": model_checks(transition, action_input),
        "continuous_A": continuous.tolist(), "current_B": current_input.tolist(),
        "discrete_A": transition.tolist(), "normalized_action_B": action_input.tolist(),
        "existing_lqr_gain_A_per_state": LQR_GAIN.tolist(), "held_lqr_target_modes": target,
        "records": records,
    }
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    payload["source_sha256"] = {}
    for name in ("lqr_weight_study.py", "test_lqr_weight_study.py", "furuta_model.py", "furuta_env.py", "nominal.py", "analyze_damping.py"):
        path = Path(__file__).with_name(name)
        shutil.copy2(path, source / name)
        payload["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "poles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "gamma", "stable", "target_distance", *records[0]["modes"][0]])
        writer.writeheader()
        for record in records:
            for mode in record["modes"]:
                writer.writerow({**{key: record[key] for key in ("scenario", "gamma", "stable", "target_distance")}, **mode})
    report = [
        "# Local LQR Reward-Weight Study", "",
        f"Runtime: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}.", "",
        "Nominal upright linearization, 200 Hz zero-order-held normalized current, fixed arm reference.",
        "State cost is evaluated AFTER the held action, matching matlab_reward on full nonterminal steps.",
        "Previous action is an explicit fifth state. Current/change weights act on normalized current, not amperes.", "",
        "This is an unconstrained infinite-horizon deterministic quadratic surrogate, NOT SAC training or swing-up validation.",
        "Entropy, terminal success bonus, early termination, safety boundaries, saturation, and broad resets are omitted.",
        "The common reward scale is omitted because it does not change deterministic LQR; it can matter for SAC.",
        "Discounted optimality alone does not guarantee physical stability; stable checks use the unscaled closed-loop matrix.", "",
        "## Held LQR Target", "",
        "| Pole (1/s) | omega0 (rad/s) | Damping |", "|---|---:|---:|",
    ]
    for mode in target:
        if mode["imag"] >= 0:
            report.append(f"| {mode['real']:.4f} +/- {mode['imag']:.4f}j | {mode['omega0']:.4f} | {mode['damping']:.4f} |")
    report += ["", "## One-Weight Sweeps", "",
               "Distance is mean relative complex-pole distance under optimal matching of four target poles; it is not a safety or performance score.",
               "All five candidate poles are listed (one row entry per conjugate pair). No automatic promotion.", "",
               "| gamma | Scenario | Stable | Distance | Poles: real +/- imag j; damping |",
               "|---:|---|---|---:|---|"]
    for record in records:
        text = "; ".join(
            f"{mode['real']:.3f} +/- {mode['imag']:.3f}j; {mode['damping']:.3f}"
            if abs(mode["imag"]) > 1e-8 else f"{mode['real']:.3f} (real)"
            for mode in record["modes"] if mode["imag"] >= 0
        )
        report.append(f"| {record['gamma']} | {record['scenario']} | {record['stable']} | {record['target_distance']:.4f} | {text} |")
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "cases": len(records), "checks": payload["checks"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
