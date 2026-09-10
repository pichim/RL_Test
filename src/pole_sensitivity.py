"""Read-only local actor-gain sensitivity study; no training or policy changes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from stable_baselines3 import SAC

from analyze_damping import jacobian, plant_step, policy_analysis
from current_penalty import DEFAULT_JOB, TASK_CONFIG
from train import atomic_write_json, file_sha256


NAMES = ("theta1", "upright_error", "omega1", "omega2", "previous_action")
PERIOD = TASK_CONFIG.sample_time * TASK_CONFIG.action_repeat


def closed_loop(plant_a, plant_b, gain):
    matrix = np.zeros((5, 5))
    matrix[:4, :4] = plant_a
    matrix += np.r_[plant_b[:, 0], 1.0][:, None] @ np.asarray(gain)[None, :]
    return matrix


def match_poles(reference, poles):
    rows, columns = linear_sum_assignment(abs(reference[:, None] - poles[None, :]))
    ordered = np.empty_like(poles)
    ordered[rows] = poles[columns]
    return ordered


def track_poles(plant_a, plant_b, original, changed, reference, steps=100):
    poles = reference.copy()
    for fraction in np.linspace(0, 1, steps + 1)[1:]:
        gain = original + fraction * (changed - original)
        poles = match_poles(poles, np.linalg.eigvals(closed_loop(plant_a, plant_b, gain)).astype(complex))
    return poles


def metrics(poles, fast_indices):
    if np.any(abs(poles) < 1e-14):
        return {"stable": bool(np.all(abs(poles) < 1)), "finite_log": False,
                "tracked_pair_complex": False, "fast_omega_rad_s": None, "fast_damping": None,
                "slowest_decay_per_s": None, "other_max_omega_rad_s": None, "other_min_damping": None,
                "discrete_poles": [[float(pole.real), float(pole.imag)] for pole in poles],
                "continuous_poles": None}
    continuous = np.log(poles.astype(complex)) / PERIOD
    fast = continuous[fast_indices]
    others = np.delete(continuous, fast_indices)
    conjugate = bool(abs(fast[0].imag) > 1e-5 and np.isclose(fast[0], fast[1].conjugate()))
    damping = -continuous.real / np.maximum(abs(continuous), 1e-15)
    return {
        "stable": bool(np.all(abs(poles) < 1)),
        "finite_log": True,
        "tracked_pair_complex": conjugate,
        "fast_omega_rad_s": float(max(abs(fast))),
        "fast_damping": float(min(damping[fast_indices])),
        "slowest_decay_per_s": float(-max(continuous.real)),
        "other_max_omega_rad_s": float(max(abs(others))),
        "other_min_damping": float(min(np.delete(damping, fast_indices))),
        "discrete_poles": [[float(pole.real), float(pole.imag)] for pole in poles],
        "continuous_poles": [[float(pole.real), float(pole.imag)] for pole in continuous],
    }


def qualifies(candidate, reference):
    return bool(candidate["stable"] and candidate["finite_log"] and candidate["tracked_pair_complex"]
                and candidate["fast_omega_rad_s"] < reference["fast_omega_rad_s"]
                and candidate["fast_damping"] > reference["fast_damping"]
                and candidate["slowest_decay_per_s"] >= .9 * reference["slowest_decay_per_s"]
                and candidate["other_min_damping"] >= .7
                and candidate["other_max_omega_rad_s"] <= candidate["fast_omega_rad_s"])


def run(job, output):
    job, output = Path(job).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    saved = json.loads((job / "poles/analysis.json").read_text())
    protocol = json.loads((job / "protocol.json").read_text())
    model_path = job / "reference/candidate.zip"
    if file_sha256(model_path) != saved["models"]["candidate"]["sha256"]:
        raise ValueError("Candidate model differs from the saved pole analysis")
    for name in ("furuta_model.py", "furuta_env.py", "analyze_damping.py"):
        if file_sha256(Path(__file__).with_name(name)) != protocol["source_sha256"][name]:
            raise ValueError(f"Source differs from evaluated experiment: {name}")
    for name in ("sample_time", "action_repeat", "max_current"):
        if protocol["config"][name] != getattr(TASK_CONFIG, name):
            raise ValueError(f"Sample/current convention differs: {name}")
    equilibria = [item for item in saved["models"]["candidate"]["equilibria"]
                  if item["stable"] and abs(item["arm_angle_rad"]) <= TASK_CONFIG.balance_theta1]
    if len(equilibria) != 1:
        raise ValueError("Expected exactly one saved stable centered equilibrium")
    equilibrium = equilibria[0]
    model = SAC.load(model_path, device="cpu")
    recomputed = policy_analysis(model)
    verified = min(recomputed["equilibria"], key=lambda item: abs(item["arm_angle_rad"] - equilibrium["arm_angle_rad"]))
    np.testing.assert_allclose(verified["arm_angle_rad"], equilibrium["arm_angle_rad"], atol=1e-9, rtol=0)
    gain = np.array(verified["normalized_action_jacobian"])
    np.testing.assert_allclose(gain, equilibrium["normalized_action_jacobian"], atol=1e-8, rtol=1e-7)
    state = np.array([equilibrium["arm_angle_rad"], 0, 0, 0], dtype=float)
    plant_a = jacobian(lambda physical: plant_step(physical, 0), state, 1e-5)
    plant_b = jacobian(lambda action: plant_step(state, action[0]), np.zeros(1), 1e-5)
    poles = np.linalg.eigvals(closed_loop(plant_a, plant_b, gain)).astype(complex)
    expected = np.array([complex(item["discrete_real"], item["discrete_imag"])
                         for item in equilibrium["modes"]])
    np.testing.assert_allclose(match_poles(expected, poles), expected, atol=1e-8, rtol=1e-7)
    continuous = np.log(poles) / PERIOD
    positive = [index for index, pole in enumerate(continuous) if pole.imag > 1e-5]
    fast_index = max(positive, key=lambda index: abs(continuous[index]))
    partner = int(np.argmin(abs(poles - poles[fast_index].conjugate())))
    fast_indices = [fast_index, partner]
    reference = metrics(poles, fast_indices)
    rows = []

    def measure(label, factors, group):
        changed = gain * np.asarray(factors)
        tracked = track_poles(plant_a, plant_b, gain, changed, poles)
        result = metrics(tracked, fast_indices)
        row = {"label": label, "group": group, "factors": list(factors), "gain": changed.tolist(), **result}
        row["qualifies"] = qualifies(row, reference)
        rows.append(row)
        return row

    derivatives = []
    for index, name in enumerate(NAMES):
        near = []
        for factor in (.99, 1.01):
            factors = np.ones(5)
            factors[index] = factor
            near.append(measure(f"{name} x{factor:g}", factors.tolist(), "derivative"))
        derivatives.append({"gain": name,
            "omega_change_per_1pct_scale": (near[1]["fast_omega_rad_s"] - near[0]["fast_omega_rad_s"]) / 2,
            "damping_change_per_1pct_scale": (near[1]["fast_damping"] - near[0]["fast_damping"]) / 2})
        for factor in (.5, .75, .9, 1.1, 1.25, 1.5):
            factors = np.ones(5)
            factors[index] = factor
            measure(f"{name} x{factor:g}", factors.tolist(), "single")
    for state_scale in (.25, .5, .75, 1.0, 1.25):
        for memory_scale in (0.0, .25, .5, .75, 1.0, 1.25, 1.5):
            measure(f"states x{state_scale:g}, memory x{memory_scale:g}",
                    [state_scale] * 4 + [memory_scale], "combined")
    eligible = sorted([row for row in rows if row["qualifies"] and row["group"] != "derivative"],
                      key=lambda row: row["fast_omega_rad_s"])
    checks = []
    for row in eligible:
        changed = np.array(row["gain"])
        tracked = track_poles(plant_a, plant_b, gain, changed, poles, steps=200)
        original_tracking = np.array([complex(*value) for value in row["discrete_poles"]])
        np.testing.assert_allclose(match_poles(original_tracking, tracked), original_tracking, atol=1e-8)
        np.testing.assert_allclose(match_poles(original_tracking[fast_indices], tracked[fast_indices]),
                       original_tracking[fast_indices], atol=1e-8)
        point = np.r_[state, 0.0]

        def modified_step(local_state):
            action = float(changed @ (local_state - point))
            return np.r_[plant_step(local_state[:4], action), action]

        difference = float(np.max(abs(jacobian(modified_step, point, 1e-6) - closed_loop(plant_a, plant_b, changed))))
        if difference > 1e-5:
            raise AssertionError(f"Modified local map finite-difference discrepancy: {difference}")
        checks.append({"label": row["label"], "finite_difference_max_abs_error": difference})
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": file_sha256(model_path),
              "job": str(job), "python": sys.executable, "sample_period_s": PERIOD,
              "equilibrium_arm_deg": equilibrium["arm_angle_deg"], "gain_order": NAMES,
              "original_gain": gain.tolist(), "plant_a": plant_a.tolist(), "plant_b": plant_b.tolist(),
              "reference": reference, "derivatives": derivatives, "cases": rows,
              "eligible_labels": [row["label"] for row in eligible], "verification": checks,
              "original_fd_error": verified["finite_difference_check_max_abs_error"]}
    output.mkdir(parents=True)
    atomic_write_json(output / "analysis.json", result)
    shutil.copy2(Path(__file__), output / "pole_sensitivity_snapshot.py")
    lines = ["# Local Actor-Gain Sensitivity", "",
             f"Candidate SHA256: {result['model_sha256']}", "",
             f"Upright equilibrium: arm {equilibrium['arm_angle_deg']:.4f} degrees; sample period {PERIOD:g} s.",
             "Normalized-action derivatives are with respect to radians, rad/s, and previous normalized action.",
             "Gain factors multiply signed derivatives, not reward weights. The equilibrium and zero-action bias are held fixed.",
             "No model was changed or trained. These are hypothetical local feedback laws, not evaluated SAC policies.", "",
             "## Original Gain", "", "```json", json.dumps(dict(zip(NAMES, gain.tolist())), indent=2), "```", "",
             "## Differential Sensitivity", "",
             "Central differences at +/-1% gain scaling; entries are changes per +1% scaling.", "",
             "| Gain | Fast natural frequency change (rad/s) | Fast damping change |",
             "|---|---:|---:|"]
    for row in derivatives:
        lines.append(f"| {row['gain']} | {row['omega_change_per_1pct_scale']:.5f} | {row['damping_change_per_1pct_scale']:.6f} |")
    lines += ["", "## Screened Local Alternatives", "",
              "Screen: stable discrete poles, tracked fast pair remains complex, smaller fast natural frequency,",
              "greater fast damping, slowest decay >=90% of this candidate's original decay, other-mode damping >=0.7,",
              "and no other mode with greater natural frequency than the tracked pair. This is an exploratory screen.", "",
              "| Change | Fast omega (rad/s) | Fast damping | Slowest decay (1/s) |",
              "|---|---:|---:|---:|",
              f"| Original | {reference['fast_omega_rad_s']:.3f} | {reference['fast_damping']:.4f} | {reference['slowest_decay_per_s']:.4f} |"]
    for row in eligible[:12]:
        lines.append(f"| {row['label']} | {row['fast_omega_rad_s']:.3f} | {row['fast_damping']:.4f} | {row['slowest_decay_per_s']:.4f} |")
    if not eligible:
        lines.append("\nNo tested finite change passed all screens.")
    lines += ["", "## Interpretation Limits", "",
              "All cases and all five poles are retained in analysis.json, including rejected and unstable cases.",
              "Modes are tracked by minimum-distance assignment in the discrete plane along 100 interpolation steps;",
              "eligible endpoints are cross-checked at 200 steps. Mode identity at repeated roots is not unique.",
              "Zero discrete poles have no finite logarithmic equivalent; such cases are retained but excluded from the screen.",
              "Continuous-equivalent poles use the principal logarithm of the discrete poles, not a continuous-time controller model.",
              "The saved actor's gain/poles are reproduced and its Jacobian is finite-difference checked.",
              "Eligible hypothetical gains are also checked against the nonlinear plant's local held-action map.",
              "The full trained actor, its activation regions away from equilibrium, swing-up, saturation, and hardware dynamics",
              "have not been changed or tested by this study. A gain direction is not a prediction of what SAC will learn",
              "after changing a reward weight. A larger action-change penalty does not guarantee greater previous-action sensitivity."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"reference": reference, "derivatives": derivatives,
                      "best_screened": eligible[:3], "report": str(output / "REPORT.md")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.job_dir, args.output)
