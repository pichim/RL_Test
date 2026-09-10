"""Re-score saved slow weight-4 trajectories without changing old results."""

import csv
import json
from pathlib import Path

import numpy as np

from nominal import CONFIG


def diagnose():
    root = Path(__file__).resolve().parent / "runs/nominal_sac_actor64_delta4_seed0_v0/comparison/fresh_long/weight4"
    constraints = {
        "pendulum_angle": ("upright_error_rad", CONFIG.balance_angle),
        "arm_position": ("theta1_rad", CONFIG.balance_theta1),
        "arm_speed": ("omega1_rad_s", CONFIG.balance_omega1),
        "pendulum_speed": ("omega2_rad_s", CONFIG.balance_omega2),
    }
    results = []
    for episode, seed in ((204, 280022), (211, 280029), (258, 280076)):
        with (root / f"episode_{episode:03d}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        times = np.array([float(row["time_s"]) for row in rows])
        masks = {name: np.array([abs(float(row[field])) <= limit for row in rows])
                 for name, (field, limit) in constraints.items()}
        final_second = (times > 4.0) & (times <= 5.0)
        result = {"seed": seed, "violating_samples_4_to_5s": {
            name: int(np.sum(~mask[final_second])) for name, mask in masks.items()
        }}
        for mode in ("original", "no_centering"):
            valid = np.logical_and.reduce([mask for name, mask in masks.items()
                                           if mode == "original" or name != "arm_position"])
            valid &= np.array([abs(float(row["theta1_rad"])) <= CONFIG.arm_angle_limit
                               and row["unsafe"] == "0" for row in rows])
            bad = np.flatnonzero(~valid)
            start = float(times[bad[-1]]) if bad.size else 0.0
            result[mode] = {"success_at_5s": bool(np.all(valid[final_second])),
                            "final_hold_complete_s": start + CONFIG.balance_hold_time}
        results.append(result)
    return results


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2))
