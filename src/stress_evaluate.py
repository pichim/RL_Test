"""Run deterministic uncertainty-corner and long-duration controller tests."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np
from stable_baselines3 import SAC

from evaluate import (
    EPISODE_FIELDS,
    MODEL_PATH,
    episode_row,
    load_run_config,
    run_prepared_episode,
    save_detailed_episode,
)
from furuta_env import FurutaConfig, wrap_angle
from furuta_model import FurutaPendulum


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "runs" / "stress_stage3c_half_rps_v0"
STRESS_FIELDS = (
    "case_name",
    "profile",
    "duration_s",
    "expected_feasible",
    *EPISODE_FIELDS,
)


def parameter_profiles() -> dict[str, np.ndarray]:
    """Return adverse endpoint combinations for the seven plant scales."""
    return {
        "low_authority": np.array(
            [0.90, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10],
            dtype=np.float64,
        ),
        "high_authority": np.array(
            [1.10, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
            dtype=np.float64,
        ),
    }


def support_boundary_states(config: FurutaConfig) -> dict[str, np.ndarray]:
    """Return feasible-focused states that isolate reset-support boundaries."""
    maximum_arm_velocity = config.randomized_reset_omega1_half_range
    maximum_pendulum_velocity = config.randomized_reset_omega2_half_range
    return {
        "downward_center": np.array([0.0, 0.0, 0.0, 0.0]),
        "downward_offset": np.array([np.pi / 4.0, np.pi / 4.0, 0.0, 0.0]),
        # Near-travel-limit cases retain nominal reset speed; combining the
        # travel edge with maximum outward speed would intentionally create
        # states with questionable stopping distance.
        "positive_arm_outward": np.array([np.pi / 2.0, np.pi, 2.0, 0.0]),
        "negative_arm_outward": np.array([-np.pi / 2.0, np.pi, -2.0, 0.0]),
        # Maximum-velocity cases start at the arm center so travel remains
        # available for recovery.
        "positive_velocity_edges": np.array(
            [
                0.0,
                np.pi / 2.0,
                maximum_arm_velocity,
                -maximum_pendulum_velocity,
            ]
        ),
        "negative_velocity_edges": np.array(
            [
                0.0,
                -np.pi / 2.0,
                -maximum_arm_velocity,
                maximum_pendulum_velocity,
            ]
        ),
    }


def corner_cases(config: FurutaConfig) -> list[dict]:
    """Build exact endpoint combinations for deterministic acceptance tests."""
    cases = []
    for profile_name, scales in parameter_profiles().items():
        for cutoff_hz in (50.0, 100.0):
            for delay_samples in (0, 1):
                for state_name, state in support_boundary_states(config).items():
                    cases.append(
                        {
                            "name": (
                                f"corner_{profile_name}_f{int(cutoff_hz)}_"
                                f"d{delay_samples}_{state_name}"
                            ),
                            "profile": "corner",
                            "duration_s": config.episode_time,
                            "expected_feasible": True,
                            "scales": scales.copy(),
                            "cutoff_hz": cutoff_hz,
                            "delay_samples": delay_samples,
                            "state": state.copy(),
                        }
                    )
    return cases


def long_duration_cases(config: FurutaConfig, duration_s: float) -> list[dict]:
    """Build representative long-duration settling and drift cases."""
    profiles = parameter_profiles()
    return [
        {
            "name": "long_nominal_downward",
            "profile": "long_duration",
            "duration_s": duration_s,
            "expected_feasible": True,
            "scales": np.ones(7, dtype=np.float64),
            "cutoff_hz": 100.0,
            "delay_samples": 0,
            "state": np.array([0.0, 0.0, 0.0, 0.0]),
        },
        {
            "name": "long_low_authority_max_lag",
            "profile": "long_duration",
            "duration_s": duration_s,
            "expected_feasible": True,
            "scales": profiles["low_authority"].copy(),
            "cutoff_hz": 50.0,
            "delay_samples": 1,
            "state": np.array([0.0, 0.0, 0.0, 0.0]),
        },
        {
            "name": "long_high_authority_max_lag",
            "profile": "long_duration",
            "duration_s": duration_s,
            "expected_feasible": True,
            "scales": profiles["high_authority"].copy(),
            "cutoff_hz": 50.0,
            "delay_samples": 1,
            "state": np.array([np.pi / 4.0, np.pi / 4.0, 0.0, 0.0]),
        },
        {
            "name": "long_low_authority_broad_state",
            "profile": "long_duration",
            "duration_s": duration_s,
            "expected_feasible": True,
            "scales": profiles["low_authority"].copy(),
            "cutoff_hz": 50.0,
            "delay_samples": 1,
            "state": np.array([0.4, np.pi / 2.0, 2.0, -2.0]),
        },
    ]


def prepare_plant(case: dict, config: FurutaConfig) -> FurutaPendulum:
    """Create the exact deterministic plant specified by one stress case."""
    plant = FurutaPendulum(config.sample_time)
    plant.set_parameter_scales(*case["scales"])
    plant.current_filter_cutoff_hz = float(case["cutoff_hz"])
    plant.action_dead_time_samples = int(case["delay_samples"])
    plant.reset(np.asarray(case["state"], dtype=np.float64))
    plant.x[1] = wrap_angle(plant.x[1])
    return plant


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STRESS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(
    model_path: Path = MODEL_PATH,
    results_dir: Path = RESULTS_DIR,
    config_path: Path | None = None,
    controller: str = "both",
    profile: str = "both",
    long_duration_s: float = 30.0,
    overwrite: bool = False,
) -> None:
    if controller not in {"sac", "hybrid", "both"}:
        raise ValueError(f"unknown controller: {controller}")
    if profile not in {"corners", "long", "both"}:
        raise ValueError(f"unknown stress profile: {profile}")
    if long_duration_s < 5.0:
        raise ValueError("long duration must be at least five seconds")
    if not model_path.is_file():
        raise FileNotFoundError(f"No trained SAC model found: {model_path}")

    config, resolved_config_path = load_run_config(model_path, config_path)
    results_dir = results_dir.expanduser().resolve()
    protected_directories = {
        Path(results_dir.anchor),
        ROOT.resolve(),
        ROOT.parent.resolve(),
        (ROOT / "runs").resolve(),
        resolved_config_path.parent.resolve(),
    }
    if results_dir in protected_directories or results_dir.is_relative_to(
        resolved_config_path.parent.resolve()
    ):
        raise ValueError(f"Refusing to replace protected directory: {results_dir}")
    archive = results_dir.parent / f"{results_dir.name}_results.zip"
    if (results_dir.exists() or archive.exists()) and not overwrite:
        raise FileExistsError(
            "Stress output already exists. Choose another --results-dir or "
            "pass --overwrite."
        )
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    cases = []
    if profile in {"corners", "both"}:
        cases.extend(corner_cases(config))
    if profile in {"long", "both"}:
        cases.extend(long_duration_cases(config, long_duration_s))

    model = SAC.load(model_path, device="cpu")
    controllers = ("sac", "hybrid") if controller == "both" else (controller,)
    rows = []
    failures = []
    for active_controller in controllers:
        for index, case in enumerate(cases, start=1):
            episode_config = replace(config, episode_time=case["duration_s"])
            result = run_prepared_episode(
                model,
                prepare_plant(case, episode_config),
                episode_config,
                active_controller,
            )
            row = {
                "case_name": case["name"],
                "profile": case["profile"],
                "duration_s": case["duration_s"],
                "expected_feasible": int(case["expected_feasible"]),
                **episode_row(
                    active_controller,
                    case["profile"],
                    index,
                    -1,
                    result,
                ),
            }
            rows.append(row)
            if not result["success"] or result["unsafe"]:
                failures.append(
                    {
                        "controller": active_controller,
                        "case_name": case["name"],
                        "unsafe": bool(result["unsafe"]),
                    }
                )
                save_detailed_episode(
                    results_dir / active_controller / "failures" / case["name"],
                    f"{active_controller}/{case['name']}",
                    1,
                    -1,
                    result,
                    episode_config,
                )

    write_rows(results_dir / "stress_episodes.csv", rows)
    summary = {
        "model": str(model_path),
        "configuration": str(resolved_config_path),
        "profile": profile,
        "controllers": list(controllers),
        "cases_per_controller": len(cases),
        "total_cases": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "unsafe_cases": sum(int(row["unsafe"]) for row in rows),
        "failures": failures,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (results_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = results_dir / "artifacts"
    artifacts.mkdir()
    shutil.copy2(model_path, artifacts / "model.zip")
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=results_dir)
    print(json.dumps(summary, indent=2))
    print(f"results saved under: {results_dir}")
    print(f"shareable archive: {archive}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--controller", choices=("sac", "hybrid", "both"), default="both"
    )
    parser.add_argument(
        "--profile", choices=("corners", "long", "both"), default="both"
    )
    parser.add_argument("--long-duration", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.long_duration < 5.0:
        parser.error("--long-duration must be at least five seconds")
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    main(
        arguments.model,
        arguments.results_dir,
        arguments.config,
        arguments.controller,
        arguments.profile,
        arguments.long_duration,
        arguments.overwrite,
    )
