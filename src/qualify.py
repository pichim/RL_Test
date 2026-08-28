"""Re-run the frozen Stage-3c controller qualification protocols."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys

from reproduce import file_sha256, working_tree_dirty


WORKSPACE = Path(__file__).resolve().parent.parent
RELEASE_DIR = WORKSPACE / "models" / "stage3c_half_rps_v0"
MANIFEST_PATH = RELEASE_DIR / "manifest.json"
MODEL_PATH = RELEASE_DIR / "model.zip"
SUITES = {
    "holdout": {
        "config": RELEASE_DIR / "config.json",
        "protocol": RELEASE_DIR / "acceptance.json",
        "results": WORKSPACE
        / "src"
        / "runs"
        / "qualification_stage3c_half_rps_v0_holdout",
    },
    "nominal80": {
        "config": RELEASE_DIR / "nominal80_no_delay_config.json",
        "protocol": RELEASE_DIR / "nominal80_no_delay_protocol.json",
        "results": WORKSPACE
        / "src"
        / "runs"
        / "qualification_stage3c_half_rps_v0_nominal80_no_delay",
    },
}
MODES = (
    "training_nominal",
    "training_randomized",
    "downward_rest_randomized",
)
SAC_LIMIT_FIELDS = {
    "swingup_current_change_rms_A": "mean_swingup_current_change_rms_A",
    "swingup_current_total_variation_A": (
        "mean_swingup_current_total_variation_A"
    ),
    "arm_excursion_rad": "mean_arm_excursion_rad",
    "omega1_rms_rad_s": "mean_omega1_rms_rad_s",
    "omega2_rms_rad_s": "mean_omega2_rms_rad_s",
    "final_arm_error_rms_rad": "mean_final_arm_error_rms_rad",
    "final_omega1_rms_rad_s": "mean_final_omega1_rms_rad_s",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_release() -> dict:
    """Verify every immutable release artifact against its manifest hash."""
    manifest = load_json(MANIFEST_PATH)
    for relative_path, expected_hash in manifest["files"].items():
        path = RELEASE_DIR / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing release artifact: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"release artifact hash mismatch for {relative_path}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
    for recipe_key in ("source_recipe", "canonical_reproduction_recipe"):
        recipe = manifest[recipe_key]
        recipe_path = WORKSPACE / recipe["path"]
        actual_hash = file_sha256(recipe_path)
        if actual_hash != recipe["sha256"]:
            raise RuntimeError(
                f"recipe hash mismatch for {recipe['path']}: "
                f"expected {recipe['sha256']}, found {actual_hash}"
            )
    if manifest["model_sha256"] != manifest["files"]["model.zip"]:
        raise RuntimeError("manifest model hashes are inconsistent")
    for record_name in (
        "acceptance.json",
        "nominal80_no_delay_protocol.json",
        "selection_report.json",
        "holdout_report.json",
        "nominal80_no_delay_report.json",
    ):
        record = load_json(RELEASE_DIR / record_name)
        if record["model_sha256"] != manifest["model_sha256"]:
            raise RuntimeError(f"{record_name} identifies a different model")
    return manifest


def build_suite_command(suite: str, overwrite: bool = False) -> list[str]:
    definition = SUITES[suite]
    protocol = load_json(definition["protocol"])
    command = [
        sys.executable,
        str(WORKSPACE / "src" / "evaluate.py"),
        "--model",
        str(MODEL_PATH),
        "--config",
        str(definition["config"]),
        "--controller",
        "both",
        "--episodes",
        str(protocol["episodes_per_mode"]),
        "--base-seed",
        str(protocol["base_seed"]),
        "--results-dir",
        str(definition["results"]),
    ]
    if overwrite:
        command.append("--overwrite")
    return command


def verify_nominal80_config(config: dict, protocol: dict) -> None:
    requirements = protocol["configuration_requirements"]
    if config["current_filter_cutoff_hz"] != requirements[
        "current_filter_cutoff_hz"
    ]:
        raise RuntimeError("nominal-80 profile has the wrong filter cutoff")
    if config["current_filter_cutoff_min_hz"] != requirements[
        "current_filter_cutoff_hz"
    ] or config["current_filter_cutoff_max_hz"] != requirements[
        "current_filter_cutoff_hz"
    ]:
        raise RuntimeError("nominal-80 profile does not fix the filter cutoff")
    if config["action_dead_time_max_samples"] != 0:
        raise RuntimeError("nominal-80 profile does not disable action dead time")
    randomized_fields = (
        "motor_torque_randomization",
        "arm_mass_randomization",
        "arm_inertia_randomization",
        "pendulum_mass_randomization",
        "pendulum_inertia_randomization",
        "arm_damping_randomization",
        "pendulum_damping_randomization",
    )
    if config["training_parameter_randomization"] or any(
        config[field] != 0.0 for field in randomized_fields
    ):
        raise RuntimeError("nominal-80 profile still varies physical parameters")


def verify_summary(
    suite: str,
    summary_paths: list[Path] | None = None,
    verify_episode_details: bool = True,
) -> None:
    """Raise unless an evaluation result satisfies its frozen protocol."""
    definition = SUITES[suite]
    protocol = load_json(definition["protocol"])
    if suite == "nominal80":
        verify_nominal80_config(load_json(definition["config"]), protocol)
    paths = summary_paths or [definition["results"] / "summary.csv"]
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    indexed = {(row["controller"], row["mode"]): row for row in rows}
    required_keys = {
        (controller, mode)
        for controller in protocol["controllers"]
        for mode in MODES
    }
    if set(indexed) != required_keys:
        raise RuntimeError(
            f"qualification summary rows differ from protocol: {set(indexed)}"
        )
    hard = protocol["hard_requirements"]
    for key, row in indexed.items():
        observed = {
            "successes_per_mode": int(row["success_count"]),
            "unsafe_per_mode": int(row["unsafe_count"]),
            "captures_per_mode": int(row["capture_count"]),
        }
        for requirement, expected in hard.items():
            if observed[requirement] != expected:
                raise RuntimeError(
                    f"{suite} {key} failed {requirement}: "
                    f"expected {expected}, found {observed[requirement]}"
                )
    for mode, limits in protocol.get("sac_mean_limits", {}).items():
        row = indexed[("sac", mode)]
        for metric, limit in limits.items():
            value = float(row[SAC_LIMIT_FIELDS[metric]])
            if value > limit:
                raise RuntimeError(
                    f"{suite} SAC/{mode} exceeded {metric}: {value} > {limit}"
                )
    if suite == "nominal80" and verify_episode_details:
        scale_fields = (
            "motor_torque_scale",
            "arm_mass_scale",
            "arm_inertia_scale",
            "pendulum_mass_scale",
            "pendulum_inertia_scale",
            "arm_damping_scale",
            "pendulum_damping_scale",
        )
        for controller, mode in required_keys:
            row = indexed[(controller, mode)]
            for field in (
                "mean_current_filter_cutoff_hz",
                "minimum_current_filter_cutoff_hz",
                "maximum_current_filter_cutoff_hz",
            ):
                if float(row[field]) != 80.0:
                    raise RuntimeError(f"nominal80 {controller}/{mode} varied cutoff")
            if int(row["maximum_action_dead_time_samples"]) != 0:
                raise RuntimeError(f"nominal80 {controller}/{mode} used dead time")
            episode_path = (
                definition["results"] / controller / mode / "episodes.csv"
            )
            with episode_path.open(newline="", encoding="utf-8") as file:
                episode_rows = list(csv.DictReader(file))
            for episode in episode_rows:
                if any(float(episode[field]) != 1.0 for field in scale_fields):
                    raise RuntimeError(
                        f"nominal80 {controller}/{mode} varied a plant scale"
                    )
                if float(episode["current_filter_cutoff_hz"]) != 80.0:
                    raise RuntimeError(
                        f"nominal80 {controller}/{mode} varied filter cutoff"
                    )
                if int(episode["action_dead_time_samples"]) != 0:
                    raise RuntimeError(
                        f"nominal80 {controller}/{mode} used action dead time"
                    )


def verify_frozen_evidence() -> None:
    """Check that the committed raw summaries still pass their protocols."""
    evidence = RELEASE_DIR / "evidence"
    verify_summary(
        "holdout",
        [
            evidence / "holdout_sac_summary.csv",
            evidence / "holdout_hybrid_summary.csv",
        ],
        verify_episode_details=False,
    )
    verify_summary(
        "nominal80",
        [evidence / "nominal80_no_delay_summary.csv"],
        verify_episode_details=False,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("holdout", "nominal80", "all"),
        default="all",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute qualification; without this flag only print commands",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the canonical local result directories",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit noncanonical execution from an uncommitted code state",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    manifest = verify_release()
    verify_frozen_evidence()
    if arguments.execute and not arguments.allow_dirty and working_tree_dirty():
        raise RuntimeError(
            "Refusing qualification from a dirty Git tree. Commit the intended "
            "code and release records first, or pass --allow-dirty for an "
            "explicitly noncanonical recheck."
        )
    selected = (
        SUITES
        if arguments.suite == "all"
        else {arguments.suite: SUITES[arguments.suite]}
    )
    print(
        f"Release: {manifest['artifact_version']}\n"
        f"Model SHA-256: {manifest['model_sha256']}\n"
        f"Mode: {'execute' if arguments.execute else 'dry run'}",
        flush=True,
    )
    for suite in selected:
        command = build_suite_command(suite, arguments.overwrite)
        print(f"\n[{suite}]\n{shlex.join(command)}", flush=True)
        if not arguments.execute:
            continue
        subprocess.run(command, cwd=WORKSPACE, check=True)
        verify_summary(suite)
        print(f"qualification passed: {suite}", flush=True)


if __name__ == "__main__":
    main()
