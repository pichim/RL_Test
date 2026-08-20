"""Run a tracked, resumable multi-stage SAC training recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_RECIPE = WORKSPACE / "experiments" / "stage3c_half_rps_v0.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def working_tree_dirty() -> bool:
    """Return whether tracked or untracked workspace changes are present."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=WORKSPACE,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def load_recipe(path: Path) -> dict:
    path = path.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("recipe_version") != 1:
        raise ValueError(f"unsupported recipe version in {path}")
    stages = data.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("recipe must contain a nonempty stages list")
    identifiers = [stage.get("id") for stage in stages]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("every stage requires a nonempty id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("recipe stage ids must be unique")
    known: set[str] = set()
    for stage in stages:
        mode = stage.get("initialization")
        if mode not in {"scratch", "resume", "curriculum"}:
            raise ValueError(f"invalid initialization for {stage['id']}: {mode}")
        parent = stage.get("parent")
        if mode == "scratch" and parent is not None:
            raise ValueError(f"scratch stage {stage['id']} cannot have a parent")
        if mode != "scratch" and parent not in known:
            raise ValueError(
                f"stage {stage['id']} must reference an earlier parent stage"
            )
        if not isinstance(stage.get("timesteps"), int) or stage["timesteps"] <= 0:
            raise ValueError(f"stage {stage['id']} requires positive timesteps")
        if not isinstance(stage.get("run_dir"), str):
            raise ValueError(f"stage {stage['id']} requires run_dir")
        if not isinstance(stage.get("args", []), list) or not all(
            isinstance(value, str) for value in stage.get("args", [])
        ):
            raise ValueError(f"stage {stage['id']} args must be strings")
        known.add(stage["id"])
    data["_path"] = str(path)
    data["_sha256"] = file_sha256(path)
    return data


def stage_run_dir(stage: dict, seed: int) -> Path:
    rendered = stage["run_dir"].format(seed=seed)
    path = (WORKSPACE / rendered).resolve()
    if not path.is_relative_to((WORKSPACE / "src" / "runs").resolve()):
        raise ValueError(f"stage run directory must stay under src/runs: {path}")
    return path


def build_stage_command(recipe: dict, stage: dict, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(WORKSPACE / "src" / "train.py"),
        "--timesteps",
        str(stage["timesteps"]),
        "--seed",
        str(seed),
        "--run-dir",
        str(stage_run_dir(stage, seed)),
    ]
    if stage["initialization"] != "scratch":
        parent = next(
            item for item in recipe["stages"] if item["id"] == stage["parent"]
        )
        parent_state = stage_run_dir(parent, seed) / "final"
        command.extend(
            [
                "--curriculum-model"
                if stage["initialization"] == "curriculum"
                else "--resume-model",
                str(parent_state),
            ]
        )
    command.extend(stage.get("args", []))
    return command


def stage_marker(recipe: dict, stage: dict, seed: int, command: list[str]) -> dict:
    run_dir = stage_run_dir(stage, seed)
    final = run_dir / "final"
    state = json.loads((final / "state.json").read_text(encoding="utf-8"))
    marker = {
        "recipe": str(Path(recipe["_path"]).relative_to(WORKSPACE)),
        "recipe_sha256": recipe["_sha256"],
        "stage": stage["id"],
        "seed": seed,
        "command": command,
        "output": {
            "num_timesteps": state["num_timesteps"],
            "model_sha256": file_sha256(final / "model.zip"),
            "replay_buffer_sha256": file_sha256(final / "replay_buffer.pkl"),
        },
    }
    if stage["initialization"] != "scratch":
        parent = next(
            item for item in recipe["stages"] if item["id"] == stage["parent"]
        )
        parent_final = stage_run_dir(parent, seed) / "final"
        marker["parent"] = {
            "stage": parent["id"],
            "model_sha256": file_sha256(parent_final / "model.zip"),
            "replay_buffer_sha256": file_sha256(
                parent_final / "replay_buffer.pkl"
            ),
        }
    return marker


def completed_stage_matches(recipe: dict, stage: dict, seed: int) -> bool:
    marker_path = stage_run_dir(stage, seed) / "pipeline_stage.json"
    if not marker_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("recipe_sha256") != recipe["_sha256"]
        or marker.get("stage") != stage["id"]
        or marker.get("seed") != seed
    ):
        return False
    final = stage_run_dir(stage, seed) / "final"
    return (
        (final / "model.zip").is_file()
        and (final / "replay_buffer.pkl").is_file()
        and marker.get("output", {}).get("model_sha256")
        == file_sha256(final / "model.zip")
        and marker.get("output", {}).get("replay_buffer_sha256")
        == file_sha256(final / "replay_buffer.pkl")
    )


def run_stage(recipe: dict, stage: dict, seed: int, execute: bool) -> None:
    command = build_stage_command(recipe, stage, seed)
    print(f"\n[{stage['id']}]\n{shlex.join(command)}", flush=True)
    if not execute:
        return
    if stage["initialization"] != "scratch":
        parent = next(
            item for item in recipe["stages"] if item["id"] == stage["parent"]
        )
        if not completed_stage_matches(recipe, parent, seed):
            raise RuntimeError(
                f"Parent stage is not a verified product of this recipe: "
                f"{parent['id']}"
            )
    run_dir = stage_run_dir(stage, seed)
    if completed_stage_matches(recipe, stage, seed):
        print(f"verified complete; skipping: {run_dir}", flush=True)
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Unverified or incomplete stage directory exists: {run_dir}\n"
            "Preserve it and choose a new seed/recipe, or resume it explicitly "
            "with train.py. The pipeline will not overwrite training evidence."
        )
    subprocess.run(command, cwd=WORKSPACE, check=True)
    marker = stage_marker(recipe, stage, seed, command)
    (run_dir / "pipeline_stage.json").write_text(
        json.dumps(marker, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"verified and recorded: {run_dir}", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--seed", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stage", help="run or display one stage id")
    selection.add_argument(
        "--all",
        action="store_true",
        help="run or display the complete chain in recipe order",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute commands; without this flag the recipe is a dry run",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit execution from an uncommitted code state (not recommended)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.seed < 0:
        raise ValueError("seed must be nonnegative")
    if arguments.execute and not arguments.allow_dirty and working_tree_dirty():
        raise RuntimeError(
            "Refusing a reproducibility run from a dirty Git tree. Commit the "
            "intended code/configuration first, or pass --allow-dirty for an "
            "explicitly noncanonical experiment."
        )
    recipe = load_recipe(arguments.recipe)
    stages = recipe["stages"]
    if arguments.stage is not None:
        selected = [stage for stage in stages if stage["id"] == arguments.stage]
        if not selected:
            raise ValueError(f"unknown stage: {arguments.stage}")
    else:
        selected = stages
    print(
        f"Recipe: {recipe['name']}\n"
        f"Definition: {recipe['_path']}\n"
        f"SHA-256: {recipe['_sha256']}\n"
        f"Seed: {arguments.seed}\n"
        f"Mode: {'execute' if arguments.execute else 'dry run'}",
        flush=True,
    )
    for stage in selected:
        run_stage(recipe, stage, arguments.seed, arguments.execute)


if __name__ == "__main__":
    main()
