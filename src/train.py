"""Train direct-current SAC for complete Furuta swing-up and balance."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn

from furuta_env import DEFAULT_CONFIG, FurutaConfig, FurutaSwingUpEnv


SEED = 0
ACTOR_NETWORK = [8, 8]
CRITIC_NETWORK = [64, 64]
DEFAULT_TOTAL_TIMESTEPS = 1_000_000
DEFAULT_EVALUATION_EPISODES = 50
DEFAULT_RESUME_SNAPSHOT_FREQUENCY = 100_000
DEFAULT_CURRICULUM_RAMP_STEPS = 1_000_000
DEFAULT_CURRICULUM_FINAL_LEARNING_RATE = 1e-4
DEFAULT_CURRICULUM_FINAL_LEARNING_RATE_STEPS = 500_000
ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = (
    ROOT / "runs" / "sac_200hz_matlab_pure_actor8_critic64_stage1_armv015_v0"
)
EVALUATION_SEED = 10_000


def find_run_file(model_path: Path, filename: str) -> Path:
    """Find run metadata beside a model, committed state, or rolling state."""
    candidates = (
        model_path.parent / filename,
        model_path.parent.parent / filename,
        model_path.parent.parent.parent / filename,
    )
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            f"Could not find {filename} beside resume model: {model_path}"
        )
    return found


def resolve_resume_model(model_path: Path) -> Path:
    """Resolve a model file, committed state directory, or rolling snapshot root."""
    model_path = model_path.expanduser()
    if model_path.is_file():
        return model_path
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Resume model or state directory not found: {model_path}"
        )

    committed_model = model_path / "model.zip"
    committed_replay = model_path / "replay_buffer.pkl"
    if committed_model.is_file() and committed_replay.is_file():
        return committed_model

    latest_path = model_path / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(
            f"No committed model/replay pair found under: {model_path}"
        )
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    snapshot_name = latest.get("snapshot")
    if (
        not isinstance(snapshot_name, str)
        or Path(snapshot_name).name != snapshot_name
        or not snapshot_name.startswith("snapshot_")
    ):
        raise ValueError(f"Invalid rolling snapshot pointer: {latest_path}")
    snapshot = model_path / snapshot_name
    committed_model = snapshot / "model.zip"
    committed_replay = snapshot / "replay_buffer.pkl"
    if not committed_model.is_file() or not committed_replay.is_file():
        raise FileNotFoundError(
            f"Rolling snapshot pointer does not name a complete pair: {snapshot}"
        )
    return committed_model


def atomic_write_json(path: Path, data: dict) -> None:
    """Publish a small JSON file with one same-filesystem atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact without loading it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_directory(source: Path, destination: Path) -> None:
    """Publish a directory despite transient Windows file-indexing locks."""
    for attempt in range(60):
        try:
            source.rename(destination)
            return
        except PermissionError:
            if destination.exists() or attempt == 59:
                raise
            time.sleep(0.25)


def remove_directory_best_effort(path: Path) -> None:
    """Remove stale state without failing training on transient file locks."""
    for attempt in range(60):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 59:
                print(f"warning: could not remove stale state: {path}")
                return
            time.sleep(0.25)


def runtime_provenance() -> dict:
    """Capture the code and Python environment needed to interpret a run."""
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT.parent,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
        git_dirty = None
    packages = {}
    for package in ("numpy", "gymnasium", "stable-baselines3", "torch"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "command": [sys.executable, *sys.argv],
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "environment_prefix": sys.prefix,
        "packages": packages,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def save_committed_state(model: SAC, destination: Path) -> None:
    """Atomically publish one complete SAC model/replay directory."""
    if destination.exists():
        raise FileExistsError(f"Committed state already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-",
            dir=destination.parent,
        )
    )
    try:
        model.save(temporary / "model.zip")
        model.save_replay_buffer(temporary / "replay_buffer.pkl")
        model_path = temporary / "model.zip"
        replay_path = temporary / "replay_buffer.pkl"
        (temporary / "state.json").write_text(
            json.dumps(
                {
                    "num_timesteps": int(model.num_timesteps),
                    "replay_buffer_size": int(model.replay_buffer.size()),
                    "model_sha256": file_sha256(model_path),
                    "replay_buffer_sha256": file_sha256(replay_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        publish_directory(temporary, destination)
    finally:
        if temporary.exists():
            remove_directory_best_effort(temporary)


def resolve_resume_replay_buffer(
    resume_model_path: Path,
    replay_buffer_path: Path | None,
) -> Path:
    """Resolve the replay buffer paired with a full-state resume model."""
    resolved = (
        replay_buffer_path
        if replay_buffer_path is not None
        else resume_model_path.parent / "replay_buffer.pkl"
    )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Resume replay buffer not found: {resolved}\n"
            "Pass --resume-replay-buffer explicitly, or resume from a final "
            "model/resume snapshot that has a sibling replay_buffer.pkl."
        )
    return resolved


def validate_resume_config(
    resume_model_path: Path,
    config: FurutaConfig,
) -> Path:
    """Require identical environment/reward configuration for full resume."""
    config_path = find_run_file(resume_model_path, "config.json")
    saved_raw = json.loads(config_path.read_text(encoding="utf-8"))
    saved = asdict(FurutaConfig(**saved_raw))
    requested = asdict(config)
    differences = {
        key: (saved.get(key, "<absent>"), requested.get(key, "<absent>"))
        for key in sorted(set(saved) | set(requested))
        if saved.get(key, "<absent>") != requested.get(key, "<absent>")
    }
    if differences:
        details = "\n".join(
            f"  {key}: saved={old!r}, requested={new!r}"
            for key, (old, new) in differences.items()
        )
        raise ValueError(
            "Full-state resume requires the exact saved task configuration.\n"
            f"Differences:\n{details}\n"
            "Use --initial-actor for transfer to changed dynamics/rewards, "
            "or omit continuation flags to train from scratch."
        )
    return config_path


CURRICULUM_CHANGE_FIELDS = {
    "training_parameter_randomization",
    "current_filter_cutoff_randomization_hz",
    "current_filter_cutoff_min_hz",
    "current_filter_cutoff_max_hz",
    "motor_torque_randomization",
    "arm_mass_randomization",
    "arm_inertia_randomization",
    "pendulum_mass_randomization",
    "pendulum_inertia_randomization",
    "arm_damping_randomization",
    "pendulum_damping_randomization",
    "randomized_reset_omega1_half_range",
    "randomized_reset_omega2_half_range",
    "action_dead_time_max_samples",
}


def validate_curriculum_config(
    curriculum_model_path: Path,
    config: FurutaConfig,
) -> Path:
    """Allow a full-state transition that changes only uncertainty ranges."""
    config_path = find_run_file(curriculum_model_path, "config.json")
    saved_raw = json.loads(config_path.read_text(encoding="utf-8"))
    saved = asdict(FurutaConfig(**saved_raw))
    requested = asdict(config)
    forbidden = {
        key: (saved[key], requested[key])
        for key in sorted(saved)
        if key not in CURRICULUM_CHANGE_FIELDS and saved[key] != requested[key]
    }
    if forbidden:
        details = "\n".join(
            f"  {key}: saved={old!r}, requested={new!r}"
            for key, (old, new) in forbidden.items()
        )
        raise ValueError(
            "Curriculum continuation may change only the documented uncertainty "
            f"ranges.\nForbidden differences:\n{details}"
        )
    if saved["training_parameter_randomization"]:
        raise ValueError(
            "A new curriculum must start from a nominal-training source. "
            "Use --resume-model to continue an existing curriculum."
        )
    if not config.training_parameter_randomization:
        raise ValueError("Curriculum continuation requires parameter randomization")
    return config_path


def curriculum_scale(
    num_timesteps: int,
    start_num_timesteps: int,
    ramp_steps: int,
) -> float:
    """Return the linear uncertainty fraction for an absolute SAC timestep."""
    if ramp_steps <= 0:
        raise ValueError("curriculum ramp steps must be positive")
    elapsed = int(num_timesteps) - int(start_num_timesteps)
    return float(min(1.0, max(0.0, elapsed / ramp_steps)))


def set_model_learning_rate(model: SAC, learning_rate: float) -> None:
    """Set SAC's actor, critic, and entropy learning-rate schedule together."""
    learning_rate = float(learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    model.learning_rate = learning_rate
    model.lr_schedule = get_schedule_fn(learning_rate)
    optimizers = [model.actor.optimizer, model.critic.optimizer]
    if model.ent_coef_optimizer is not None:
        optimizers.append(model.ent_coef_optimizer)
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


class CurriculumCallback(BaseCallback):
    """Ramp episode uncertainty and reduce SAC's final refinement step size."""

    def __init__(
        self,
        environment: FurutaSwingUpEnv,
        start_num_timesteps: int,
        ramp_steps: int,
        final_learning_rate_start: int,
        final_learning_rate: float,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.environment = environment
        self.start_num_timesteps = int(start_num_timesteps)
        self.ramp_steps = int(ramp_steps)
        self.final_learning_rate_start = int(final_learning_rate_start)
        self.final_learning_rate = float(final_learning_rate)
        self._final_rate_applied = False
        self._reported_quarter = -1

    def _apply(self) -> None:
        scale = curriculum_scale(
            self.model.num_timesteps,
            self.start_num_timesteps,
            self.ramp_steps,
        )
        self.environment.set_randomization_scale(scale)
        self.logger.record("curriculum/randomization_scale", scale)
        quarter = min(4, int(scale * 4.0 + 1e-12))
        if quarter > self._reported_quarter:
            self._reported_quarter = quarter
            if self.verbose >= 1:
                print(
                    "curriculum randomization scale: "
                    f"{scale:.1%} at timestep {self.model.num_timesteps:,}"
                )
        if (
            self.model.num_timesteps >= self.final_learning_rate_start
            and not self._final_rate_applied
        ):
            set_model_learning_rate(self.model, self.final_learning_rate)
            self._final_rate_applied = True
            if self.verbose >= 1:
                print(
                    "curriculum final learning rate: "
                    f"{self.final_learning_rate:g} at timestep "
                    f"{self.model.num_timesteps:,}"
                )
        self.logger.record(
            "curriculum/learning_rate",
            self.final_learning_rate
            if self._final_rate_applied
            else float(self.model.learning_rate),
        )

    def _on_training_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:
        self._apply()
        return True


class ResumeStateCallback(BaseCallback):
    """Publish one complete rolling model/replay state at a time."""

    def __init__(self, save_freq: int, save_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if save_freq <= 0:
            raise ValueError("save_freq must be positive")
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self._last_save_call = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Save after the collected transitions have entered the replay buffer.
        # With train_freq=2 this is normally exact; the >= form also remains
        # useful if a later experiment chooses a non-divisible frequency.
        if self.n_calls - self._last_save_call >= self.save_freq:
            self.save_path.mkdir(parents=True, exist_ok=True)
            snapshot = self.save_path / (
                f"snapshot_{int(self.model.num_timesteps):012d}"
            )
            save_committed_state(self.model, snapshot)
            atomic_write_json(
                self.save_path / "latest.json",
                {
                    "snapshot": snapshot.name,
                    "num_timesteps": int(self.model.num_timesteps),
                },
            )
            # The new pointer is committed. Removing an older generation now
            # cannot invalidate the published snapshot.
            for candidate in self.save_path.glob("snapshot_*"):
                if candidate != snapshot and candidate.is_dir():
                    remove_directory_best_effort(candidate)
            self._last_save_call = self.n_calls
            if self.verbose >= 1:
                print(f"saved rolling resume state: {snapshot}")


class MilestoneStateCallback(BaseCallback):
    """Retain complete model/replay states at selected absolute timesteps."""

    def __init__(
        self,
        milestones: list[int],
        save_path: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if not milestones or any(value <= 0 for value in milestones):
            raise ValueError("milestones must contain positive timesteps")
        self.pending = set(int(value) for value in milestones)
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        reached = sorted(
            value for value in self.pending if value <= self.model.num_timesteps
        )
        for value in reached:
            if value != self.model.num_timesteps:
                raise RuntimeError(
                    "A requested resume milestone was skipped; milestone "
                    "timesteps must align with the training frequency"
                )
            destination = self.save_path / f"state_{value:012d}"
            save_committed_state(self.model, destination)
            self.pending.remove(value)
            if self.verbose >= 1:
                print(f"saved retained resume milestone: {destination}")


def checkpoint_selection_score(
    success_rate: float,
    mean_reward: float,
) -> tuple[float, float]:
    """Rank validation results by reliability, then the training objective."""
    return float(success_rate), float(mean_reward)


class SuccessFirstEvalCallback(EvalCallback):
    """Save by success rate, then mean reward."""

    def __init__(
        self,
        *args,
        reset_seed: int,
        best_model_save_path: str,
        **kwargs,
    ) -> None:
        # EvalCallback otherwise selects only by undiscounted episode reward.
        # That is unsuitable when successful episodes terminate early.
        super().__init__(*args, best_model_save_path=None, **kwargs)
        self.reset_seed = int(reset_seed)
        self.success_model_directory = Path(best_model_save_path)
        self.best_success_score = (-1.0, -float("inf"))

    def _on_step(self) -> bool:
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating:
            self.eval_env.seed(self.reset_seed)
        continue_training = super()._on_step()
        if evaluating and self._is_success_buffer:
            success_rate = sum(self._is_success_buffer) / len(
                self._is_success_buffer
            )
            score = checkpoint_selection_score(
                success_rate,
                self.last_mean_reward,
            )
            if score > self.best_success_score:
                self.best_success_score = score
                self.success_model_directory.mkdir(parents=True, exist_ok=True)
                self.model.save(
                    self.success_model_directory / "best_model.zip"
                )
                if self.verbose >= 1:
                    print(
                        "New best success-first model: "
                        f"success={success_rate:.1%}, "
                        f"mean_reward={self.last_mean_reward:.2f}"
                    )
        return continue_training


def algorithm_settings(
    config: FurutaConfig,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    seed: int = SEED,
    actor_network: list[int] = ACTOR_NETWORK,
    critic_network: list[int] = CRITIC_NETWORK,
    initial_actor_path: Path | None = None,
    resume_model_path: Path | None = None,
    resume_replay_buffer_path: Path | None = None,
    resume_snapshot_frequency: int = DEFAULT_RESUME_SNAPSHOT_FREQUENCY,
    resume_milestone_timesteps: list[int] | None = None,
    curriculum_model_path: Path | None = None,
    curriculum_ramp_steps: int = DEFAULT_CURRICULUM_RAMP_STEPS,
    curriculum_final_learning_rate: float = (
        DEFAULT_CURRICULUM_FINAL_LEARNING_RATE
    ),
    curriculum_final_learning_rate_steps: int = (
        DEFAULT_CURRICULUM_FINAL_LEARNING_RATE_STEPS
    ),
    evaluation_episodes: int = DEFAULT_EVALUATION_EPISODES,
) -> dict:
    """Return the small, physically timed SAC training configuration."""
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if total_timesteps % 2 != 0:
        raise ValueError("total_timesteps must be even for train_freq=2")
    if evaluation_episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    if resume_snapshot_frequency < 0:
        raise ValueError("resume_snapshot_frequency must be nonnegative")
    if resume_snapshot_frequency % 2 != 0:
        raise ValueError("resume_snapshot_frequency must be even or zero")
    resume_milestone_timesteps = list(resume_milestone_timesteps or [])
    if any(value <= 0 or value % 2 != 0 for value in resume_milestone_timesteps):
        raise ValueError("resume milestones must be positive even timesteps")
    if len(set(resume_milestone_timesteps)) != len(resume_milestone_timesteps):
        raise ValueError("resume milestones must be unique")
    continuation_count = sum(
        path is not None
        for path in (initial_actor_path, resume_model_path, curriculum_model_path)
    )
    if continuation_count > 1:
        raise ValueError("initialization and continuation paths are exclusive")
    if curriculum_model_path is not None:
        if curriculum_ramp_steps <= 0 or curriculum_ramp_steps > total_timesteps:
            raise ValueError("curriculum ramp steps must be in (0, total_timesteps]")
        if (
            curriculum_final_learning_rate_steps < 0
            or curriculum_final_learning_rate_steps > total_timesteps
        ):
            raise ValueError(
                "curriculum final learning-rate steps must be in [0, total_timesteps]"
            )
        if curriculum_final_learning_rate <= 0.0:
            raise ValueError("curriculum final learning rate must be positive")
    policy_period = config.sample_time * config.action_repeat
    steps_per_episode = int(round(config.episode_time / policy_period))
    initialization = (
        "curriculum_resume"
        if curriculum_model_path is not None
        else "full_resume"
        if resume_model_path is not None
        else "actor_only"
        if initial_actor_path is not None
        else "from_scratch"
    )
    return {
        "steps_per_episode": steps_per_episode,
        "total_timesteps": total_timesteps,
        "learning_starts": 40_000,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "train_frequency": 2,
        "gradient_steps": 1,
        "target_update_interval": 1,
        "entropy_coefficient": "auto_0.1",
        "target_entropy": -1.0,
        "checkpoint_frequency": 100_000,
        "resume_snapshot_frequency": int(resume_snapshot_frequency),
        "resume_milestone_timesteps": sorted(resume_milestone_timesteps),
        # Ensure even a deliberately short smoke run produces a best model.
        "evaluation_frequency": min(100_000, total_timesteps),
        "evaluation_episodes": int(evaluation_episodes),
        "evaluation_seed": EVALUATION_SEED,
        "best_model_selection": "success_rate_then_mean_reward",
        "control_rate_hz": 1.0 / policy_period,
        "actor_network": list(actor_network),
        "critic_network": list(critic_network),
        "initialization": initialization,
        "initial_actor": (
            str(initial_actor_path.resolve()) if initial_actor_path else None
        ),
        "resume_model": (
            str(resume_model_path.resolve()) if resume_model_path else None
        ),
        "resume_replay_buffer": (
            str(resume_replay_buffer_path.resolve())
            if resume_replay_buffer_path
            else None
        ),
        "curriculum_model": (
            str(curriculum_model_path.resolve())
            if curriculum_model_path
            else None
        ),
        "curriculum_ramp_steps": int(curriculum_ramp_steps),
        "curriculum_final_learning_rate": float(
            curriculum_final_learning_rate
        ),
        "curriculum_final_learning_rate_steps": int(
            curriculum_final_learning_rate_steps
        ),
        "seed": int(seed),
    }


def make_environment(config: FurutaConfig, run_dir: Path) -> Monitor:
    """Create the configured MATLAB-reset training environment."""
    return Monitor(
        FurutaSwingUpEnv(config),
        filename=str(run_dir / "training"),
        info_keywords=("is_success", "initial_capture", "unsafe"),
    )


def train(
    config: FurutaConfig,
    run_dir: Path,
    total_timesteps: int,
    seed: int = SEED,
    actor_network: list[int] = ACTOR_NETWORK,
    critic_network: list[int] = CRITIC_NETWORK,
    initial_actor_path: Path | None = None,
    resume_model_path: Path | None = None,
    resume_replay_buffer_path: Path | None = None,
    resume_snapshot_frequency: int = DEFAULT_RESUME_SNAPSHOT_FREQUENCY,
    resume_milestone_timesteps: list[int] | None = None,
    curriculum_model_path: Path | None = None,
    curriculum_ramp_steps: int = DEFAULT_CURRICULUM_RAMP_STEPS,
    curriculum_final_learning_rate: float = (
        DEFAULT_CURRICULUM_FINAL_LEARNING_RATE
    ),
    curriculum_final_learning_rate_steps: int = (
        DEFAULT_CURRICULUM_FINAL_LEARNING_RATE_STEPS
    ),
    evaluation_episodes: int = DEFAULT_EVALUATION_EPISODES,
) -> None:
    """Train or fully continue one isolated experiment."""
    continuation_count = sum(
        path is not None
        for path in (initial_actor_path, resume_model_path, curriculum_model_path)
    )
    if continuation_count > 1:
        raise ValueError("initialization and continuation paths are mutually exclusive")
    explicit_curriculum = curriculum_model_path is not None
    resolved_resume_model = (
        curriculum_model_path if explicit_curriculum else resume_model_path
    )
    resolved_replay_buffer = resume_replay_buffer_path
    source_config_path = None
    if initial_actor_path is not None and not initial_actor_path.is_file():
        raise FileNotFoundError(f"Initial actor model not found: {initial_actor_path}")
    if resolved_resume_model is not None:
        resolved_resume_model = resolve_resume_model(resolved_resume_model)
        source_config_path = (
            validate_curriculum_config(resolved_resume_model, config)
            if explicit_curriculum
            else validate_resume_config(resolved_resume_model, config)
        )
        resolved_replay_buffer = resolve_resume_replay_buffer(
            resolved_resume_model,
            resume_replay_buffer_path,
        )

    if run_dir.exists() and (
        not run_dir.is_dir() or any(run_dir.iterdir())
    ):
        raise FileExistsError(
            f"The run directory is not empty: {run_dir}\n"
            "Choose another --run-dir to preserve the existing experiment. "
            "Continuation always writes to a new run directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    final_state_path = run_dir / "final"
    model_path = final_state_path / "model.zip"
    replay_buffer_path = final_state_path / "replay_buffer.pkl"

    settings = algorithm_settings(
        config=config,
        total_timesteps=total_timesteps,
        seed=seed,
        actor_network=actor_network,
        critic_network=critic_network,
        initial_actor_path=initial_actor_path,
        resume_model_path=(None if explicit_curriculum else resolved_resume_model),
        resume_replay_buffer_path=resolved_replay_buffer,
        evaluation_episodes=evaluation_episodes,
        resume_snapshot_frequency=resume_snapshot_frequency,
        resume_milestone_timesteps=resume_milestone_timesteps,
        curriculum_model_path=(resolved_resume_model if explicit_curriculum else None),
        curriculum_ramp_steps=curriculum_ramp_steps,
        curriculum_final_learning_rate=curriculum_final_learning_rate,
        curriculum_final_learning_rate_steps=(
            curriculum_final_learning_rate_steps
        ),
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )

    checked_env = FurutaSwingUpEnv(config)
    try:
        check_env(checked_env, warn=True)
    finally:
        checked_env.close()
    env = make_environment(config, run_dir)
    # Best-model selection stays on a fixed nominal plant. Robustness is tested
    # separately by evaluate.py, so callback scores remain comparable.
    evaluation_config = replace(config, training_parameter_randomization=False)
    evaluation_env = Monitor(
        FurutaSwingUpEnv(evaluation_config),
        info_keywords=("is_success", "initial_capture", "unsafe"),
    )
    curriculum_active = False
    curriculum_environment = env.unwrapped
    curriculum_schedule: dict[str, int | float] = {}
    if resolved_resume_model is not None:
        model = SAC.load(
            resolved_resume_model,
            env=env,
            device="auto",
            tensorboard_log=str(run_dir / "tensorboard"),
            verbose=1,
        )
        model.load_replay_buffer(resolved_replay_buffer)
        source_training_path = find_run_file(
            resolved_resume_model,
            "training.json",
        )
        source_settings = json.loads(
            source_training_path.read_text(encoding="utf-8")
        )
        # These are properties of the loaded learner, not controls for this
        # invocation. Preserve their source metadata instead of claiming that
        # current CLI defaults changed the restored model.
        for key in (
            "learning_starts",
            "buffer_size",
            "batch_size",
            "learning_rate",
            "gamma",
            "tau",
            "train_frequency",
            "gradient_steps",
            "target_update_interval",
            "entropy_coefficient",
            "target_entropy",
            "actor_network",
            "critic_network",
        ):
            if key in source_settings:
                settings[key] = source_settings[key]
        settings["source_config"] = str(source_config_path.resolve())
        settings["source_training"] = str(source_training_path.resolve())
        settings["source_seed"] = source_settings.get("seed")
        if not explicit_curriculum and "seed" in source_settings:
            settings["seed"] = source_settings["seed"]
        settings["starting_num_timesteps"] = int(model.num_timesteps)
        settings["requested_num_timesteps_after_run"] = int(
            model.num_timesteps + total_timesteps
        )
        if explicit_curriculum:
            curriculum_active = True
            model.seed = int(seed)
            model.set_random_seed(seed)
            settings["seed"] = int(seed)
            final_learning_rate_start = int(
                model.num_timesteps + total_timesteps + 1
            )
            if curriculum_final_learning_rate_steps > 0:
                final_learning_rate_start = int(
                    model.num_timesteps
                    + total_timesteps
                    - curriculum_final_learning_rate_steps
                )
            curriculum_schedule = {
                "curriculum_start_num_timesteps": int(model.num_timesteps),
                "curriculum_ramp_steps": int(curriculum_ramp_steps),
                "curriculum_total_steps": int(total_timesteps),
                "curriculum_full_range_hold_steps": int(
                    total_timesteps - curriculum_ramp_steps
                ),
                "curriculum_final_learning_rate": float(
                    curriculum_final_learning_rate
                ),
                "curriculum_final_learning_rate_steps": int(
                    curriculum_final_learning_rate_steps
                ),
                "curriculum_final_learning_rate_start": (
                    final_learning_rate_start
                ),
            }
            settings.update(curriculum_schedule)
            settings["training_mode"] = "curriculum"
        elif source_settings.get("training_mode") == "curriculum":
            curriculum_active = True
            schedule_keys = (
                "curriculum_start_num_timesteps",
                "curriculum_ramp_steps",
                "curriculum_total_steps",
                "curriculum_full_range_hold_steps",
                "curriculum_final_learning_rate",
                "curriculum_final_learning_rate_steps",
                "curriculum_final_learning_rate_start",
            )
            missing = [key for key in schedule_keys if key not in source_settings]
            if missing:
                raise ValueError(
                    "Cannot resume curriculum because training metadata is missing: "
                    + ", ".join(missing)
                )
            curriculum_schedule = {
                key: source_settings[key] for key in schedule_keys
            }
            settings.update(curriculum_schedule)
            settings["training_mode"] = "curriculum"

        if curriculum_active:
            scale = curriculum_scale(
                model.num_timesteps,
                int(curriculum_schedule["curriculum_start_num_timesteps"]),
                int(curriculum_schedule["curriculum_ramp_steps"]),
            )
            curriculum_environment.set_randomization_scale(scale)
            final_rate_start = int(
                curriculum_schedule["curriculum_final_learning_rate_start"]
            )
            if model.num_timesteps >= final_rate_start:
                set_model_learning_rate(
                    model,
                    float(curriculum_schedule["curriculum_final_learning_rate"]),
                )
            settings["starting_randomization_scale"] = scale
            settings["starting_learning_rate"] = float(model.lr_schedule(1.0))

        print(
            f"{'curriculum-resumed' if explicit_curriculum else 'fully resumed'} "
            f"model from: {resolved_resume_model}\n"
            f"loaded replay buffer: {resolved_replay_buffer}\n"
            f"starting timestep: {model.num_timesteps:,}; "
            f"additional decisions requested: {total_timesteps:,}"
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=settings["learning_rate"],
            buffer_size=settings["buffer_size"],
            learning_starts=settings["learning_starts"],
            batch_size=settings["batch_size"],
            gamma=settings["gamma"],
            tau=settings["tau"],
            train_freq=settings["train_frequency"],
            gradient_steps=settings["gradient_steps"],
            target_update_interval=settings["target_update_interval"],
            ent_coef=settings["entropy_coefficient"],
            target_entropy=settings["target_entropy"],
            policy_kwargs={
                "net_arch": {
                    "pi": settings["actor_network"],
                    "qf": settings["critic_network"],
                }
            },
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=settings["seed"],
            device="auto",
            verbose=1,
        )
        settings["starting_num_timesteps"] = 0
        settings["requested_num_timesteps_after_run"] = int(total_timesteps)
        if initial_actor_path is not None:
            source = SAC.load(initial_actor_path, device="cpu")
            try:
                model.actor.load_state_dict(source.actor.state_dict(), strict=True)
            except RuntimeError as error:
                raise ValueError(
                    "The initial actor is incompatible with the requested actor "
                    "architecture or observation/action spaces. Use matching "
                    "--actor-hidden-sizes or omit --initial-actor."
                ) from error
            print(
                f"initialized actor from: {initial_actor_path}\n"
                "critics, target critics, optimizers, and replay buffer start fresh"
            )

    settings["provenance"] = runtime_provenance()
    if resolved_resume_model is not None:
        settings["source_artifact_hashes"] = {
            "model_sha256": file_sha256(resolved_resume_model),
            "replay_buffer_sha256": file_sha256(resolved_replay_buffer),
        }
    (run_dir / "training.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"SAC rate: {settings['control_rate_hz']:g} Hz; "
        f"this invocation: {settings['total_timesteps']:,} decisions; "
        f"gamma: {model.gamma:.6f}"
    )

    callback_items: list[BaseCallback] = [
        CheckpointCallback(
            save_freq=settings["checkpoint_frequency"],
            save_path=str(run_dir / "checkpoints"),
            name_prefix="sac",
            save_replay_buffer=False,
        ),
        SuccessFirstEvalCallback(
            evaluation_env,
            reset_seed=settings["evaluation_seed"],
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "evaluation"),
            eval_freq=settings["evaluation_frequency"],
            n_eval_episodes=settings["evaluation_episodes"],
            deterministic=True,
            render=False,
            verbose=1,
            warn=False,
        ),
    ]
    if curriculum_active:
        callback_items.insert(
            0,
            CurriculumCallback(
                curriculum_environment,
                int(curriculum_schedule["curriculum_start_num_timesteps"]),
                int(curriculum_schedule["curriculum_ramp_steps"]),
                int(
                    curriculum_schedule[
                        "curriculum_final_learning_rate_start"
                    ]
                ),
                float(curriculum_schedule["curriculum_final_learning_rate"]),
                verbose=1,
            ),
        )
    if settings["resume_snapshot_frequency"] > 0:
        callback_items.append(
            ResumeStateCallback(
                settings["resume_snapshot_frequency"],
                run_dir / "resume",
                verbose=1,
            )
        )
    if settings["resume_milestone_timesteps"]:
        starting_timestep = int(settings["starting_num_timesteps"])
        ending_timestep = int(settings["requested_num_timesteps_after_run"])
        invalid = [
            value
            for value in settings["resume_milestone_timesteps"]
            if not starting_timestep < value <= ending_timestep
        ]
        if invalid:
            raise ValueError(
                "resume milestones must lie inside this invocation's absolute "
                f"timestep interval ({starting_timestep}, {ending_timestep}]: "
                f"{invalid}"
            )
        callback_items.append(
            MilestoneStateCallback(
                settings["resume_milestone_timesteps"],
                run_dir / "milestones",
                verbose=1,
            )
        )
    callbacks = CallbackList(callback_items)

    try:
        model.learn(
            total_timesteps=settings["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=resolved_resume_model is None,
            progress_bar=True,
        )
        save_committed_state(model, final_state_path)
        settings["final_num_timesteps"] = int(model.num_timesteps)
        settings["final_replay_buffer_size"] = int(model.replay_buffer.size())
        final_state = json.loads(
            (final_state_path / "state.json").read_text(encoding="utf-8")
        )
        settings["final_artifact_hashes"] = {
            "model_sha256": final_state["model_sha256"],
            "replay_buffer_sha256": final_state["replay_buffer_sha256"],
        }
        if curriculum_active:
            settings["final_randomization_scale"] = curriculum_scale(
                model.num_timesteps,
                int(curriculum_schedule["curriculum_start_num_timesteps"]),
                int(curriculum_schedule["curriculum_ramp_steps"]),
            )
            settings["final_learning_rate"] = float(model.lr_schedule(1.0))
        (run_dir / "training.json").write_text(
            json.dumps(settings, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved model: {model_path}")
        print(f"saved replay buffer: {replay_buffer_path}")
        print(f"best evaluated model: {run_dir / 'best' / 'best_model.zip'}")
    finally:
        env.close()
        evaluation_env.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help=(
            "even number of environment decisions: total for a new learner "
            "or additional for continuation (default: 1,000,000)"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"output directory (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"random seed (default: {SEED})",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=DEFAULT_EVALUATION_EPISODES,
        help=(
            "fixed nominal validation episodes per checkpoint "
            f"(default: {DEFAULT_EVALUATION_EPISODES})"
        ),
    )
    parser.add_argument(
        "--actor-hidden-sizes",
        type=int,
        nargs="+",
        default=ACTOR_NETWORK,
        metavar="N",
        help="actor hidden-layer sizes (default: 8 8)",
    )
    parser.add_argument(
        "--critic-hidden-sizes",
        type=int,
        nargs="+",
        default=CRITIC_NETWORK,
        metavar="N",
        help="critic hidden-layer sizes (default: 64 64)",
    )
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument(
        "--initial-actor",
        type=Path,
        default=None,
        help=(
            "optionally copy only the actor weights from this model; "
            "the default initializes the complete SAC model from scratch"
        ),
    )
    continuation.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help=(
            "fully continue a model file or committed final/rolling state "
            "directory, including critics, optimizers, entropy, and replay"
        ),
    )
    continuation.add_argument(
        "--curriculum-model",
        type=Path,
        default=None,
        help=(
            "start a gradual uncertainty curriculum from a complete model/replay "
            "state; unlike --resume-model, only documented randomization changes "
            "are permitted"
        ),
    )
    parser.add_argument(
        "--resume-replay-buffer",
        type=Path,
        default=None,
        help=(
            "replay buffer paired with --resume-model/--curriculum-model; "
            "by default use the "
            "sibling replay_buffer.pkl"
        ),
    )
    parser.add_argument(
        "--resume-snapshot-frequency",
        type=int,
        default=DEFAULT_RESUME_SNAPSHOT_FREQUENCY,
        help=(
            "atomically publish a rolling model/replay state every N decisions; "
            "zero disables rolling snapshots (default: 100000)"
        ),
    )
    parser.add_argument(
        "--resume-milestone-timesteps",
        type=int,
        nargs="*",
        default=[],
        metavar="N",
        help=(
            "absolute SAC timesteps at which to retain complete model/replay "
            "states; unlike the rolling snapshot, milestones are not deleted"
        ),
    )
    parser.add_argument(
        "--constraint-reward-weight",
        type=float,
        default=DEFAULT_CONFIG.constraint_reward_weight,
        help=(
            "feasible-state indicator F reward weight "
            f"(default: {DEFAULT_CONFIG.constraint_reward_weight:g})"
        ),
    )
    parser.add_argument(
        "--action-change-weight",
        type=float,
        default=DEFAULT_CONFIG.action_change_weight,
        help=(
            "normalized squared action-change penalty weight "
            f"(default: {DEFAULT_CONFIG.action_change_weight:g})"
        ),
    )
    parser.add_argument(
        "--arm-angle-weight",
        type=float,
        default=DEFAULT_CONFIG.arm_angle_weight,
        help=(
            "global squared arm-angle penalty weight "
            f"(default: {DEFAULT_CONFIG.arm_angle_weight:g})"
        ),
    )
    parser.add_argument(
        "--arm-velocity-weight",
        type=float,
        default=DEFAULT_CONFIG.arm_velocity_weight,
        help=(
            "global squared arm-velocity penalty weight "
            f"(default: {DEFAULT_CONFIG.arm_velocity_weight:g})"
        ),
    )
    parser.add_argument(
        "--pendulum-velocity-weight",
        type=float,
        default=DEFAULT_CONFIG.velocity_weight,
        help=(
            "global squared pendulum-velocity penalty weight "
            f"(default: {DEFAULT_CONFIG.velocity_weight:g})"
        ),
    )
    parser.add_argument(
        "--current-filter-cutoff-hz",
        type=float,
        default=DEFAULT_CONFIG.current_filter_cutoff_hz,
        help=(
            "hidden continuous first-order current cutoff in Hz; zero disables it "
            f"(default: {DEFAULT_CONFIG.current_filter_cutoff_hz:g})"
        ),
    )
    parser.add_argument(
        "--current-filter-cutoff-randomization-hz",
        type=float,
        default=DEFAULT_CONFIG.current_filter_cutoff_randomization_hz,
        help=(
            "symmetric per-episode cutoff range around the nominal cutoff; "
            "40 with a 100 Hz nominal produces 60--140 Hz at full scale "
            f"(default: {DEFAULT_CONFIG.current_filter_cutoff_randomization_hz:g})"
        ),
    )
    parser.add_argument(
        "--current-filter-cutoff-range-hz",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help=(
            "explicit full-curriculum cutoff interval; it expands from the "
            "nominal cutoff and is mutually exclusive with the symmetric "
            "cutoff-randomization option"
        ),
    )
    parser.add_argument(
        "--parameter-randomization",
        action="store_true",
        help=(
            "enable configured per-episode plant, actuator, delay, and reset "
            "uncertainties (default: nominal task)"
        ),
    )
    parser.add_argument(
        "--action-dead-time-max-samples",
        type=int,
        default=DEFAULT_CONFIG.action_dead_time_max_samples,
        help=(
            "maximum hidden RL action delay in 200 Hz policy samples; each "
            "episode samples an integer from zero to the active curriculum "
            f"maximum (default: {DEFAULT_CONFIG.action_dead_time_max_samples})"
        ),
    )
    parser.add_argument(
        "--randomized-reset-omega1-half-range-rps",
        type=float,
        default=(
            DEFAULT_CONFIG.randomized_reset_omega1_half_range / (2.0 * np.pi)
        ),
        help=(
            "full-curriculum arm-velocity reset half-range in rotations/s "
            "(default preserves the nominal reset distribution)"
        ),
    )
    parser.add_argument(
        "--randomized-reset-omega2-half-range-rps",
        type=float,
        default=(
            DEFAULT_CONFIG.randomized_reset_omega2_half_range / (2.0 * np.pi)
        ),
        help=(
            "full-curriculum pendulum-velocity reset half-range in rotations/s "
            "(default preserves the nominal reset distribution)"
        ),
    )
    for option, field, label in (
        ("--motor-torque-randomization", "motor_torque_randomization", "motor torque"),
        ("--arm-mass-randomization", "arm_mass_randomization", "arm mass"),
        ("--arm-inertia-randomization", "arm_inertia_randomization", "arm inertia"),
        (
            "--pendulum-mass-randomization",
            "pendulum_mass_randomization",
            "pendulum mass",
        ),
        (
            "--pendulum-inertia-randomization",
            "pendulum_inertia_randomization",
            "pendulum inertia",
        ),
        (
            "--arm-damping-randomization",
            "arm_damping_randomization",
            "arm damping",
        ),
        (
            "--pendulum-damping-randomization",
            "pendulum_damping_randomization",
            "pendulum damping",
        ),
    ):
        parser.add_argument(
            option,
            dest=field,
            type=float,
            default=getattr(DEFAULT_CONFIG, field),
            help=(
                f"symmetric fractional {label} range "
                f"(default: {getattr(DEFAULT_CONFIG, field):g})"
            ),
        )
    parser.add_argument(
        "--curriculum-ramp-steps",
        type=int,
        default=DEFAULT_CURRICULUM_RAMP_STEPS,
        help=(
            "decisions used to linearly ramp uncertainty from zero to the "
            f"configured ranges (default: {DEFAULT_CURRICULUM_RAMP_STEPS})"
        ),
    )
    parser.add_argument(
        "--curriculum-final-learning-rate",
        type=float,
        default=DEFAULT_CURRICULUM_FINAL_LEARNING_RATE,
        help=(
            "SAC learning rate during final curriculum refinement "
            f"(default: {DEFAULT_CURRICULUM_FINAL_LEARNING_RATE:g})"
        ),
    )
    parser.add_argument(
        "--curriculum-final-learning-rate-steps",
        type=int,
        default=DEFAULT_CURRICULUM_FINAL_LEARNING_RATE_STEPS,
        help=(
            "number of final decisions using the reduced learning rate; zero "
            "disables the reduction "
            f"(default: {DEFAULT_CURRICULUM_FINAL_LEARNING_RATE_STEPS})"
        ),
    )
    arguments = parser.parse_args()
    if arguments.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if arguments.timesteps % 2 != 0:
        parser.error("--timesteps must be even because train_freq is 2")
    if arguments.evaluation_episodes <= 0:
        parser.error("--evaluation-episodes must be positive")
    if arguments.resume_snapshot_frequency < 0:
        parser.error("--resume-snapshot-frequency must be nonnegative")
    if arguments.resume_snapshot_frequency % 2 != 0:
        parser.error("--resume-snapshot-frequency must be even or zero")
    if any(
        value <= 0 or value % 2 != 0
        for value in arguments.resume_milestone_timesteps
    ):
        parser.error(
            "--resume-milestone-timesteps values must be positive and even"
        )
    if len(set(arguments.resume_milestone_timesteps)) != len(
        arguments.resume_milestone_timesteps
    ):
        parser.error("--resume-milestone-timesteps values must be unique")
    if (
        arguments.resume_replay_buffer is not None
        and arguments.resume_model is None
        and arguments.curriculum_model is None
    ):
        parser.error(
            "--resume-replay-buffer requires --resume-model or --curriculum-model"
        )
    if arguments.constraint_reward_weight < 0:
        parser.error("--constraint-reward-weight must be nonnegative")
    if arguments.action_change_weight < 0:
        parser.error("--action-change-weight must be nonnegative")
    if arguments.arm_angle_weight <= 0:
        parser.error("--arm-angle-weight must be positive")
    if arguments.arm_velocity_weight <= 0:
        parser.error("--arm-velocity-weight must be positive")
    if arguments.pendulum_velocity_weight <= 0:
        parser.error("--pendulum-velocity-weight must be positive")
    if arguments.current_filter_cutoff_hz < 0:
        parser.error("--current-filter-cutoff-hz must be nonnegative")
    if arguments.action_dead_time_max_samples < 0:
        parser.error("--action-dead-time-max-samples must be nonnegative")
    for option, value, nominal, safety in (
        (
            "--randomized-reset-omega1-half-range-rps",
            arguments.randomized_reset_omega1_half_range_rps * 2.0 * np.pi,
            DEFAULT_CONFIG.reset_omega1_half_range,
            DEFAULT_CONFIG.max_abs_omega1,
        ),
        (
            "--randomized-reset-omega2-half-range-rps",
            arguments.randomized_reset_omega2_half_range_rps * 2.0 * np.pi,
            DEFAULT_CONFIG.reset_omega2_half_range,
            DEFAULT_CONFIG.max_abs_omega2,
        ),
    ):
        if not np.isfinite(value) or value < nominal or value >= safety:
            parser.error(
                f"{option} must convert to a bound in [{nominal:g}, {safety:g}) rad/s"
            )
    if arguments.current_filter_cutoff_randomization_hz < 0:
        parser.error(
            "--current-filter-cutoff-randomization-hz must be nonnegative"
        )
    if (
        arguments.current_filter_cutoff_randomization_hz > 0.0
        and arguments.current_filter_cutoff_randomization_hz
        >= arguments.current_filter_cutoff_hz
    ):
        parser.error(
            "--current-filter-cutoff-randomization-hz must be smaller than "
            "--current-filter-cutoff-hz"
        )
    if arguments.current_filter_cutoff_range_hz is not None:
        minimum_cutoff, maximum_cutoff = arguments.current_filter_cutoff_range_hz
        if arguments.current_filter_cutoff_randomization_hz > 0.0:
            parser.error(
                "--current-filter-cutoff-range-hz cannot be combined with "
                "--current-filter-cutoff-randomization-hz"
            )
        if (
            not np.isfinite(minimum_cutoff)
            or not np.isfinite(maximum_cutoff)
            or minimum_cutoff <= 0.0
            or minimum_cutoff > arguments.current_filter_cutoff_hz
            or maximum_cutoff < arguments.current_filter_cutoff_hz
        ):
            parser.error(
                "--current-filter-cutoff-range-hz must be positive and contain "
                "--current-filter-cutoff-hz"
            )
    if arguments.curriculum_model is not None:
        if not arguments.parameter_randomization:
            parser.error("--curriculum-model requires --parameter-randomization")
        if (
            arguments.curriculum_ramp_steps <= 0
            or arguments.curriculum_ramp_steps > arguments.timesteps
            or arguments.curriculum_ramp_steps % 2 != 0
        ):
            parser.error(
                "--curriculum-ramp-steps must be positive, even, and no greater "
                "than --timesteps"
            )
        if (
            arguments.curriculum_final_learning_rate_steps < 0
            or arguments.curriculum_final_learning_rate_steps > arguments.timesteps
            or arguments.curriculum_final_learning_rate_steps % 2 != 0
        ):
            parser.error(
                "--curriculum-final-learning-rate-steps must be nonnegative, "
                "even, and no greater than --timesteps"
            )
        if arguments.curriculum_final_learning_rate <= 0.0:
            parser.error("--curriculum-final-learning-rate must be positive")
    if any(size <= 0 for size in arguments.actor_hidden_sizes):
        parser.error("--actor-hidden-sizes values must be positive")
    if any(size <= 0 for size in arguments.critic_hidden_sizes):
        parser.error("--critic-hidden-sizes values must be positive")
    for field in (
        "motor_torque_randomization",
        "arm_mass_randomization",
        "arm_inertia_randomization",
        "pendulum_mass_randomization",
        "pendulum_inertia_randomization",
        "arm_damping_randomization",
        "pendulum_damping_randomization",
    ):
        value = getattr(arguments, field)
        if value < 0.0 or value >= 1.0:
            parser.error(f"--{field.replace('_', '-')} must be in [0, 1)")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    config = replace(
        DEFAULT_CONFIG,
        constraint_reward_weight=arguments.constraint_reward_weight,
        action_change_weight=arguments.action_change_weight,
        arm_angle_weight=arguments.arm_angle_weight,
        arm_velocity_weight=arguments.arm_velocity_weight,
        velocity_weight=arguments.pendulum_velocity_weight,
        current_filter_cutoff_hz=arguments.current_filter_cutoff_hz,
        current_filter_cutoff_randomization_hz=(
            arguments.current_filter_cutoff_randomization_hz
        ),
        current_filter_cutoff_min_hz=(
            arguments.current_filter_cutoff_range_hz[0]
            if arguments.current_filter_cutoff_range_hz is not None
            else 0.0
        ),
        current_filter_cutoff_max_hz=(
            arguments.current_filter_cutoff_range_hz[1]
            if arguments.current_filter_cutoff_range_hz is not None
            else 0.0
        ),
        training_parameter_randomization=arguments.parameter_randomization,
        action_dead_time_max_samples=arguments.action_dead_time_max_samples,
        randomized_reset_omega1_half_range=(
            arguments.randomized_reset_omega1_half_range_rps * 2.0 * np.pi
        ),
        randomized_reset_omega2_half_range=(
            arguments.randomized_reset_omega2_half_range_rps * 2.0 * np.pi
        ),
        motor_torque_randomization=arguments.motor_torque_randomization,
        arm_mass_randomization=arguments.arm_mass_randomization,
        arm_inertia_randomization=arguments.arm_inertia_randomization,
        pendulum_mass_randomization=arguments.pendulum_mass_randomization,
        pendulum_inertia_randomization=arguments.pendulum_inertia_randomization,
        arm_damping_randomization=arguments.arm_damping_randomization,
        pendulum_damping_randomization=arguments.pendulum_damping_randomization,
    )
    train(
        config=config,
        run_dir=arguments.run_dir,
        total_timesteps=arguments.timesteps,
        seed=arguments.seed,
        actor_network=arguments.actor_hidden_sizes,
        critic_network=arguments.critic_hidden_sizes,
        initial_actor_path=arguments.initial_actor,
        resume_model_path=arguments.resume_model,
        resume_replay_buffer_path=arguments.resume_replay_buffer,
        evaluation_episodes=arguments.evaluation_episodes,
        resume_snapshot_frequency=arguments.resume_snapshot_frequency,
        resume_milestone_timesteps=arguments.resume_milestone_timesteps,
        curriculum_model_path=arguments.curriculum_model,
        curriculum_ramp_steps=arguments.curriculum_ramp_steps,
        curriculum_final_learning_rate=(
            arguments.curriculum_final_learning_rate
        ),
        curriculum_final_learning_rate_steps=(
            arguments.curriculum_final_learning_rate_steps
        ),
    )


if __name__ == "__main__":
    main()
