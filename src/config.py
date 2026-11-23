"""Edit these settings, then run train.py."""

from dataclasses import dataclass
from math import isfinite, pi
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
REFERENCE = ROOT / "models" / "reference"
TASK_VERSION = "periodic_arm_legacy_reward_v1"


@dataclass(frozen=True)
class Task:
    task_version: str = TASK_VERSION
    sample_time: float = 0.001
    action_repeat: int = 5
    episode_time: float = 5.0
    max_current: float = 1.0
    reset_theta1_half_range: float = pi
    reset_theta2_half_range: float = pi
    reset_omega1_half_range: float = 6.0
    reset_omega2_half_range: float = 6.0
    max_abs_omega1: float = 30.0
    max_abs_omega2: float = 30.0
    capture_angle: float = pi / 18
    capture_omega1: float = 2.0
    capture_omega2: float = 2.0
    balance_angle: float = pi / 36
    balance_theta1: float = 0.25
    balance_omega1: float = 0.50
    balance_omega2: float = 1.0
    balance_hold_time: float = 1.0
    teacher_bounds: tuple[float, ...] = (0.20, 0.10, 0.50, 1.00)
    terminal_success_bonus: float = 100.0
    constraint_reward_weight: float = 1.0
    reward_scale: float = 0.1
    velocity_weight: float = 0.010
    arm_angle_weight: float = 1.5
    arm_velocity_weight: float = 0.015
    # Weights multiply normalized commands inside the reward scale.
    current_weight: float = 3.0
    action_change_weight: float = 15.0

    def __post_init__(self):
        if self.task_version != TASK_VERSION:
            raise ValueError(f"Unsupported task version {self.task_version!r}")
        if type(self.action_repeat) is not int or self.action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        if any(not isfinite(value) or value < 0 for key, value in vars(self).items()
               if key not in ("task_version", "teacher_bounds")):
            raise ValueError("Task parameters must be finite and nonnegative")
        if len(self.teacher_bounds) != 4 or any(not isfinite(v) or v <= 0 for v in self.teacher_bounds):
            raise ValueError("teacher_bounds must contain four positive bounds")
        if not (self.sample_time > 0 and self.max_current > 0
                and self.episode_time >= self.balance_hold_time > 0
                and self.max_abs_omega1 > self.reset_omega1_half_range
                and self.max_abs_omega2 > self.reset_omega2_half_range):
            raise ValueError("Invalid integration, balance, current, or speed limits")
        if (round(self.episode_time / self.sample_time) < 1
                or round(self.balance_hold_time / self.sample_time) < 1):
            raise ValueError("Episode and balance time must each span at least one plant step")


@dataclass(frozen=True)
class Training:
    seed: int = 0
    decisions: int = 250_000
    validation_every: int = 25_000
    validation_episodes: int = 50
    validation_seed: int = 10_000
    actor_widths: tuple[int, ...] = (16, 16)
    critic_widths: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    learning_rate: float = 0.0003
    learning_starts: int = 40_000
    replay_capacity: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    train_frequency: int = 2
    gradient_steps: int = 1
    target_update_interval: int = 1
    entropy_coefficient: str = "auto_0.1"
    target_entropy: float = -1.0
    energy_weight: float = 0.25
    teacher_pretrain_updates: int = 500
    teacher_loss_weight: float = 5.0


@dataclass(frozen=True)
class Evaluation:
    # Run evaluate.py directly to score this model.
    model: Path = REFERENCE / "model.zip"
    episodes: int = 100
    base_seed: int = 80_000
    capture_starts: Path = REFERENCE / "capture_starts.csv"
    plot_episodes: int = 5


TASK = Task()
TRAINING = Training()
EVALUATION = Evaluation()


# Match the reference CPU math settings even when launched directly from VS Code.
CPU_THREADS = 1


def configure_runtime():
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
    os.environ.setdefault("MPLBACKEND", "Agg")
