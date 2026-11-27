"""Train a 200 Hz Furuta controller and evaluate it at 200 and 500 Hz."""

from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from math import isclose
import platform
import shutil
import sys
from time import perf_counter

from config import EVALUATION, RUNS, TASK, TRAINING, configure_runtime

configure_runtime()

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from furuta_env import FurutaSwingUpEnv
from guided_sac import (
    MONITOR_KEYS, EnergyShaping, LQRTeaching, PeriodicValidation,
    PeriodicValidationCallback, ReflectedPolicy, atomic_write_json,
    export_actor, lqr_observation_limits,
)


def train(directory, task=TASK, settings=TRAINING):
    if not isclose(task.control_period, 0.005):
        raise ValueError("Training requires a 5 ms control period (200 Hz)")
    interval = min(settings.validation_every, settings.decisions)
    if (interval <= 0 or settings.train_frequency < 1 or settings.decisions % interval
            or settings.decisions % settings.train_frequency or settings.validation_episodes < 1
            or settings.teacher_pretrain_updates < 1
            or not 0 <= settings.seed < 2**32):
        raise ValueError("Choose a valid seed and an aligned training and validation budget")

    directory.mkdir(parents=True)
    print(f"Training on CPU\n{directory}", flush=True)
    atomic_write_json(directory / "config.json", asdict(task))
    atomic_write_json(directory / "training.json", {
        **asdict(settings),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": "cpu",
        "packages": {name: version(name) for name in ("numpy", "torch", "gymnasium", "stable-baselines3")},
    })
    training = Monitor(
        EnergyShaping(FurutaSwingUpEnv(task), settings.gamma, settings.energy_weight),
        filename=str(directory / "training"), info_keywords=MONITOR_KEYS,
    )
    validation = Monitor(
        PeriodicValidation(FurutaSwingUpEnv(task), settings.gamma), info_keywords=MONITOR_KEYS,
    )
    try:
        model = SAC(
            ReflectedPolicy, training, seed=settings.seed,
            learning_rate=settings.learning_rate, buffer_size=settings.replay_capacity,
            learning_starts=settings.learning_starts, batch_size=settings.batch_size,
            gamma=settings.gamma, tau=settings.tau, train_freq=settings.train_frequency,
            gradient_steps=settings.gradient_steps, target_update_interval=settings.target_update_interval,
            ent_coef=settings.entropy_coefficient, target_entropy=settings.target_entropy,
            policy_kwargs={
                "net_arch": {"pi": list(settings.actor_widths), "qf": list(settings.critic_widths)},
                "actor_observes_previous_command": True,
                "lqr_observation_limits": lqr_observation_limits(task),
            },
            tensorboard_log=str(directory / "tensorboard"), device="cpu", verbose=1,
        )
        validator = PeriodicValidationCallback(validation, directory, settings)
        model.learn(
            settings.decisions, callback=[LQRTeaching(task, settings), validator], progress_bar=True,
        )
    finally:
        training.close()
        validation.close()
    model_path = directory / "best" / "best_model.zip"
    atomic_write_json(
        directory / "local_diagnostics.json", export_actor(model_path, directory / "actor.pt2", task),
    )
    return model_path


def main(settings=TRAINING):
    """Train, export, and evaluate one controller, then return its checkpoint path."""
    from evaluate import evaluate_rates

    capture_starts = EVALUATION.capture_starts
    if not capture_starts.is_file():
        raise FileNotFoundError(capture_starts)
    directory = RUNS / datetime.now().strftime("local_%Y%m%d_%H%M%S_%f")
    start = perf_counter()
    model_path = train(directory, TASK, settings)
    evaluation_start = perf_counter()
    capture_snapshot = directory / "capture_starts.csv"
    shutil.copyfile(capture_starts, capture_snapshot)
    evaluate_rates(model_path, directory, EVALUATION, capture_starts=capture_snapshot)
    atomic_write_json(directory / "timing.json", {
        "training_seconds": evaluation_start - start,
        "evaluation_seconds": perf_counter() - evaluation_start,
    })
    return model_path


if __name__ == "__main__":
    ArgumentParser(description=__doc__).parse_args()
    main()
