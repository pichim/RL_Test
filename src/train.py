"""Run nominal periodic Furuta training from scratch using config.py."""

from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
import platform
import shutil
import sys
from time import perf_counter

from config import EVALUATION, RUNS, TASK, TRAINING, configure_runtime

configure_runtime()

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from furuta_env import FurutaSwingUpEnv
from guided_sac import (
    MONITOR_KEYS, EnergyShaping, LQRTeaching, PeriodicValidation,
    PeriodicValidationCallback, ReflectedPolicy, atomic_write_json,
    export_actor, lqr_observation_limits,
)


def train(directory, task=TASK, settings=TRAINING):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")
    probe = torch.ones((8, 8), device="cuda")
    if (probe @ probe)[0, 0].item() != 8.0:
        raise RuntimeError("CUDA preflight failed")
    del probe
    interval = min(settings.validation_every, settings.decisions)
    if (interval <= 0 or settings.train_frequency < 1 or settings.decisions % interval
            or settings.decisions % settings.train_frequency or settings.validation_episodes < 1
            or settings.teacher_pretrain_updates < 1
            or not 0 <= settings.seed < 2**32):
        raise ValueError("Choose a valid seed and an aligned training and validation budget")

    directory.mkdir(parents=True)
    print(f"Training on {torch.cuda.get_device_name()}\n{directory}", flush=True)
    atomic_write_json(directory / "config.json", asdict(task))
    atomic_write_json(directory / "training.json", {
        **asdict(settings),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
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
            tensorboard_log=str(directory / "tensorboard"), device="cuda", verbose=1,
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


def main():
    directory = RUNS / datetime.now().strftime("local_%Y%m%d_%H%M%S_%f")
    if not EVALUATION.capture_starts.is_file():
        raise FileNotFoundError(EVALUATION.capture_starts)
    start = perf_counter()
    model_path = train(directory, TASK, TRAINING)

    from evaluate import evaluate

    training_seconds = perf_counter() - start
    capture_snapshot = directory / "capture_starts.csv"
    shutil.copyfile(EVALUATION.capture_starts, capture_snapshot)
    result = evaluate(model_path, directory / "evaluation", EVALUATION, capture_starts=capture_snapshot)
    atomic_write_json(directory / "timing.json", {
        "training_seconds": training_seconds,
        "evaluation_seconds": perf_counter() - start - training_seconds,
    })
    print(result)


if __name__ == "__main__":
    main()
