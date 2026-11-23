"""Train one nominal direct-current SAC controller and evaluate its best model."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import torch

from capture_recovery import load_capture_starts, score_capture_starts
from evaluate import main as evaluate
from furuta_env import (
    DEFAULT_CONFIG, PERIODIC_LEGACY_REWARD_TASK_VERSION, PERIODIC_STEP_COST,
    PERIODIC_TASK_VERSION, PERIODIC_UNSAFE_PENALTY,
)
from guided_sac import PERIODIC_VALIDATION_INTERVAL, upright_benchmark, train_guided
from train import atomic_write_json, train


PERIODIC_GAMMA = 0.999
PERIODIC_SUCCESS_BONUS = 250.0

DEFAULT_JOB = Path(__file__).resolve().parent / "runs" / "nominal_sac_actor64_seed0_v0"
CONFIG = replace(
    DEFAULT_CONFIG,
    current_filter_cutoff_hz=0.0,
    current_filter_cutoff_randomization_hz=0.0,
    current_filter_cutoff_min_hz=0.0,
    current_filter_cutoff_max_hz=0.0,
    action_dead_time_max_samples=0,
    training_parameter_randomization=False,
    motor_torque_randomization=0.0,
    arm_mass_randomization=0.0,
    arm_inertia_randomization=0.0,
    pendulum_mass_randomization=0.0,
    pendulum_inertia_randomization=0.0,
    arm_damping_randomization=0.0,
    pendulum_damping_randomization=0.0,
)


def cuda_preflight() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use the CUDA-enabled rl-env interpreter "
            "and make sure the NVIDIA GPU is visible to this process."
        )
    sample = torch.ones((8, 8), device="cuda")
    if (sample @ sample)[0, 0].item() != 8.0:
        raise RuntimeError("CUDA preflight matrix multiplication returned an unexpected result.")
    return {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def run(
    job: Path = DEFAULT_JOB,
    execute: bool = False,
    action_change_weight: float = CONFIG.action_change_weight,
    *,
    task_config=None,
    require_cuda: bool = False,
    guided: bool = False,
    energy_weight: float | None = None,
    max_current: float | None = None,
    current_weight: float | None = None,
    timesteps: int | None = None,
    seed: int = 0,
    actor_network: list[int] | None = None,
    actor_observes_previous_command: bool = True,
    lqr_only_inside: bool = False,
    reset_speed_rad_s: float | None = None,
    full_arm_travel: bool = False,
    periodic_pilot: bool = False,
    scratch_pilot: str | None = None,
    comparison_model: Path | None = None,
    capture_benchmark: Path | None = None,
) -> None:
    if scratch_pilot not in (None, "local", "recovery") or (scratch_pilot is not None and periodic_pilot):
        raise ValueError("Choose a single periodic or scratch profile.")
    periodic_profile = periodic_pilot or scratch_pilot is not None
    if timesteps is None:
        timesteps = 250_000 if scratch_pilot else (500_000 if periodic_pilot else 2_000_000)
    if periodic_profile:
        guided = lqr_only_inside = full_arm_travel = True
        max_current = 1.0
        reset_speed_rad_s = 6.0
        current_weight, action_change_weight = 0.75, 3.75
        actor_network = [16, 16]
        if timesteps > PERIODIC_VALIDATION_INTERVAL and timesteps % PERIODIC_VALIDATION_INTERVAL:
            raise ValueError("The periodic pilot requires a multiple of 25,000 decisions.")
    if comparison_model is not None:
        if not periodic_profile:
            raise ValueError("A comparison model requires a periodic profile.")
        comparison_model = comparison_model.resolve()
        if not comparison_model.is_file():
            raise FileNotFoundError(comparison_model)
    if max_current is not None and (not guided or max_current not in (0.5, 1.0)):
        raise ValueError("Choose a guided current limit of 0.5 A or 1 A.")
    if energy_weight is not None and (
        not guided or not math.isfinite(energy_weight) or energy_weight < 0
    ):
        raise ValueError("Energy weight must be finite and nonnegative, and requires --guided.")
    if timesteps <= 0 or timesteps % 2 or not 0 <= seed < 2**32:
        raise ValueError("Choose a positive even timestep budget and a seed in [0, 2**32).")
    if actor_network is not None and (not actor_network or any(size <= 0 for size in actor_network)):
        raise ValueError("Every actor hidden layer must have a positive width.")
    if not guided and not actor_observes_previous_command:
        raise ValueError("The six-input actor requires --guided.")
    if not guided and lqr_only_inside:
        raise ValueError("The LQR-only region requires --guided.")
    if guided and task_config is not None:
        raise ValueError("Guided SAC is defined on the centered nominal task.")
    if guided and energy_weight is None:
        energy_weight = 0.25
    job = job.resolve()
    config = replace(
        CONFIG if task_config is None else task_config,
        action_change_weight=action_change_weight,
    )
    if full_arm_travel:
        config = replace(config, reset_theta1_half_range=math.pi, arm_angle_limit=math.pi)
    if periodic_pilot:
        config = replace(
            config, task_version=PERIODIC_TASK_VERSION, constraint_reward_weight=0.0,
            terminal_success_bonus=PERIODIC_SUCCESS_BONUS,
        )
    elif scratch_pilot:
        config = replace(config, task_version=PERIODIC_LEGACY_REWARD_TASK_VERSION)
    if reset_speed_rad_s is not None:
        config = replace(
            config, reset_omega1_half_range=reset_speed_rad_s,
            reset_omega2_half_range=reset_speed_rad_s,
            randomized_reset_omega1_half_range=reset_speed_rad_s,
            randomized_reset_omega2_half_range=reset_speed_rad_s,
        )
    if current_weight is not None:
        config = replace(config, current_weight=current_weight)
    if max_current is not None:
        # The reward uses normalized commands. Preserve the cost per ampere squared.
        cost_scale = (max_current / CONFIG.max_current) ** 2
        config = replace(
            config,
            max_current=max_current,
            current_weight=config.current_weight * cost_scale,
            action_change_weight=config.action_change_weight * cost_scale,
        )
    if capture_benchmark is not None:
        if not periodic_profile:
            raise ValueError("A capture benchmark requires a periodic profile.")
        capture_benchmark = capture_benchmark.resolve()
        capture_count = len(load_capture_starts(capture_benchmark, config))
        print(f"Loaded {capture_count} fixed capture states for final evaluation.", flush=True)
    settings = {
        "seed": seed,
        "total_timesteps": timesteps,
        "gamma": PERIODIC_GAMMA if periodic_pilot else 0.99,
        "actor_network": [64, 64] if actor_network is None else list(actor_network),
        "critic_network": [64, 64],
        "evaluation_episodes": 50,
    }
    arm_travel = (
        "Arm uses periodic orientation with no angle stop."
        if config.periodic_arm else f"Arm travel limit: ±{math.degrees(config.arm_angle_limit):g}°."
    )
    print(
        f"Nominal SAC: {settings}\n"
        f"Current limit: ±{config.max_current:g} A. "
        f"Normalized current and action-change weights: {config.current_weight:g}, {config.action_change_weight:g}.\n"
        f"Control period: {1000 * config.sample_time * config.action_repeat:g} ms.\n"
        f"{arm_travel}\n"
        f"Reset bounds: arm ±{math.degrees(config.reset_theta1_half_range):g}°, "
        f"pendulum ±{math.degrees(config.reset_theta2_half_range):g}°, "
        f"velocities ±{config.reset_omega1_half_range:g}/±{config.reset_omega2_half_range:g} rad/s.\n"
        f"Output: {job}",
        flush=True,
    )
    if job.exists():
        raise FileExistsError(f"Choose a fresh job directory: {job}")
    if guided:
        actor_inputs = 7 if actor_observes_previous_command else 6
        print(
            f"Guided SAC with {actor_inputs} actor inputs, reflected policy and critics, "
            f"LQR teaching, and energy weight {energy_weight:g}.",
            flush=True,
        )
    if lqr_only_inside:
        print("Inside the upright box, only LQR supervision directly updates the actor.", flush=True)
    if periodic_pilot:
        print(
            f"Validation every {min(timesteps, PERIODIC_VALIDATION_INTERVAL):,} decisions. "
            f"Reward adds a step cost of {PERIODIC_STEP_COST:g}, a success bonus of "
            f"{config.terminal_success_bonus:g}, and an unsafe penalty of {PERIODIC_UNSAFE_PENALTY:g}.",
            flush=True,
        )
    if scratch_pilot:
        print(
            f"Scratch pilot ({scratch_pilot}) with fresh networks and replay, "
            f"500 local teacher warm-up updates and validation every "
            f"{min(timesteps, PERIODIC_VALIDATION_INTERVAL):,} decisions. "
            "The prior reward uses periodic arm error.",
            flush=True,
        )
    cuda = cuda_preflight() if require_cuda or guided else None
    if cuda is not None:
        print(
            f"CUDA preflight passed on {cuda['device']} (PyTorch {cuda['torch_version']}).",
            flush=True,
        )
    if not execute:
        print("Dry run. Add --execute to train and evaluate.", flush=True)
        return
    job.mkdir(parents=True)
    phase = "training"
    timings: dict[str, float] = {}

    def status(state: str, **details) -> None:
        atomic_write_json(job / "status.json", {
            "state": state,
            "phase": phase,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            **timings,
            **details,
        })

    try:
        status("running")
        source = job / "source"
        source.mkdir()
        for path in Path(__file__).resolve().parent.glob("*.py"):
            shutil.copy2(path, source / path.name)
        atomic_write_json(job / "experiment.json", {
            "training": settings,
            "config": asdict(config),
            "cuda_preflight": cuda,
            "guided_energy_weight": energy_weight,
            "actor_observes_previous_command": actor_observes_previous_command,
            "lqr_only_inside": lqr_only_inside,
            "scratch_pilot": scratch_pilot,
            "capture_benchmark": str(capture_benchmark) if capture_benchmark else None,
            "comparison_model": str(comparison_model) if comparison_model else None,
            "periodic_reward": {
                "step_cost": PERIODIC_STEP_COST,
                "unsafe_penalty": PERIODIC_UNSAFE_PENALTY,
            } if periodic_pilot else None,
            "evaluation": {"controller": "sac", "mode": "nominal", "episodes": 100, "base_seed": 80000},
            "selection": "Fixed nominal training validation; no selection on evaluation seeds.",
        })
        if capture_benchmark is not None:
            benchmark_snapshot = job / "capture_starts.csv"
            shutil.copy2(capture_benchmark, benchmark_snapshot)
        start = perf_counter()
        if guided:
            train_guided(
                config, job / "training", energy_weight=energy_weight,
                actor_observes_previous_command=actor_observes_previous_command,
                lqr_only_inside=lqr_only_inside,
                teach_recovery=scratch_pilot == "recovery",
                **settings,
            )
        else:
            train(config=config, run_dir=job / "training", **settings)
        timings["training_seconds"] = perf_counter() - start
        phase = "evaluation"
        status("running")
        start = perf_counter()
        evaluate(
            model_path=job / "training" / "best" / "best_model.zip",
            results_dir=job / "evaluation",
            controller="sac",
            mode="nominal",
            episodes=100,
            base_seed=80000,
        )
        if periodic_profile:
            actor_path = job / "training" / "best" / "best_model.zip"
            local = {"trained_actor": upright_benchmark(actor_path, config)}
            if comparison_model is not None:
                evaluate(
                    model_path=comparison_model,
                    config_path=job / "training" / "config.json",
                    results_dir=job / "baseline_evaluation",
                    controller="sac", mode="nominal", episodes=100, base_seed=80000,
                )
                local["previous_actor"] = upright_benchmark(comparison_model, config)
            atomic_write_json(job / "upright_benchmark.json", local)
            if capture_benchmark is not None:
                starts = load_capture_starts(benchmark_snapshot, config)
                capture_scores = {"trained_actor": score_capture_starts(actor_path, config, starts)}
                if comparison_model is not None:
                    capture_scores["previous_actor"] = score_capture_starts(comparison_model, config, starts)
                atomic_write_json(job / "capture_benchmark.json", capture_scores)
        timings["evaluation_seconds"] = perf_counter() - start
        with (job / "evaluation" / "summary.csv").open(newline="", encoding="utf-8") as handle:
            summary = next(csv.DictReader(handle))
        report = (
            "# Nominal SAC Result\n\n"
            f"Current limit: ±{config.max_current:g} A. "
            f"Normalized current and action-change weights: {config.current_weight:g}, {config.action_change_weight:g}.\n\n"
            f"Successes: {summary['success_count']}/{summary['episodes']}. "
            f"Unsafe episodes: {summary['unsafe_count']}. "
            f"Captures: {summary['capture_count']}.\n\n"
            f"Training wall time: {timings['training_seconds']:.1f} s. "
            f"Final evaluation wall time: {timings['evaluation_seconds']:.1f} s.\n\n"
            + (
                "Fixed nominal plant and direct current. LQR supplied training targets. "
                "The standalone actor is saved to training/actor.pt2. "
                if guided else "Fixed nominal plant, direct current, no delay, no LQR. "
            )
            + "Final evaluation uses the learned neural actor. "
            "The policy is selected using training validation, not evaluation seeds.\n\n"
            "Success requires the final second within 5 degrees of upright, "
            f"{config.balance_theta1:g} rad arm error"
            f"{' modulo one turn' if config.periodic_arm else ''}, "
            f"{config.balance_omega1:g} rad/s arm speed, and {config.balance_omega2:g} rad/s pendulum speed. "
            "Initial states still vary across the existing broad reset distribution.\n\n"
            "See evaluation/summary.txt and evaluation/sac/training_nominal for metrics, PNGs, and traces. "
            "One seed and a five-second test do not establish long-duration or hardware reliability.\n"
        )
        if periodic_profile:
            report += (
                "\nSee training/validation_metrics.json for validation at each checkpoint "
                "and upright_benchmark.json for local recovery and LQR command errors.\n"
            )
            if comparison_model is not None:
                report += "The previous actor's matched results are in baseline_evaluation/summary.txt.\n"
        if capture_benchmark is not None:
            report += "See capture_benchmark.json for the fixed post-capture panel.\n"
        (job / "FINDINGS.md").write_text(report, encoding="utf-8")
        phase = "report"
        status("completed", success_count=int(summary["success_count"]), unsafe_count=int(summary["unsafe_count"]))
    except Exception:
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--current-weight", type=float,
        help="current-squared weight at the 0.5 A reference limit, before current-limit scaling",
    )
    parser.add_argument(
        "--action-change-weight", type=float, default=CONFIG.action_change_weight,
        help="current-change-squared weight at the 0.5 A reference limit, before current-limit scaling",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="verify CUDA before training and fail if it is unavailable",
    )
    parser.add_argument("--guided", action="store_true", help="train reflected SAC with centered LQR targets on CUDA")
    parser.add_argument("--energy-weight", type=float, help="guided potential weight (default: 0.25; 0 for the control run)")
    parser.add_argument(
        "--max-current", type=float, choices=(0.5, 1.0), metavar="AMPS",
        help="guided current limit in A; preserve the baseline physical current penalties",
    )
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--periodic-pilot", action="store_true",
                        help="run the 16×16 periodic-arm pilot with boundary LQR teaching")
    parser.add_argument("--scratch-pilot", choices=("local", "recovery"),
                        help="train a fresh periodic actor with local LQR or added recovery demonstrations")
    parser.add_argument("--capture-benchmark", type=Path,
                        help="fixed capture-start CSV used at final evaluation")
    parser.add_argument("--comparison-model", type=Path,
                        help="score an earlier actor on the same periodic tests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-hidden-sizes", type=int, nargs="+", metavar="N", help="actor hidden widths (default: 64 64)")
    parser.add_argument(
        "--actor-no-previous-command", action="store_true",
        help="use six plant inputs in the guided actor and keep command history in the critics",
    )
    parser.add_argument(
        "--lqr-only-inside", action="store_true",
        help="train the guided actor from LQR targets only inside its upright box",
    )
    parser.add_argument(
        "--reset-speed-rad-s", type=float, metavar="RAD_PER_S",
        help="symmetric starting speed half-range applied to both joints",
    )
    parser.add_argument(
        "--full-arm-travel", action="store_true",
        help="set both the simulated motor arm reset range and its travel limit to ±π",
    )
    arguments = parser.parse_args()
    run(
        arguments.job_dir,
        arguments.execute,
        arguments.action_change_weight,
        require_cuda=arguments.require_cuda,
        guided=arguments.guided,
        energy_weight=arguments.energy_weight,
        max_current=arguments.max_current,
        current_weight=arguments.current_weight,
        timesteps=arguments.timesteps,
        seed=arguments.seed,
        actor_network=arguments.actor_hidden_sizes,
        actor_observes_previous_command=not arguments.actor_no_previous_command,
        lqr_only_inside=arguments.lqr_only_inside,
        reset_speed_rad_s=arguments.reset_speed_rad_s,
        full_arm_travel=arguments.full_arm_travel,
        periodic_pilot=arguments.periodic_pilot,
        scratch_pilot=arguments.scratch_pilot,
        comparison_model=arguments.comparison_model,
        capture_benchmark=arguments.capture_benchmark,
    )
