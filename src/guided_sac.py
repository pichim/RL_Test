"""Nominal SAC with exact reflection, energy shaping, and LQR supervision."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from itertools import product
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, SACPolicy

from capture_recovery import RECOVERY_SEED_STREAM, recovery_examples
from controllers import LQRController
from furuta_env import (
    FurutaSwingUpEnv, arm_position_error, mechanical_capture_ready, observation_from_plant, upright_error,
)
from furuta_model import FurutaPendulum
from train import ResumeStateCallback, SuccessFirstEvalCallback, algorithm_settings, atomic_write_json
from train import runtime_provenance, save_committed_state


# Flip sin(angle), velocity, and previous command. Cosines are even.
REFLECTION = (-1, 1, -1, -1, 1, -1, -1)
TEACHER_BOUNDS = np.array([0.10, 0.05, 0.25, 0.50])
TEACHER_BOUNDS_1A = np.array([0.20, 0.10, 0.50, 1.00])
TEACHER_BATCH = 128
RECOVERY_BATCH = 128
PRETRAIN_UPDATES = 500
PERIODIC_PRETRAIN_UPDATES = 4_000
TEACHER_LOSS_WEIGHT = 5.0
SLOPE_FRACTION = 0.10
NEAR_ORIGIN_SLOPE_FRACTION = 0.002
LOCAL_EXPLORATION_STD = 0.02
PERIODIC_VALIDATION_INTERVAL = 25_000
BOUNDARY_PREVIOUS_COMMANDS = (-1.0, -0.5, 0.0, 0.5, 1.0)
MONITOR_KEYS = ("is_success", "initial_capture", "unsafe")


class ReflectedActor(Actor):
    def __init__(self, *args, lqr_observation_limits=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde:
            raise ValueError("The reflected actor uses a squashed Gaussian without gSDE.")
        self.register_buffer("reflection", torch.tensor(REFLECTION, dtype=torch.float32))
        limits = None if lqr_observation_limits is None else torch.tensor(
            lqr_observation_limits, dtype=torch.float32,
        )
        self.register_buffer("lqr_observation_limits", limits, persistent=False)

    def inside_lqr_region(self, obs):
        arm_cos, pendulum_cos, arm_speed, pendulum_speed = self.lqr_observation_limits
        return (
            (obs[..., 1] >= arm_cos)
            & (obs[..., 4] >= pendulum_cos)
            & (obs[..., 2].abs() <= arm_speed)
            & (obs[..., 5].abs() <= pendulum_speed)
        )

    def get_action_dist_params(self, obs):
        pair = torch.cat((obs, obs * self.reflection))
        mean, log_std, kwargs = super().get_action_dist_params(pair)
        direct_mean, mirror_mean = mean.chunk(2)
        direct_scale, mirror_scale = log_std.chunk(2)
        scale = (direct_scale + mirror_scale) * 0.5
        if self.lqr_observation_limits is not None:
            scale = torch.where(
                self.inside_lqr_region(obs).unsqueeze(-1), math.log(LOCAL_EXPLORATION_STD), scale,
            )
        return (direct_mean - mirror_mean) * 0.5, scale, kwargs

    def action_log_prob(self, obs):
        action, log_prob = super().action_log_prob(obs)
        if self.lqr_observation_limits is not None and torch.is_grad_enabled():
            local = self.inside_lqr_region(obs)
            # Only the teacher updates the actor directly on local states.
            # Preserve forward values for SAC's critic and entropy calculations.
            action = torch.where(local.unsqueeze(-1), action.detach(), action)
            log_prob = torch.where(local, log_prob.detach(), log_prob)
        return action, log_prob


class ReflectedCritic(ContinuousCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("reflection", torch.tensor(REFLECTION, dtype=torch.float32))

    def forward(self, obs, actions):
        pairs = super().forward(
            torch.cat((obs, obs * self.reflection)), torch.cat((actions, -actions))
        )
        return tuple((direct + mirror) * 0.5 for direct, mirror in (q.chunk(2) for q in pairs))

    def q1_forward(self, obs, actions):
        pair = super().q1_forward(
            torch.cat((obs, obs * self.reflection)), torch.cat((actions, -actions))
        )
        direct, mirror = pair.chunk(2)
        return (direct + mirror) * 0.5


class PlantObservationFeatures(BaseFeaturesExtractor):
    """Give the actor the six plant features from the shared observation."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=len(REFLECTION) - 1)

    def forward(self, observations):
        return observations[..., :self.features_dim]


class ReflectedPolicy(SACPolicy):
    def __init__(
        self, *args, actor_observes_previous_command=True, lqr_observation_limits=None, **kwargs
    ):
        self.actor_observes_previous_command = actor_observes_previous_command
        self.lqr_observation_limits = lqr_observation_limits
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self):
        parameters = super()._get_constructor_parameters()
        parameters["actor_observes_previous_command"] = self.actor_observes_previous_command
        parameters["lqr_observation_limits"] = self.lqr_observation_limits
        return parameters

    def make_actor(self, features_extractor=None):
        if not self.actor_observes_previous_command:
            features_extractor = PlantObservationFeatures(self.observation_space)
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ReflectedActor(**kwargs, lqr_observation_limits=self.lqr_observation_limits).to(self.device)

    def make_critic(self, features_extractor=None):
        if not self.actor_observes_previous_command and features_extractor is not None:
            raise ValueError("The six-input actor and the critics require separate feature extractors.")
        kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ReflectedCritic(**kwargs).to(self.device)


class EnergyShaping(gym.Wrapper):
    def __init__(self, env: FurutaSwingUpEnv, gamma: float, weight: float):
        super().__init__(env)
        self.gamma = gamma
        self.weight = weight
        self.previous_potential = 0.0

    def potential(self, info: dict) -> float:
        squared_error = info["energy_error"] ** 2
        return -self.weight * squared_error / (1.0 + squared_error)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential = self.potential(info)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # SB3 bootstraps through timeouts and uses zero future value at termination.
        successor = 0.0 if terminated else self.potential(info)
        shaping = self.gamma * successor - self.previous_potential
        self.previous_potential = successor
        return observation, reward + shaping, terminated, truncated, info


def teacher_bounds(config):
    return TEACHER_BOUNDS_1A if config.max_current == 1.0 else TEACHER_BOUNDS


def lqr_observation_limits(config):
    arm, pendulum, arm_speed, pendulum_speed = teacher_bounds(config)
    # The sine/cosine observation encodes angles modulo a full turn.
    return (
        math.cos(arm), math.cos(pendulum),
        arm_speed / config.max_abs_omega1, pendulum_speed / config.max_abs_omega2,
    )


def teacher_examples(config, seed, *, previous_command_half_range=0.25):
    """Supervise on near-upright states and their held-current recovery trajectories."""
    rng = np.random.default_rng(seed)
    plant = FurutaPendulum(config.sample_time)
    teacher = LQRController(config.max_current)
    observations, commands = [], []
    bounds = teacher_bounds(config)

    for error in rng.uniform(-bounds, bounds, size=(192, 4)):
        plant.reset(error + [0.0, np.pi, 0.0, 0.0])
        for _ in range(24):
            state = plant.x.copy()
            state[1] = upright_error(state)
            if np.any(np.abs(state) > bounds):
                break
            action = teacher.current(plant.x) / config.max_current
            # The memoryless teacher's target is valid across previous commands.
            previous = float(rng.uniform(-previous_command_half_range, previous_command_half_range))
            observations.append(observation_from_plant(plant, config, previous))
            commands.append([action])
            for _ in range(config.action_repeat):
                plant.step(action * config.max_current)

    # The policy architecture supplies the exact mirrored examples.
    return np.asarray(observations), np.asarray(commands, dtype=np.float32)


def teacher_boundary_examples(config):
    """Label the faces, edges and corners at common previous commands."""
    plant = FurutaPendulum(config.sample_time)
    teacher = LQRController(config.max_current)
    bounds = teacher_bounds(config)
    observations, targets = [], []
    for point in product((-1.0, 0.0, 1.0), repeat=4):
        if not any(point):
            continue
        plant.reset(bounds * point + [0.0, np.pi, 0.0, 0.0])
        target = teacher.current(plant.x) / config.max_current
        for previous in BOUNDARY_PREVIOUS_COMMANDS:
            observations.append(observation_from_plant(plant, config, previous))
            targets.append([target])
    return np.asarray(observations), np.asarray(targets, dtype=np.float32)


class LQRTeaching(BaseCallback):
    """Add local LQR supervision to each actor update made by stock SAC."""

    def __init__(self, config, seed, *, lqr_only_inside=False, recovery_data=None):
        super().__init__()
        self.slope_fractions = [SLOPE_FRACTION]
        if lqr_only_inside:
            self.slope_fractions.append(NEAR_ORIGIN_SLOPE_FRACTION)
        self.seed = seed
        self.pretrain_updates = PERIODIC_PRETRAIN_UPDATES if config.periodic_time_cost else PRETRAIN_UPDATES
        self.recovery_data = recovery_data
        self.observations, self.targets = teacher_examples(
            config, seed, previous_command_half_range=1.0 if lqr_only_inside else 0.25,
        )
        plant = FurutaPendulum(config.sample_time)
        teacher = LQRController(config.max_current)
        upright = np.array([0.0, np.pi, 0.0, 0.0])
        self.boundary_examples = teacher_boundary_examples(config) if config.periodic_time_cost else None
        self.slope_observations, self.slope_targets = [], []
        for fraction in self.slope_fractions:
            for offset in np.diag(np.append(TEACHER_BOUNDS, 0.25)) * fraction:
                plant.reset(upright + offset[:4])
                self.slope_observations.append(observation_from_plant(plant, config, offset[4]))
                self.slope_targets.append([teacher.current(plant.x) / config.max_current])

    def loss(self) -> torch.Tensor:
        index = torch.randint(
            len(self.observations), (TEACHER_BATCH,),
            generator=self.rng, device=self.model.device,
        )
        observations = self.teacher_obs[index]
        targets = self.teacher_actions[index]
        if self.boundary_examples is not None:
            count = len(BOUNDARY_PREVIOUS_COMMANDS)
            states = torch.randint(
                len(self.boundary_obs) // count, (TEACHER_BATCH // count,),
                generator=self.rng, device=self.model.device,
            )
            indices = (states[:, None] * count + self.boundary_commands).flatten()
            observations = torch.cat((observations, self.boundary_obs[indices]))
            targets = torch.cat((targets, self.boundary_actions[indices]))
        actions = self.model.actor(torch.cat((observations, self.slope_obs)), deterministic=True)
        predicted, slope = actions.split((len(targets), len(self.slope_observations)))
        local_mse = torch.nn.functional.mse_loss(predicted, targets)
        slope_errors = (slope - self.slope_actions).view(len(self.slope_fractions), -1)
        slope_mse = (slope_errors.square().mean(dim=1) / self.slope_fractions_squared).mean()
        return local_mse + slope_mse

    def _on_training_start(self):
        device = self.model.device
        self.rng = torch.Generator(device=device).manual_seed(self.seed)
        self.teacher_obs = torch.as_tensor(self.observations, device=device)
        self.teacher_actions = torch.as_tensor(self.targets, device=device)
        if self.boundary_examples is not None:
            obs, actions = self.boundary_examples
            self.boundary_obs = torch.as_tensor(obs, device=device)
            self.boundary_actions = torch.as_tensor(actions, device=device)
            self.boundary_commands = torch.arange(len(BOUNDARY_PREVIOUS_COMMANDS), device=device)
        if self.recovery_data is not None:
            observations, targets = self.recovery_data
            self.recovery_obs = torch.as_tensor(observations, device=device)
            self.recovery_actions = torch.as_tensor(targets, device=device)
            self.recovery_rng = torch.Generator(device=device).manual_seed(self.seed + RECOVERY_SEED_STREAM)
        self.slope_obs = torch.as_tensor(np.asarray(self.slope_observations), device=device)
        self.slope_actions = torch.as_tensor(self.slope_targets, dtype=torch.float32, device=device)
        self.slope_fractions_squared = torch.tensor(
            self.slope_fractions, dtype=torch.float32, device=device,
        ).square()
        optimizer = self.model.actor.optimizer
        for _ in range(self.pretrain_updates):
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss()
            loss.backward()
            optimizer.step()
        self.logger.record("teacher/pretrain_loss", loss.item())
        self.hook = optimizer.register_step_pre_hook(self.add_teacher_gradient)

    def add_teacher_gradient(self, optimizer, args, kwargs):
        parameters = [p for p in self.model.actor.parameters() if p.requires_grad]
        with torch.enable_grad():
            local_loss = self.loss()
            loss = local_loss
            if self.recovery_data is not None:
                indices = torch.randint(
                    len(self.recovery_obs), (RECOVERY_BATCH,), device=self.model.device,
                    generator=self.recovery_rng,
                )
                prediction = self.model.actor(self.recovery_obs[indices], deterministic=True)
                recovery_loss = torch.nn.functional.mse_loss(prediction, self.recovery_actions[indices])
                loss = loss + recovery_loss
                if self.model.num_timesteps % 64 == 0:
                    self.logger.record("teacher/recovery_loss", recovery_loss.detach().item())
            gradients = torch.autograd.grad(
                TEACHER_LOSS_WEIGHT * loss, parameters, allow_unused=True,
            )
        for param, grad in zip(parameters, gradients):
            if grad is not None:
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad.add_(grad)
        if self.model.num_timesteps % 64 == 0:
            self.logger.record("teacher/local_loss", local_loss.detach().item())

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self):
        self.hook.remove()


class DeterministicActor(torch.nn.Module):
    """The saved controller needs only the learned deterministic actor layers."""

    def __init__(self, actor: ReflectedActor):
        super().__init__()
        self.latent = deepcopy(actor.latent_pi)
        self.mean = deepcopy(actor.mu)
        feature_count = actor.features_extractor.features_dim
        self.register_buffer("reflection", actor.reflection[:feature_count].clone())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        pair = torch.stack((observation, observation * self.reflection))
        direct, mirror = self.mean(self.latent(pair)).unbind(0)
        return torch.tanh((direct - mirror) * 0.5)


def local_diagnostics(controller: DeterministicActor, config) -> dict:
    """Finite differences of the exported actor and its nonlinear held-current map."""
    plant = FurutaPendulum(config.sample_time)
    upright = np.array([0.0, np.pi, 0.0, 0.0])
    feature_count = controller.reflection.numel()
    uses_history = feature_count == len(REFLECTION)
    state_order = ["arm_angle", "upright_error", "arm_velocity", "pendulum_velocity"]
    teacher_gain = (-LQRController(config.max_current).gain).tolist()
    if uses_history:
        state_order.append("previous_normalized_action")
        teacher_gain.append(0.0)
    state_size = len(state_order)

    def transition(point):
        plant.reset(upright + point[:4])
        previous_command = point[4] if uses_history else 0.0
        observation = observation_from_plant(plant, config, previous_command)[:feature_count]
        with torch.no_grad():
            action = controller(torch.from_numpy(observation).unsqueeze(0)).item()
        for _ in range(config.action_repeat):
            plant.step(config.max_current * action)
        state = plant.x.copy()
        state[1] = upright_error(state)
        return np.append(state, action)

    step = 1e-4
    axes = np.eye(state_size) * step
    jacobian = np.column_stack([
        (transition(axis) - transition(-axis)) / (2 * step) for axis in axes
    ])
    poles = np.linalg.eigvals(jacobian[:state_size])
    return {
        "state_order": state_order,
        "equilibrium_current_A": config.max_current * transition(np.zeros(state_size))[-1],
        "actor_current_jacobian": (config.max_current * jacobian[-1]).tolist(),
        "teacher_current_jacobian": teacher_gain,
        "closed_loop_poles": [{"real": float(p.real), "imag": float(p.imag)} for p in poles],
        "spectral_radius": float(np.max(np.abs(poles))),
        "finite_difference_step": step,
    }


def export_actor(model_path: Path, output_path: Path, config) -> dict:
    actor = SAC.load(model_path, device="cpu").actor
    controller = DeterministicActor(actor).eval()
    with torch.no_grad():
        exported = torch.export.export(
            controller, (torch.zeros(2, controller.reflection.numel()),),
            dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
        )
    torch.export.save(exported, output_path)
    return local_diagnostics(controller, config)


def upright_benchmark(model_path, config):
    """Measure held-out LQR command agreement and nonlinear local recovery."""
    model = SAC.load(model_path, device="cpu")
    teacher = LQRController(config.max_current)
    bounds = teacher_bounds(config)
    upright = np.array([0.0, np.pi, 0.0, 0.0])
    recovered_actor = recovered_teacher = 0
    unsafe_actor = unsafe_teacher = 0
    env = FurutaSwingUpEnv(config)
    try:
        for i, point in enumerate(product((-0.85, 0.0, 0.85), repeat=4)):
            initial = upright + bounds * point
            previous = (-0.75, 0.0, 0.75)[i % 3]
            for use_teacher in (False, True):
                _, info = env.reset(options={"initial_state": initial})
                env.last_normalized_action = previous
                observation = observation_from_plant(env.plant, config, previous)
                done = False
                while not done:
                    if use_teacher:
                        action = [teacher.current(info["state"]) / config.max_current]
                    else:
                        action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                if use_teacher:
                    recovered_teacher += int(info["is_success"])
                    unsafe_teacher += int(info["unsafe"])
                else:
                    recovered_actor += int(info["is_success"])
                    unsafe_actor += int(info["unsafe"])

        observations = []
        targets = []
        for point in product((-1.0, -0.6, 0.0, 0.6, 1.0), repeat=4):
            env.plant.reset(upright + bounds * point)
            for previous in (-0.75, -0.25, 0.25, 0.75):
                observations.append(observation_from_plant(env.plant, config, previous))
                targets.append(teacher.current(env.plant.x))
        action, _ = model.predict(np.asarray(observations), deterministic=True)
        command_errors = np.abs(action.ravel() * config.max_current - targets)
    finally:
        env.close()
    return {
        "recovery_starts": 81,
        "actor_recovered": recovered_actor,
        "actor_unsafe": unsafe_actor,
        "lqr_recovered": recovered_teacher,
        "lqr_unsafe": unsafe_teacher,
        "current_probes": len(command_errors),
        "mean_abs_current_error_A": float(np.mean(command_errors)),
        "p95_abs_current_error_A": float(np.quantile(command_errors, 0.95)),
        "max_abs_current_error_A": float(np.max(command_errors)),
    }


class PeriodicValidation(gym.Wrapper):
    """Collect per-episode validation metrics on the existing rollouts."""

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.return_discounted = 0.0
        self.discount = 1.0
        self.captured = False
        self.entered_teacher_box = False
        self._track_regions(info["state"])
        self.initial_capture = self.captured
        return observation, info

    def _track_regions(self, state):
        self.captured |= mechanical_capture_ready(state, self.env.config)
        error = np.array([state[0], upright_error(state), state[2], state[3]])
        error[0] = arm_position_error(error[0], 0.0, self.env.config)
        self.entered_teacher_box |= bool(np.all(np.abs(error) <= teacher_bounds(self.env.config)))

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._track_regions(info["state"])
        self.return_discounted += self.discount * reward
        self.discount *= self.gamma
        if terminated or truncated:
            info["validation_metrics"] = {
                "balanced": info["is_success"],
                "captured": self.captured,
                "genuine_swingup": info["is_success"] and not self.initial_capture,
                "entered_teacher_box": self.entered_teacher_box,
                "balanced_after_teacher_entry": info["is_success"] and self.entered_teacher_box,
                "unsafe": info["unsafe"],
                "timeout": truncated,
                "discounted_reward": self.return_discounted,
            }
        return observation, reward, terminated, truncated, info


class PeriodicValidationCallback(SuccessFirstEvalCallback):
    def __init__(self, *args, metrics_path, rank_recovery=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_path = metrics_path
        self.rank_recovery = rank_recovery
        self.episodes = []
        self.history = []

    def validation_score(self, success_rate):
        if self.rank_recovery:
            recovered = sum(ep["balanced_after_teacher_entry"] for ep in self.episodes)
            return success_rate, recovered / len(self.episodes), self.last_mean_reward
        return super().validation_score(success_rate)

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            self.episodes.append(locals_["info"]["validation_metrics"])

    def _on_step(self):
        proceed = super()._on_step()
        if self.episodes:
            outcomes = (
                "balanced", "captured", "genuine_swingup", "entered_teacher_box",
                "balanced_after_teacher_entry", "unsafe", "timeout",
            )
            self.history.append({
                "timesteps": int(self.num_timesteps),
                "episodes": len(self.episodes),
                **{name: sum(int(ep[name]) for ep in self.episodes) for name in outcomes},
                "mean_discounted_reward": float(np.mean([ep["discounted_reward"] for ep in self.episodes])),
            })
            self.episodes.clear()
            atomic_write_json(self.metrics_path, {"checkpoints": self.history})
        return proceed


def train_guided(
    config, run_dir, *, energy_weight, actor_observes_previous_command=True,
    lqr_only_inside=False, teach_recovery=False, **options
):
    """Train the fresh guided variant with unshaped validation and stock SAC."""
    settings = algorithm_settings(config, **options)
    if config.periodic_arm:
        interval = min(PERIODIC_VALIDATION_INTERVAL, settings["total_timesteps"])
        if settings["total_timesteps"] % interval:
            raise ValueError("Periodic pilot budget must be a multiple of 25,000 above 25,000 decisions.")
        settings["evaluation_frequency"] = interval
        settings["checkpoint_frequency"] = interval
        if not config.periodic_time_cost:
            settings["best_model_selection"] = "success_rate_then_recovery_after_box_entry_then_mean_reward"
    recovery_data = None
    recovery_metadata = None
    if teach_recovery:
        if not config.periodic_arm or not lqr_only_inside:
            raise ValueError("Recovery demonstrations require the periodic LQR-guided setup.")
        observations, commands, recovery_metadata = recovery_examples(
            config, settings["seed"], teacher_bounds(config),
        )
        recovery_data = observations, commands
        recovery_metadata["batch_size"] = RECOVERY_BATCH
        print(
            f"LQR recovered {recovery_metadata['recovered_starts_used']}/"
            f"{recovery_metadata['candidate_starts']} sampled starts. "
            f"Using {len(commands)} approach observations during SAC updates.",
            flush=True,
        )
    settings["guided"] = {
        "energy_weight": energy_weight,
        "reflection": list(REFLECTION),
        "actor_observes_previous_command": actor_observes_previous_command,
        "actor_input_count": len(REFLECTION) - int(not actor_observes_previous_command),
        "teacher": "centered held-current LQR",
        "teacher_gain_ampere": LQRController(config.max_current).gain.tolist(),
        "teacher_bounds": teacher_bounds(config).tolist(),
        "slope_state_bounds": TEACHER_BOUNDS.tolist(),
        "pretrain_updates": PERIODIC_PRETRAIN_UPDATES if config.periodic_time_cost else PRETRAIN_UPDATES,
        "teacher_loss_weight": TEACHER_LOSS_WEIGHT,
        "slope_fraction": SLOPE_FRACTION,
        "lqr_only_inside": lqr_only_inside,
        "near_origin_slope_fraction": NEAR_ORIGIN_SLOPE_FRACTION if lqr_only_inside else None,
        "local_exploration_std": LOCAL_EXPLORATION_STD if lqr_only_inside else None,
        "teacher_previous_command_half_range": 1.0 if lqr_only_inside else 0.25,
        "boundary_previous_commands": list(BOUNDARY_PREVIOUS_COMMANDS) if config.periodic_time_cost else None,
        "boundary_batch_size": (
            TEACHER_BATCH // len(BOUNDARY_PREVIOUS_COMMANDS) * len(BOUNDARY_PREVIOUS_COMMANDS)
            if config.periodic_time_cost else 0
        ),
        "recovery_teaching": recovery_metadata,
    }
    settings["provenance"] = runtime_provenance()
    run_dir.mkdir()
    atomic_write_json(run_dir / "config.json", asdict(config))
    atomic_write_json(run_dir / "training.json", settings)

    training = Monitor(
        EnergyShaping(FurutaSwingUpEnv(config), settings["gamma"], energy_weight),
        filename=str(run_dir / "training"), info_keywords=MONITOR_KEYS,
    )
    validation_env = FurutaSwingUpEnv(config)
    if config.periodic_arm:
        validation_env = PeriodicValidation(validation_env, settings["gamma"])
    validation = Monitor(validation_env, info_keywords=MONITOR_KEYS)
    evaluation_callback = PeriodicValidationCallback if config.periodic_arm else SuccessFirstEvalCallback
    metrics_option = {}
    if config.periodic_arm:
        metrics_option = {
            "metrics_path": run_dir / "validation_metrics.json",
            "rank_recovery": not config.periodic_time_cost,
        }
    try:
        model = SAC(
            ReflectedPolicy, training, seed=settings["seed"],
            learning_rate=settings["learning_rate"], buffer_size=settings["buffer_size"],
            learning_starts=settings["learning_starts"], batch_size=settings["batch_size"],
            gamma=settings["gamma"], tau=settings["tau"], train_freq=settings["train_frequency"],
            gradient_steps=settings["gradient_steps"], target_update_interval=settings["target_update_interval"],
            ent_coef=settings["entropy_coefficient"], target_entropy=settings["target_entropy"],
            policy_kwargs={
                "net_arch": {"pi": settings["actor_network"], "qf": settings["critic_network"]},
                "actor_observes_previous_command": actor_observes_previous_command,
                "lqr_observation_limits": lqr_observation_limits(config) if lqr_only_inside else None,
            },
            tensorboard_log=str(run_dir / "tensorboard"), device="cuda", verbose=1,
        )
        if model.device.type != "cuda":
            raise RuntimeError("Guided SAC requires CUDA.")
        callbacks = [
            LQRTeaching(
                config, settings["seed"], lqr_only_inside=lqr_only_inside, recovery_data=recovery_data,
            ),
            CheckpointCallback(settings["checkpoint_frequency"], str(run_dir / "checkpoints"), name_prefix="sac"),
            evaluation_callback(
                validation, reset_seed=settings["evaluation_seed"], best_model_save_path=str(run_dir / "best"),
                log_path=str(run_dir / "evaluation"), eval_freq=settings["evaluation_frequency"],
                n_eval_episodes=settings["evaluation_episodes"], deterministic=True, warn=False,
                **metrics_option,
            ),
            ResumeStateCallback(settings["resume_snapshot_frequency"], run_dir / "resume", verbose=1),
        ]
        model.learn(settings["total_timesteps"], callback=callbacks, progress_bar=True)
        save_committed_state(model, run_dir / "final")
        settings["final_num_timesteps"] = int(model.num_timesteps)
        settings["final_replay_buffer_size"] = int(model.replay_buffer.size())
        settings["device"] = str(model.device)
        local = export_actor(run_dir / "best/best_model.zip", run_dir / "actor.pt2", config)
        atomic_write_json(run_dir / "local_diagnostics.json", local)
        atomic_write_json(run_dir / "training.json", settings)
    finally:
        training.close()
        validation.close()
