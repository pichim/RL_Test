"""Symmetric SAC and centered, held-current LQR teaching."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import Actor, SACPolicy

from furuta_env import (
    FurutaSwingUpEnv, arm_position_error, mechanical_capture_ready, observation_from_plant, upright_error,
)
from furuta_model import FurutaPendulum

REFLECTION = (-1, 1, -1, -1, 1, -1, -1)
LQR_GAIN = np.array([-0.4472, 3.4376, -0.1555, 0.3031], dtype=np.float64)
SLOPE_BOUNDS = (0.10, 0.05, 0.25, 0.50)
TEACHER_BATCH = 128
LOCAL_EXPLORATION_STD = 0.02
MONITOR_KEYS = ("is_success", "initial_capture", "unsafe")


def atomic_write_json(path, value):
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class LQRController:
    """Centered, saturated teacher held at the policy period."""

    def __init__(self, max_current):
        self.max_current = max_current
        self.gain = LQR_GAIN

    def current(self, state):
        theta1, _, omega1, omega2 = state
        if not -np.pi <= theta1 < np.pi:
            theta1 = arm_position_error(theta1, 0.0)
        error_state = np.array([theta1, upright_error(state), omega1, omega2], dtype=np.float64)
        return float(np.clip(-self.gain @ error_state, -self.max_current, self.max_current))


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


class ReflectedPolicy(SACPolicy):
    def __init__(
        self, *args, actor_observes_previous_command=True, lqr_observation_limits=None, **kwargs
    ):
        # The saved reference's SB3 policy kwargs include this value.
        if not actor_observes_previous_command:
            raise ValueError("The nominal controller requires seven observations")
        self.lqr_observation_limits = lqr_observation_limits
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self):
        parameters = super()._get_constructor_parameters()
        parameters["actor_observes_previous_command"] = True
        parameters["lqr_observation_limits"] = self.lqr_observation_limits
        return parameters

    def make_actor(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ReflectedActor(**kwargs, lqr_observation_limits=self.lqr_observation_limits).to(self.device)

    def make_critic(self, features_extractor=None):
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
    return np.asarray(config.teacher_bounds)


def lqr_observation_limits(config):
    arm, pendulum, arm_speed, pendulum_speed = teacher_bounds(config)
    # The sine/cosine observation encodes angles modulo a full turn.
    return (
        math.cos(arm), math.cos(pendulum),
        arm_speed / config.max_abs_omega1, pendulum_speed / config.max_abs_omega2,
    )


def teacher_examples(config, seed):
    """Supervise on near-upright states and their held-current recovery trajectories."""
    rng = np.random.default_rng(seed)
    plant = FurutaPendulum(config.sample_time)
    teacher = LQRController(config.max_current)
    observations, commands = [], []
    bounds = teacher_bounds(config)
    rollout_decisions = max(1, round(0.120 / config.control_period))

    for error in rng.uniform(-bounds, bounds, size=(192, 4)):
        plant.reset(error + [0.0, np.pi, 0.0, 0.0])
        for _ in range(rollout_decisions):
            state = plant.x.copy()
            state[1] = upright_error(state)
            if np.any(np.abs(state) > bounds):
                break
            action = teacher.current(plant.x) / config.max_current
            # The memoryless teacher's target is valid across previous commands.
            previous = float(rng.uniform(-1.0, 1.0))
            observations.append(observation_from_plant(plant, config, previous))
            commands.append([action])
            for _ in range(config.action_repeat):
                plant.step(action * config.max_current)

    # The policy architecture supplies the exact mirrored examples.
    return np.asarray(observations), np.asarray(commands, dtype=np.float32)


class LQRTeaching(BaseCallback):
    """Add local LQR supervision to each actor update made by stock SAC."""

    def __init__(self, config, settings):
        super().__init__()
        self.slope_fractions = (0.10, 0.002)
        self.settings = settings
        self.observations, self.targets = teacher_examples(config, settings.seed)
        plant = FurutaPendulum(config.sample_time)
        teacher = LQRController(config.max_current)
        upright = np.array([0.0, np.pi, 0.0, 0.0])
        self.slope_observations, self.slope_targets = [], []
        for fraction in self.slope_fractions:
            for offset in np.diag(np.append(SLOPE_BOUNDS, 0.25)) * fraction:
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
        actions = self.model.actor(torch.cat((observations, self.slope_obs)), deterministic=True)
        predicted, slope = actions.split((len(targets), len(self.slope_observations)))
        local_mse = torch.nn.functional.mse_loss(predicted, targets)
        slope_errors = (slope - self.slope_actions).view(len(self.slope_fractions), -1)
        slope_mse = (slope_errors.square().mean(dim=1) / self.slope_fractions_squared).mean()
        return local_mse + slope_mse

    def _on_training_start(self):
        device = self.model.device
        self.rng = torch.Generator(device=device).manual_seed(self.settings.seed)
        self.teacher_obs = torch.as_tensor(self.observations, device=device)
        self.teacher_actions = torch.as_tensor(self.targets, device=device)
        self.slope_obs = torch.as_tensor(np.asarray(self.slope_observations), device=device)
        self.slope_actions = torch.as_tensor(self.slope_targets, dtype=torch.float32, device=device)
        self.slope_fractions_squared = torch.tensor(
            self.slope_fractions, dtype=torch.float32, device=device,
        ).square()
        optimizer = self.model.actor.optimizer
        for _ in range(self.settings.teacher_pretrain_updates):
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss()
            loss.backward()
            optimizer.step()
        self.logger.record("teacher/pretrain_loss", loss.item())
        self.hook = optimizer.register_step_pre_hook(self.add_teacher_gradient)

    def add_teacher_gradient(self, optimizer, args, kwargs):
        parameters = [p for p in self.model.actor.parameters() if p.requires_grad]
        with torch.enable_grad():
            loss = self.loss()
            gradients = torch.autograd.grad(
                self.settings.teacher_loss_weight * loss, parameters, allow_unused=True,
            )
        for param, grad in zip(parameters, gradients):
            if grad is not None:
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad.add_(grad)
        if self.model.num_timesteps % 64 == 0:
            self.logger.record("teacher/local_loss", loss.detach().item())

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
        self.register_buffer("reflection", actor.reflection.clone())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        pair = torch.stack((observation, observation * self.reflection))
        direct, mirror = self.mean(self.latent(pair)).unbind(0)
        return torch.tanh((direct - mirror) * 0.5)


def local_diagnostics(controller: DeterministicActor, config) -> dict:
    """Finite differences of the exported actor and its nonlinear held-current map."""
    plant = FurutaPendulum(config.sample_time)
    upright = np.array([0.0, np.pi, 0.0, 0.0])
    state_order = ["arm_angle", "upright_error", "arm_velocity", "pendulum_velocity"]
    teacher_gain = (-LQRController(config.max_current).gain).tolist()
    state_order.append("previous_normalized_action")
    teacher_gain.append(0.0)
    state_size = len(state_order)

    def transition(point):
        plant.reset(upright + point[:4])
        previous_command = point[4]
        observation = observation_from_plant(plant, config, previous_command)
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
        error[0] = arm_position_error(error[0], 0.0)
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


class PeriodicValidationCallback(EvalCallback):
    def __init__(self, env, directory, settings):
        super().__init__(
            env, log_path=str(directory / "validation"),
            eval_freq=min(settings.validation_every, settings.decisions),
            n_eval_episodes=settings.validation_episodes, deterministic=True, warn=False,
        )
        self.directory = directory
        self.reset_seed = settings.validation_seed
        self.episodes = []
        self.history = []
        self.best_score = (-1.0, -1.0, -float("inf"))
        self.best_step = 0

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            self.episodes.append(locals_["info"]["validation_metrics"])

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.reset_seed)
        proceed = super()._on_step()
        if self.episodes:
            score = (
                sum(self._is_success_buffer) / len(self._is_success_buffer),
                sum(ep["balanced_after_teacher_entry"] for ep in self.episodes) / len(self.episodes),
                self.last_mean_reward,
            )
            if score > self.best_score:
                self.best_score = score
                self.best_step = self.num_timesteps
                destination = self.directory / "best"
                destination.mkdir(exist_ok=True)
                self.model.save(destination / "best_model.zip")
                print(f"Selected checkpoint at {self.best_step:,} decisions", flush=True)
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
            atomic_write_json(self.directory / "validation_metrics.json", {
                "selected_step": self.best_step, "checkpoints": self.history,
            })
        return proceed
