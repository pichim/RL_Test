"""The periodic direct-current swing-up and balance task."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import TASK, Task
from furuta_model import FurutaPendulum

_NOMINAL_ENERGY_MODEL = FurutaPendulum()


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def arm_position_error(theta1: float, reference: float) -> float:
    return wrap_angle(float(theta1 - reference))


def upright_error(state: np.ndarray) -> float:
    """Return pendulum error from upright in the MATLAB angle convention."""
    return wrap_angle(float(state[1]) - np.pi)


def nominal_pendulum_energy_error(state: np.ndarray) -> float:
    """Return normalized nominal pendulum-energy error for shaping and diagnostics."""
    _, theta2, _, omega2 = np.asarray(state, dtype=np.float64)
    pendulum_inertia = (
        _NOMINAL_ENERGY_MODEL.J2zz
        + _NOMINAL_ENERGY_MODEL.m2 * _NOMINAL_ENERGY_MODEL.l2**2
    )
    energy = (
        0.5 * pendulum_inertia * omega2**2
        + _NOMINAL_ENERGY_MODEL.m2
        * _NOMINAL_ENERGY_MODEL.g
        * _NOMINAL_ENERGY_MODEL.l2
        * (1.0 - np.cos(theta2))
    )
    return float(
        (energy - _NOMINAL_ENERGY_MODEL.upright_energy)
        / _NOMINAL_ENERGY_MODEL.upright_energy
    )


def mechanical_capture_ready(state: np.ndarray, config: Task) -> bool:
    """Check the broad capture limits for upright error and angular velocities."""
    _, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    return bool(
        abs(upright_error(state)) <= config.capture_angle
        and abs(omega1) <= config.capture_omega1
        and abs(omega2) <= config.capture_omega2
    )


def balanced_state_ready(
    state: np.ndarray,
    theta1_reference: float,
    config: Task,
) -> bool:
    """Return True inside the tight final-balance box."""
    theta1, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    return bool(
        np.all(np.isfinite(state))
        and abs(upright_error(state)) <= config.balance_angle
        and abs(arm_position_error(theta1, theta1_reference)) <= config.balance_theta1
        and abs(omega1) <= config.balance_omega1
        and abs(omega2) <= config.balance_omega2
    )


def within_matlab_constraints(state: np.ndarray, config: Task) -> bool:
    """Return the conservative constraint flag F used by this task."""
    _, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    return bool(
        np.all(np.isfinite(state))
        and abs(omega1) <= config.max_abs_omega1
        and abs(omega2) <= config.max_abs_omega2
    )


def unsafe(state: np.ndarray, config: Task) -> bool:
    """Return True after a MATLAB constraint or numerical-guard violation."""
    return not within_matlab_constraints(state, config)


def sample_matlab_initial_state(
    rng: np.random.Generator,
    config: Task,
) -> np.ndarray:
    """Draw four independent uniform reset coordinates in state order."""
    return np.array(
        [
            rng.uniform(
                -config.reset_theta1_half_range,
                config.reset_theta1_half_range,
            ),
            rng.uniform(
                -config.reset_theta2_half_range,
                config.reset_theta2_half_range,
            ),
            rng.uniform(
                -config.reset_omega1_half_range,
                config.reset_omega1_half_range,
            ),
            rng.uniform(
                -config.reset_omega2_half_range,
                config.reset_omega2_half_range,
            ),
        ],
        dtype=np.float64,
    )


def encode_observation(state, config, previous_action=0.0):
    theta1, theta2, omega1, omega2 = state
    pendulum_error = wrap_angle(theta2 - np.pi)
    return np.array([
        np.sin(theta1),
        np.cos(theta1),
        np.clip(omega1 / config.max_abs_omega1, -1.0, 1.0),
        np.sin(pendulum_error),
        np.cos(pendulum_error),
        np.clip(omega2 / config.max_abs_omega2, -1.0, 1.0),
        np.clip(previous_action, -1.0, 1.0),
    ], dtype=np.float32)


def observation_from_plant(plant, config, previous_action=0.0):
    """Return the seven policy features using the true simulated state."""
    return encode_observation(plant.x, config, previous_action)


class Encoder:
    """Simulate encoder samples and a filtered backward-difference velocity."""

    def __init__(self, settings, task):
        self.settings = settings
        self.period_steps = settings.plant_steps(task)
        self.noise_std = np.array([settings.arm_noise_std_rad, settings.pendulum_noise_std_rad])
        decay = 2 * np.pi * settings.velocity_cutoff_hz * settings.sample_time
        self.alpha = np.exp(-decay)
        self.velocity_gain = -np.expm1(-decay) / settings.sample_time
        self.warmup_steps = int(np.ceil(5 / decay))
        self.reset_rng = np.random.default_rng()

    def _measure(self, angles):
        return angles + self.rng.normal(size=2) * self.noise_std

    def _sample(self, angles):
        measured = self._measure(angles)
        difference = (measured - self.angles + np.pi) % (2 * np.pi) - np.pi
        self.velocity = self.alpha * self.velocity + self.velocity_gain * difference
        self.angles = measured

    def reset(self, state, seed=None):
        if seed is not None:
            self.reset_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.rng = np.random.default_rng(int(self.reset_rng.integers(2**63)))
        dt = self.settings.sample_time
        self.velocity = np.zeros(2)
        # Synthesize pre-reset encoder readings assuming constant initial velocity.
        self.angles = self._measure(state[:2] - self.warmup_steps * dt * state[2:])
        for age in range(self.warmup_steps - 1, -1, -1):
            self._sample(state[:2] - age * dt * state[2:])

    def advance(self, state, plant_steps):
        if plant_steps % self.period_steps == 0:
            self._sample(state[:2])

    def observation(self, config, previous_action):
        return encode_observation(np.concatenate((self.angles, self.velocity)), config, previous_action)


def policy_observation(plant, config, previous_action, encoder=None):
    clean = observation_from_plant(plant, config, previous_action)
    if encoder is None:
        return clean
    return {"measured": encoder.observation(config, previous_action), "clean": clean}


def matlab_reward(
    state: np.ndarray,
    action: float,
    previous_action: float,
    config: Task,
) -> float:
    """Return the conservative MATLAB-derived reward with current input."""
    theta1, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    pendulum_error = upright_error(state)
    action_change = float(action) - float(previous_action)
    arm_cost = arm_position_error(theta1, 0.0)**2
    step_reward = config.constraint_reward_weight * within_matlab_constraints(state, config)
    cost = (
        config.arm_angle_weight * arm_cost
        + pendulum_error**2
        + config.arm_velocity_weight * omega1**2
        + config.velocity_weight * omega2**2
        + config.current_weight * float(action) ** 2
        + config.action_change_weight * action_change**2
    )
    return float(step_reward - config.reward_scale * cost)


class FurutaSwingUpEnv(gym.Env):
    """Hold each current over RK4 steps and track uninterrupted balance."""

    metadata = {"render_modes": []}

    def __init__(self, config: Task = TASK, *, continue_after_success=False, encoder_settings=None) -> None:
        super().__init__()
        self.config = config
        self.continue_after_success = continue_after_success
        self.render_mode = None
        self.plant = FurutaPendulum(sample_time=config.sample_time)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        features = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.encoder = None if encoder_settings is None else Encoder(encoder_settings, config)
        self.observation_space = features if self.encoder is None else spaces.Dict({
            "measured": features, "clean": features,
        })
        self.max_plant_steps = int(round(config.episode_time / config.sample_time))
        self.plant_step_count = 0
        self.last_normalized_action = 0.0
        self.last_command_current = 0.0
        self.reset_in_capture_region = False
        self.success_step_count = 0
        self.longest_balance_steps = 0
        self.hold_completed = False
        self.required_success_steps = int(
            round(config.balance_hold_time / config.sample_time)
        )

    @staticmethod
    def _parse_action(action) -> float:
        values = np.asarray(action, dtype=np.float64)
        if values.size != 1 or not np.all(np.isfinite(values)):
            raise ValueError("action must contain one finite value")
        return float(np.clip(values.reshape(-1)[0], -1.0, 1.0))

    def _info(self, is_unsafe: bool) -> dict:
        success = bool(
            not is_unsafe
            and self.success_step_count >= self.required_success_steps
        )
        return {
            "state": self.plant.x.copy(),
            "normalized_action": self.last_normalized_action,
            "command_current": self.last_command_current,
            "energy_error": nominal_pendulum_energy_error(self.plant.x),
            "initial_capture": self.reset_in_capture_region,
            "success": success,
            "is_success": success,
            "hold_completed": self.hold_completed,
            "longest_hold_s": self.longest_balance_steps * self.config.sample_time,
            "balance_finish_s": (
                (self.plant_step_count - self.success_step_count + self.required_success_steps)
                * self.config.sample_time if success else None
            ),
            "unsafe": is_unsafe,
        }

    def observation(self):
        return policy_observation(self.plant, self.config, self.last_normalized_action, self.encoder)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options

        # Retain the RNG draw order even for explicit reset states.
        sampled = sample_matlab_initial_state(self.np_random, self.config)
        state = np.asarray(options.get("initial_state", sampled), dtype=np.float64)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain four finite values")
        self.plant.reset(state)
        self.plant.x[1] = wrap_angle(self.plant.x[1])
        self.reset_in_capture_region = mechanical_capture_ready(
            self.plant.x,
            self.config,
        )
        self.plant_step_count = 0
        self.last_normalized_action = 0.0
        self.last_command_current = 0.0
        self.success_step_count = 0
        self.longest_balance_steps = 0
        self.hold_completed = False
        if self.encoder is not None:
            self.encoder.reset(self.plant.x, seed)
        return self.observation(), self._info(False)

    def step(self, action):
        normalized_action = self._parse_action(action)
        previous_action = self.last_normalized_action
        command_current = normalized_action * self.config.max_current
        integration_failed = False
        completed_this_step = False

        steps_remaining = self.max_plant_steps - self.plant_step_count
        for _ in range(min(self.config.action_repeat, steps_remaining)):
            last_finite_state = self.plant.x.copy()
            try:
                self.plant.step(command_current)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                # Never expose NaN/Inf observations or rewards to the replay
                # buffer. Terminate from the last known finite plant state.
                self.plant.x = last_finite_state
                integration_failed = True
                self.success_step_count = 0
                break
            self.plant.x[1] = wrap_angle(self.plant.x[1])
            self.plant_step_count += 1
            if self.encoder is not None:
                self.encoder.advance(self.plant.x, self.plant_step_count)
            step_unsafe = unsafe(self.plant.x, self.config)
            step_success = balanced_state_ready(
                self.plant.x,
                theta1_reference=0.0,
                config=self.config,
            )
            if not step_unsafe and step_success:
                self.success_step_count += 1
                self.longest_balance_steps = max(self.longest_balance_steps, self.success_step_count)
                if self.success_step_count == self.required_success_steps and not self.hold_completed:
                    self.hold_completed = completed_this_step = True
            else:
                self.success_step_count = 0
            if step_unsafe or (completed_this_step and not self.continue_after_success):
                break

        self.last_normalized_action = normalized_action
        self.last_command_current = command_current
        is_unsafe = bool(integration_failed or unsafe(self.plant.x, self.config))
        terminated = is_unsafe or (completed_this_step and not self.continue_after_success)
        truncated = bool(
            self.plant_step_count >= self.max_plant_steps and not terminated
        )
        reward = matlab_reward(
            self.plant.x,
            normalized_action,
            previous_action,
            self.config,
        )
        if completed_this_step:
            reward += self.config.terminal_success_bonus
        if integration_failed and within_matlab_constraints(
            self.plant.x,
            self.config,
        ):
            # An integration failure is a constraint failure even though the
            # state was rolled back for a finite terminal transition.
            reward -= self.config.constraint_reward_weight
        return (
            self.observation(),
            reward,
            terminated,
            truncated,
            self._info(is_unsafe),
        )
