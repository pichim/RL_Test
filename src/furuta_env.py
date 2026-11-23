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
    """Return nominal normalized pendulum-energy error for diagnostics only."""
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
    """Return True when the mechanical state lies inside the LQR entry box."""
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


def observation_from_plant(
    plant: FurutaPendulum,
    config: Task,
    previous_action: float = 0.0,
) -> np.ndarray:
    """Return the seven-element MATLAB observation in this angle convention."""
    theta1, _, omega1, omega2 = plant.x
    pendulum_error = upright_error(plant.x)
    omega1 = np.clip(omega1 / config.max_abs_omega1, -1.0, 1.0)
    omega2 = np.clip(omega2 / config.max_abs_omega2, -1.0, 1.0)
    return np.array(
        [
            np.sin(theta1),
            np.cos(theta1),
            omega1,
            np.sin(pendulum_error),
            np.cos(pendulum_error),
            omega2,
            np.clip(previous_action, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


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
    """Five held RK4 steps per decision and a terminal one-second balance hold."""

    metadata = {"render_modes": []}

    def __init__(self, config: Task = TASK) -> None:
        super().__init__()
        self.config = config
        self.render_mode = None
        self.plant = FurutaPendulum(sample_time=config.sample_time)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        velocity_bound = 1.0
        self.observation_space = spaces.Box(
            low=np.array(
                [-1.0, -1.0, -velocity_bound, -1.0, -1.0, -velocity_bound, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, velocity_bound, 1.0, 1.0, velocity_bound, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.max_plant_steps = int(round(config.episode_time / config.sample_time))
        self.plant_step_count = 0
        self.last_normalized_action = 0.0
        self.last_command_current = 0.0
        self.reset_in_capture_region = False
        self.success_step_count = 0
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
            "unsafe": is_unsafe,
        }

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
        return observation_from_plant(self.plant, self.config, 0.0), self._info(False)

    def step(self, action):
        normalized_action = self._parse_action(action)
        previous_action = self.last_normalized_action
        command_current = normalized_action * self.config.max_current
        integration_failed = False

        for _ in range(self.config.action_repeat):
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
            step_unsafe = unsafe(self.plant.x, self.config)
            step_success = balanced_state_ready(
                self.plant.x,
                theta1_reference=0.0,
                config=self.config,
            )
            if not step_unsafe and step_success:
                self.success_step_count += 1
            else:
                self.success_step_count = 0
            if (
                step_unsafe
                or self.success_step_count >= self.required_success_steps
            ):
                break

        self.last_normalized_action = normalized_action
        self.last_command_current = command_current
        is_unsafe = bool(integration_failed or unsafe(self.plant.x, self.config))
        is_success = bool(
            not is_unsafe
            and self.success_step_count >= self.required_success_steps
        )
        terminated = bool(is_unsafe or is_success)
        truncated = bool(
            self.plant_step_count >= self.max_plant_steps and not terminated
        )
        reward = matlab_reward(
            self.plant.x,
            normalized_action,
            previous_action,
            self.config,
        )
        if is_success:
            reward += self.config.terminal_success_bonus
        if integration_failed and within_matlab_constraints(
            self.plant.x,
            self.config,
        ):
            # An integration failure is a constraint failure even though the
            # state was rolled back for a finite terminal transition.
            reward -= 1.0
        return (
            observation_from_plant(
                self.plant,
                self.config,
                previous_action=normalized_action,
            ),
            reward,
            terminated,
            truncated,
            self._info(is_unsafe),
        )
