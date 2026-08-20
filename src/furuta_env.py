"""MATLAB-derived pure-SAC swing-up environment for a Furuta pendulum.

The task combines the published QUBE observation and action-change penalty
with the smaller velocity weights used in MathWorks' hardware SAC example.
SAC directly commands normalized motor current. An optional hidden
continuous-time first-order actuator response can be enabled between the
requested current and the mechanical plant. The current task trains complete
swing-up and balance; the retained LQR is exercised only by the standalone
comparison evaluator.

The state convention of :mod:`furuta_model` is retained: ``theta2 = 0`` is
downward and ``theta2 = +/-pi`` is upright. Consequently, the MATLAB pendulum
angle is represented here by ``upright_error(state)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from furuta_model import FurutaPendulum


TASK_VERSION = "matlab_hardware_sac_pure_v0"


@dataclass(frozen=True)
class FurutaConfig:
    """Plant-independent task, handover, and evaluation parameters."""

    task_version: str = TASK_VERSION

    # A 1 kHz RK4 grid represents the MATLAB 5 ms agent sample time exactly.
    sample_time: float = 0.001
    action_repeat: int = 5
    episode_time: float = 5.0
    max_current: float = 0.5
    current_filter_cutoff_hz: float = 0.0
    current_filter_cutoff_randomization_hz: float = 0.0
    current_filter_cutoff_min_hz: float = 0.0
    current_filter_cutoff_max_hz: float = 0.0

    # MATLAB reset and termination limits, converted to this model's angle
    # convention. Both velocities have a conservative hard safety bound.
    reset_theta1_half_range: float = np.pi / 2.0
    reset_theta2_half_range: float = np.pi
    reset_omega1_half_range: float = 2.0
    reset_omega2_half_range: float = 2.0
    randomized_reset_omega1_half_range: float = 2.0
    randomized_reset_omega2_half_range: float = 2.0
    arm_angle_limit: float = 5.0 * np.pi / 8.0
    max_abs_omega1: float = 30.0
    max_abs_omega2: float = 30.0

    # When enabled, one plant is sampled per episode. The same ranges are
    # reused by the randomized standalone evaluation modes.
    training_parameter_randomization: bool = False
    motor_torque_randomization: float = 0.10
    arm_mass_randomization: float = 0.10
    arm_inertia_randomization: float = 0.10
    pendulum_mass_randomization: float = 0.10
    pendulum_inertia_randomization: float = 0.10
    arm_damping_randomization: float = 0.10
    pendulum_damping_randomization: float = 0.10
    action_dead_time_max_samples: int = 0
    # Mandatory deployment handover retained from this project.
    capture_angle: float = np.deg2rad(10.0)
    capture_omega1: float = 2.0
    capture_omega2: float = 2.0
    handover_current_tolerance: float = 0.10
    handover_hold_time: float = 0.10
    terminal_success_bonus: float = 100.0

    # Shared definition used by TensorBoard and the standalone evaluator.
    balance_angle: float = np.deg2rad(5.0)
    balance_theta1: float = 0.25
    balance_omega1: float = 0.50
    balance_omega2: float = 1.00
    balance_hold_time: float = 1.00

    # The hardware SAC example weights both velocity squares by 0.01. Keep
    # separate arm and pendulum weights for controlled trajectory shaping.
    constraint_reward_weight: float = 1.0
    reward_scale: float = 0.10
    velocity_weight: float = 0.010
    arm_angle_weight: float = 1.5
    arm_velocity_weight: float = 0.015
    action_change_weight: float = 3.0

    def __post_init__(self) -> None:
        if self.task_version != TASK_VERSION:
            raise ValueError(f"unsupported task_version: {self.task_version}")
        positive = {
            "sample_time": self.sample_time,
            "action_repeat": self.action_repeat,
            "episode_time": self.episode_time,
            "max_current": self.max_current,
            "reset_theta1_half_range": self.reset_theta1_half_range,
            "reset_theta2_half_range": self.reset_theta2_half_range,
            "reset_omega1_half_range": self.reset_omega1_half_range,
            "reset_omega2_half_range": self.reset_omega2_half_range,
            "randomized_reset_omega1_half_range": (
                self.randomized_reset_omega1_half_range
            ),
            "randomized_reset_omega2_half_range": (
                self.randomized_reset_omega2_half_range
            ),
            "arm_angle_limit": self.arm_angle_limit,
            "max_abs_omega1": self.max_abs_omega1,
            "max_abs_omega2": self.max_abs_omega2,
            "capture_angle": self.capture_angle,
            "capture_omega1": self.capture_omega1,
            "capture_omega2": self.capture_omega2,
            "handover_current_tolerance": self.handover_current_tolerance,
            "handover_hold_time": self.handover_hold_time,
            "terminal_success_bonus": self.terminal_success_bonus,
            "balance_angle": self.balance_angle,
            "balance_theta1": self.balance_theta1,
            "balance_omega1": self.balance_omega1,
            "balance_omega2": self.balance_omega2,
            "balance_hold_time": self.balance_hold_time,
            "reward_scale": self.reward_scale,
            "velocity_weight": self.velocity_weight,
            "arm_angle_weight": self.arm_angle_weight,
            "arm_velocity_weight": self.arm_velocity_weight,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.action_repeat, int):
            raise ValueError("action_repeat must be an integer")
        if (
            not isinstance(self.action_dead_time_max_samples, int)
            or self.action_dead_time_max_samples < 0
        ):
            raise ValueError(
                "action_dead_time_max_samples must be a nonnegative integer"
            )
        if self.reset_theta1_half_range > self.arm_angle_limit:
            raise ValueError("the arm reset range must lie inside the arm limit")
        for randomized_name, nominal_name, safety_name in (
            (
                "randomized_reset_omega1_half_range",
                "reset_omega1_half_range",
                "max_abs_omega1",
            ),
            (
                "randomized_reset_omega2_half_range",
                "reset_omega2_half_range",
                "max_abs_omega2",
            ),
        ):
            randomized_bound = getattr(self, randomized_name)
            if randomized_bound < getattr(self, nominal_name):
                raise ValueError(
                    f"{randomized_name} must not be smaller than {nominal_name}"
                )
            if randomized_bound >= getattr(self, safety_name):
                raise ValueError(
                    f"{randomized_name} must be smaller than {safety_name}"
                )
        nonnegative = {
            "current_filter_cutoff_hz": self.current_filter_cutoff_hz,
            "current_filter_cutoff_randomization_hz": (
                self.current_filter_cutoff_randomization_hz
            ),
            "current_filter_cutoff_min_hz": self.current_filter_cutoff_min_hz,
            "current_filter_cutoff_max_hz": self.current_filter_cutoff_max_hz,
            "constraint_reward_weight": self.constraint_reward_weight,
            "action_change_weight": self.action_change_weight,
            "motor_torque_randomization": self.motor_torque_randomization,
            "arm_mass_randomization": self.arm_mass_randomization,
            "arm_inertia_randomization": self.arm_inertia_randomization,
            "pendulum_mass_randomization": self.pendulum_mass_randomization,
            "pendulum_inertia_randomization": self.pendulum_inertia_randomization,
            "arm_damping_randomization": self.arm_damping_randomization,
            "pendulum_damping_randomization": self.pendulum_damping_randomization,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "motor_torque_randomization",
            "arm_mass_randomization",
            "arm_inertia_randomization",
            "pendulum_mass_randomization",
            "pendulum_inertia_randomization",
            "arm_damping_randomization",
            "pendulum_damping_randomization",
        ):
            if getattr(self, name) >= 1.0:
                raise ValueError(f"{name} must be smaller than one")
        if not isinstance(self.training_parameter_randomization, bool):
            raise ValueError("training_parameter_randomization must be boolean")
        if self.current_filter_cutoff_randomization_hz > 0.0:
            if self.current_filter_cutoff_hz <= 0.0:
                raise ValueError(
                    "current filter randomization requires a positive nominal cutoff"
                )
            if (
                self.current_filter_cutoff_randomization_hz
                >= self.current_filter_cutoff_hz
            ):
                raise ValueError(
                    "current filter randomization must keep the cutoff positive"
                )
        explicit_cutoff_range = bool(
            self.current_filter_cutoff_min_hz > 0.0
            or self.current_filter_cutoff_max_hz > 0.0
        )
        if explicit_cutoff_range:
            if self.current_filter_cutoff_randomization_hz > 0.0:
                raise ValueError(
                    "use either a symmetric or explicit current-filter range"
                )
            if (
                self.current_filter_cutoff_min_hz <= 0.0
                or self.current_filter_cutoff_max_hz <= 0.0
                or self.current_filter_cutoff_hz <= 0.0
            ):
                raise ValueError(
                    "an explicit current-filter range requires positive minimum, "
                    "maximum, and nominal cutoffs"
                )
            if not (
                self.current_filter_cutoff_min_hz
                <= self.current_filter_cutoff_hz
                <= self.current_filter_cutoff_max_hz
            ):
                raise ValueError(
                    "the nominal current-filter cutoff must lie inside its range"
                )


DEFAULT_CONFIG = FurutaConfig()
_NOMINAL_ENERGY_MODEL = FurutaPendulum()


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


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


def mechanical_capture_ready(state: np.ndarray, config: FurutaConfig) -> bool:
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
    config: FurutaConfig,
) -> bool:
    """Return True inside the tight final-balance box."""
    theta1, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    return bool(
        np.all(np.isfinite(state))
        and abs(upright_error(state)) <= config.balance_angle
        and abs(theta1 - theta1_reference) <= config.balance_theta1
        and abs(omega1) <= config.balance_omega1
        and abs(omega2) <= config.balance_omega2
    )


LQR_GAIN = np.array([-0.4472, 3.4376, -0.1555, 0.3031], dtype=np.float64)


def lqr_handover_current(
    state: np.ndarray,
    max_current: float,
    gain: np.ndarray = LQR_GAIN,
) -> float:
    """Return LQR current if the current arm angle becomes its reference."""
    _, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    error_state = np.array(
        [0.0, upright_error(state), omega1, omega2],
        dtype=np.float64,
    )
    return float(np.clip(-np.asarray(gain) @ error_state, -max_current, max_current))


def capture_ready(
    state: np.ndarray,
    current: float,
    config: FurutaConfig,
) -> bool:
    """Return True when mechanical and current states permit a gentle switch."""
    candidate = lqr_handover_current(state, config.max_current)
    return bool(
        mechanical_capture_ready(state, config)
        and abs(float(current) - candidate) <= config.handover_current_tolerance
    )


def within_matlab_constraints(state: np.ndarray, config: FurutaConfig) -> bool:
    """Return the conservative constraint flag F used by this task."""
    theta1, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    return bool(
        np.all(np.isfinite(state))
        and abs(theta1) <= config.arm_angle_limit
        and abs(omega1) <= config.max_abs_omega1
        and abs(omega2) <= config.max_abs_omega2
    )


def unsafe(state: np.ndarray, config: FurutaConfig) -> bool:
    """Return True after a MATLAB constraint or numerical-guard violation."""
    return not within_matlab_constraints(state, config)


def sample_matlab_initial_state(
    rng: np.random.Generator,
    config: FurutaConfig,
    randomization_scale: float = 0.0,
) -> np.ndarray:
    """Sample the reset distribution, optionally widening its velocities."""
    randomization_scale = float(randomization_scale)
    if not np.isfinite(randomization_scale) or not 0.0 <= randomization_scale <= 1.0:
        raise ValueError("randomization scale must be finite and in [0, 1]")
    omega1_half_range = config.reset_omega1_half_range + randomization_scale * (
        config.randomized_reset_omega1_half_range
        - config.reset_omega1_half_range
    )
    omega2_half_range = config.reset_omega2_half_range + randomization_scale * (
        config.randomized_reset_omega2_half_range
        - config.reset_omega2_half_range
    )
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
                -omega1_half_range,
                omega1_half_range,
            ),
            rng.uniform(
                -omega2_half_range,
                omega2_half_range,
            ),
        ],
        dtype=np.float64,
    )


def sample_parameter_scales(
    rng: np.random.Generator,
    config: FurutaConfig,
    scale: float = 1.0,
) -> np.ndarray:
    """Sample independent plant variations for training or evaluation."""
    scale = float(scale)
    if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("randomization scale must be finite and in [0, 1]")
    ranges = np.array(
        [
            config.motor_torque_randomization,
            config.arm_mass_randomization,
            config.arm_inertia_randomization,
            config.pendulum_mass_randomization,
            config.pendulum_inertia_randomization,
            config.arm_damping_randomization,
            config.pendulum_damping_randomization,
        ],
        dtype=np.float64,
    ) * scale
    return rng.uniform(1.0 - ranges, 1.0 + ranges)


def sample_current_filter_cutoff_hz(
    rng: np.random.Generator,
    config: FurutaConfig,
    scale: float = 1.0,
) -> float:
    """Sample one episode cutoff from the scaled hidden-response interval."""
    scale = float(scale)
    if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("randomization scale must be finite and in [0, 1]")
    target_minimum, target_maximum = current_filter_cutoff_range_hz(config)
    minimum = config.current_filter_cutoff_hz + scale * (
        target_minimum - config.current_filter_cutoff_hz
    )
    maximum = config.current_filter_cutoff_hz + scale * (
        target_maximum - config.current_filter_cutoff_hz
    )
    if minimum == maximum:
        return float(config.current_filter_cutoff_hz)
    return float(rng.uniform(minimum, maximum))


def current_filter_cutoff_range_hz(
    config: FurutaConfig,
) -> tuple[float, float]:
    """Return the configured full-scale cutoff interval."""
    if (
        config.current_filter_cutoff_min_hz > 0.0
        or config.current_filter_cutoff_max_hz > 0.0
    ):
        return (
            float(config.current_filter_cutoff_min_hz),
            float(config.current_filter_cutoff_max_hz),
        )
    half_range = config.current_filter_cutoff_randomization_hz
    return (
        float(config.current_filter_cutoff_hz - half_range),
        float(config.current_filter_cutoff_hz + half_range),
    )


def sample_action_dead_time_samples(
    rng: np.random.Generator,
    config: FurutaConfig,
    scale: float = 1.0,
) -> int:
    """Sample an integer hidden delay, progressively expanding its support."""
    scale = float(scale)
    if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("randomization scale must be finite and in [0, 1]")
    scaled_maximum = scale * config.action_dead_time_max_samples
    maximum = int(np.floor(scaled_maximum))
    if rng.random() < scaled_maximum - maximum:
        maximum += 1
    return int(rng.integers(0, maximum + 1)) if maximum > 0 else 0


def observation_from_plant(
    plant: FurutaPendulum,
    config: FurutaConfig,
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
    config: FurutaConfig,
) -> float:
    """Return the conservative MATLAB-derived reward with current input."""
    theta1, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
    pendulum_error = upright_error(state)
    action_change = float(action) - float(previous_action)
    constraint_flag = 1.0 if within_matlab_constraints(state, config) else 0.0
    cost = (
        config.arm_angle_weight * theta1**2
        + pendulum_error**2
        + config.arm_velocity_weight * omega1**2
        + config.velocity_weight * omega2**2
        + float(action) ** 2
        + config.action_change_weight * action_change**2
    )
    return float(
        config.constraint_reward_weight * constraint_flag
        - config.reward_scale * cost
    )


class FurutaSwingUpEnv(gym.Env):
    """Direct-current SAC task derived from the MathWorks QUBE formulation.

    Observation
        ``[sin(theta1), cos(theta1), omega1, sin(upright_error),``
        ``cos(upright_error), omega2, previous_normalized_action]``

    Action
        One normalized current command in ``[-1, 1]``. It is converted to
        ``[-max_current, max_current]`` and held for one 5 ms policy interval.
        When configured, a hidden continuous-time first-order response filters
        the requested current; the applied current remains absent from the
        observation.

    Success
        SAC must remain inside the tight balance region for
        ``balance_hold_time``. Success terminates with a continuation-value
        bonus after SAC has learned complete swing-up and sustained balance.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: FurutaConfig = DEFAULT_CONFIG) -> None:
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
        self.last_plant_command_current = 0.0
        self.last_current = 0.0
        self.randomization_scale = 1.0
        self.episode_current_filter_cutoff_hz = (
            config.current_filter_cutoff_hz
        )
        self.episode_action_dead_time_samples = 0
        self._delayed_current_commands: list[float] = []
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

    def current_for_action(self, action) -> float:
        """Convert a normalized action to its requested motor current."""
        return self._parse_action(action) * self.config.max_current

    def set_randomization_scale(self, scale: float) -> None:
        """Set the fraction of configured uncertainty used on future resets."""
        scale = float(scale)
        if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("randomization scale must be finite and in [0, 1]")
        self.randomization_scale = scale

    def _info(self, is_unsafe: bool) -> dict:
        success = bool(
            not is_unsafe
            and self.success_step_count >= self.required_success_steps
        )
        return {
            "state": self.plant.x.copy(),
            "normalized_action": self.last_normalized_action,
            "command_current": self.last_command_current,
            "plant_command_current": self.last_plant_command_current,
            "current": self.last_current,
            "current_filter_cutoff_hz": self.episode_current_filter_cutoff_hz,
            "action_dead_time_samples": self.episode_action_dead_time_samples,
            "randomization_scale": self.randomization_scale,
            "energy_error": nominal_pendulum_energy_error(self.plant.x),
            "parameter_scales": self.plant.parameter_scales.copy(),
            "initial_capture": self.reset_in_capture_region,
            "success": success,
            "is_success": success,
            "unsafe": is_unsafe,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options

        default_scales = (
            sample_parameter_scales(
                self.np_random,
                self.config,
                self.randomization_scale,
            )
            if self.config.training_parameter_randomization
            else np.ones(7)
        )
        scales = np.asarray(
            options.get("parameter_scales", default_scales),
            dtype=np.float64,
        )
        if scales.shape != (7,) or not np.all(np.isfinite(scales)):
            raise ValueError("parameter_scales must contain seven finite values")
        self.plant.set_parameter_scales(*scales)

        default_filter_cutoff = (
            sample_current_filter_cutoff_hz(
                self.np_random,
                self.config,
                self.randomization_scale,
            )
            if self.config.training_parameter_randomization
            else self.config.current_filter_cutoff_hz
        )
        filter_cutoff = float(
            options.get("current_filter_cutoff_hz", default_filter_cutoff)
        )
        if not np.isfinite(filter_cutoff) or filter_cutoff < 0.0:
            raise ValueError("current_filter_cutoff_hz must be finite and nonnegative")
        self.episode_current_filter_cutoff_hz = filter_cutoff

        default_dead_time = (
            sample_action_dead_time_samples(
                self.np_random,
                self.config,
                self.randomization_scale,
            )
            if self.config.training_parameter_randomization
            else 0
        )
        dead_time = options.get("action_dead_time_samples", default_dead_time)
        if not isinstance(dead_time, (int, np.integer)) or not (
            0 <= int(dead_time) <= self.config.action_dead_time_max_samples
        ):
            raise ValueError(
                "action_dead_time_samples must be an integer within the "
                "configured range"
            )
        self.episode_action_dead_time_samples = int(dead_time)
        self._delayed_current_commands = [0.0] * self.episode_action_dead_time_samples

        state = np.asarray(
            options.get(
                "initial_state",
                sample_matlab_initial_state(
                    self.np_random,
                    self.config,
                    self.randomization_scale
                    if self.config.training_parameter_randomization
                    else 0.0,
                ),
            ),
            dtype=np.float64,
        )
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
        self.last_plant_command_current = 0.0
        self.last_current = 0.0
        self.success_step_count = 0
        return observation_from_plant(self.plant, self.config, 0.0), self._info(False)

    def step(self, action):
        normalized_action = self._parse_action(action)
        previous_action = self.last_normalized_action
        command_current = normalized_action * self.config.max_current
        if self.episode_action_dead_time_samples > 0:
            self._delayed_current_commands.append(command_current)
            plant_command_current = self._delayed_current_commands.pop(0)
        else:
            plant_command_current = command_current
        applied_current = self.last_current
        integration_failed = False

        for _ in range(self.config.action_repeat):
            last_finite_state = self.plant.x.copy()
            last_finite_current = applied_current
            try:
                _, applied_current = self.plant.step_filtered_current(
                    plant_command_current,
                    applied_current,
                    self.episode_current_filter_cutoff_hz,
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                # Never expose NaN/Inf observations or rewards to the replay
                # buffer. Terminate from the last known finite plant state.
                self.plant.x = last_finite_state
                applied_current = last_finite_current
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
        self.last_plant_command_current = plant_command_current
        self.last_current = applied_current
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
