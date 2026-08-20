"""Relative-angle LQR and the two-mode evaluation supervisor."""

from __future__ import annotations

from enum import Enum

import numpy as np

from furuta_env import LQR_GAIN, FurutaConfig, capture_ready, upright_error


class LQRController:
    """Saturated local feedback around a motor angle chosen at handover."""

    def __init__(self, max_current: float, gain: np.ndarray = LQR_GAIN) -> None:
        self.max_current = float(max_current)
        self.gain = np.asarray(gain, dtype=np.float64)
        self.theta1_reference = 0.0

    def set_reference(self, state: np.ndarray) -> None:
        """Use the current motor angle as the new local zero."""
        self.theta1_reference = float(np.asarray(state, dtype=np.float64)[0])

    def theta1_error(self, state: np.ndarray) -> float:
        """Return motor angle relative to the most recent handover angle."""
        return float(np.asarray(state, dtype=np.float64)[0] - self.theta1_reference)

    def current(self, state: np.ndarray) -> float:
        _, _, omega1, omega2 = np.asarray(state, dtype=np.float64)
        error_state = np.array(
            [self.theta1_error(state), upright_error(state), omega1, omega2],
            dtype=np.float64,
        )
        return float(
            np.clip(-self.gain @ error_state, -self.max_current, self.max_current)
        )


class Mode(Enum):
    SWING_UP = 0
    LQR = 1


class HybridSupervisor:
    """Hard SAC/LQR switch with angle hysteresis and velocity-gated entry."""

    def __init__(
        self,
        config: FurutaConfig,
        exit_angle: float = np.deg2rad(20.0),
    ) -> None:
        if exit_angle <= config.capture_angle:
            raise ValueError("exit_angle must exceed the LQR entry angle")
        self.config = config
        self.exit_angle = float(exit_angle)
        self.mode = Mode.SWING_UP
        self.capture_step_count = 0
        self.required_capture_steps = max(
            1,
            int(round(config.handover_hold_time / config.sample_time)),
        )

    def reset(self) -> None:
        self.mode = Mode.SWING_UP
        self.capture_step_count = 0

    def update(self, state: np.ndarray, applied_current: float) -> Mode:
        if self.mode is Mode.SWING_UP:
            if capture_ready(state, applied_current, self.config):
                self.capture_step_count += 1
            else:
                self.capture_step_count = 0
            if self.capture_step_count >= self.required_capture_steps:
                self.mode = Mode.LQR
        elif abs(upright_error(state)) >= self.exit_angle:
            self.mode = Mode.SWING_UP
            self.capture_step_count = 0
        return self.mode
