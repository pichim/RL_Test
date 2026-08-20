"""Nonlinear current-controlled Furuta pendulum model.

State order
-----------
    x = [theta1, theta2, omega1, omega2]

``theta2 = 0`` is the downward equilibrium and ``theta2 = +/-pi`` is upright.
The input is motor current in ampere.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class FurutaPendulum:
    """Small deterministic simulator for the mechanical plant."""

    def __init__(self, sample_time: float = 0.001) -> None:
        self.sample_time = float(sample_time)
        if not np.isfinite(self.sample_time) or self.sample_time <= 0.0:
            raise ValueError("sample_time must be finite and positive")

        self.g = 9.80665

        # Nominal motor and mechanical parameters.
        self.nominal_km = 97.5e-3

        self.nominal_m1 = 122.2e-3
        self.nominal_J1zz = 11270.84e-9
        self.l1 = 0.027248
        self.L1 = 0.09 - 9e-3 / 2.0
        self.nominal_b1 = 1e-6

        self.nominal_m2 = 28e-3
        self.r2 = 9e-3 / 2.0
        self.L2 = 161e-3
        self.l2 = self.L2 / 2.0
        self.nominal_b2 = 4.4e-5

        self.x = np.zeros(4, dtype=np.float64)
        # Episode metadata used by the evaluator. The actuator state itself is
        # integrated explicitly by ``step_filtered_current`` and remains hidden
        # from the mechanical state vector.
        self.current_filter_cutoff_hz = 0.0
        self.action_dead_time_samples = 0
        self.set_parameter_scales()

    def set_parameter_scales(
        self,
        motor_torque: float = 1.0,
        arm_mass: float = 1.0,
        arm_inertia: float = 1.0,
        pendulum_mass: float = 1.0,
        pendulum_inertia: float = 1.0,
        arm_damping: float = 1.0,
        pendulum_damping: float = 1.0,
    ) -> None:
        """Set independent motor, mass, rotary-inertia, and damping scales."""
        scales = np.asarray(
            [
                motor_torque,
                arm_mass,
                arm_inertia,
                pendulum_mass,
                pendulum_inertia,
                arm_damping,
                pendulum_damping,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("parameter scales must be finite and positive")

        self.parameter_scales = scales.copy()
        self.km = self.nominal_km * motor_torque

        self.m1 = self.nominal_m1 * arm_mass
        self.J1zz = self.nominal_J1zz * arm_inertia

        self.m2 = self.nominal_m2 * pendulum_mass
        nominal_J2xx = 0.5 * self.nominal_m2 * self.r2**2
        nominal_J2yy = (
            0.25 * self.nominal_m2 * self.r2**2
            + (1.0 / 12.0) * self.nominal_m2 * self.L2**2
        )
        self.J2xx = nominal_J2xx * pendulum_inertia
        self.J2yy = nominal_J2yy * pendulum_inertia
        self.J2zz = nominal_J2yy * pendulum_inertia

        self.b1 = self.nominal_b1 * arm_damping
        self.b2 = self.nominal_b2 * pendulum_damping

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        """Reset and return the four-state vector."""
        if state is None:
            self.x[:] = 0.0
            return self.x.copy()

        state = np.asarray(state, dtype=np.float64)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("state must contain four finite values")
        self.x = state.copy()
        return self.x.copy()

    @property
    def upright_energy(self) -> float:
        """Potential energy of the stationary upright equilibrium."""
        return float(2.0 * self.m2 * self.g * self.l2)

    def mass_matrix(self, state: np.ndarray | None = None) -> np.ndarray:
        """Return the 2x2 generalized mass matrix."""
        x = self.x if state is None else np.asarray(state, dtype=np.float64)
        theta2 = float(x[1])
        s2 = np.sin(theta2)
        c2 = np.cos(theta2)

        m11 = (
            self.J1zz
            + self.m1 * self.l1**2
            + self.m2 * self.L1**2
            + (self.J2yy + self.m2 * self.l2**2) * s2**2
            + self.J2xx * c2**2
        )
        m12 = self.m2 * self.L1 * self.l2 * c2
        m22 = self.J2zz + self.m2 * self.l2**2
        return np.array([[m11, m12], [m12, m22]], dtype=np.float64)

    def total_energy(self, state: np.ndarray | None = None) -> float:
        """Return mechanical energy relative to the downward equilibrium."""
        x = self.x if state is None else np.asarray(state, dtype=np.float64)
        omega = x[2:4]
        kinetic = 0.5 * omega @ self.mass_matrix(x) @ omega
        potential = self.m2 * self.g * self.l2 * (1.0 - np.cos(x[1]))
        return float(kinetic + potential)

    def _dynamics_matrices(
        self,
        theta: np.ndarray,
        omega: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta2 = theta[1]
        omega1 = omega[0]
        omega2 = omega[1]

        s2 = np.sin(theta2)
        c2 = np.cos(theta2)
        s2c2 = np.sin(2.0 * theta2)

        mass = self.mass_matrix(np.array([theta[0], theta2, omega1, omega2]))

        c11 = omega2 * s2c2 * (
            self.m2 * self.l2**2 + self.J2yy - self.J2xx
        )
        c12 = -self.m2 * self.L1 * self.l2 * s2 * omega2
        c21 = 0.5 * omega1 * s2c2 * (
            -self.m2 * self.l2**2 - self.J2yy + self.J2xx
        )
        coriolis = np.array([[c11, c12], [c21, 0.0]], dtype=np.float64)

        gravity = np.array([0.0, self.m2 * self.g * self.l2 * s2])
        damping = np.diag([self.b1, self.b2])
        return mass, coriolis, gravity, damping

    def state_derivative(self, state: np.ndarray, current: float) -> np.ndarray:
        """Return x_dot = f(x, current)."""
        x = np.asarray(state, dtype=np.float64)
        theta = x[0:2]
        omega = x[2:4]
        mass, coriolis, gravity, damping = self._dynamics_matrices(theta, omega)

        torque = np.array([self.km * float(current), 0.0], dtype=np.float64)
        acceleration = np.linalg.solve(
            mass,
            torque - coriolis @ omega - gravity - damping @ omega,
        )
        return np.array(
            [omega[0], omega[1], acceleration[0], acceleration[1]],
            dtype=np.float64,
        )

    def step(self, current: float) -> np.ndarray:
        """Advance one sample with fourth-order Runge-Kutta integration."""
        current = float(current)
        if not np.isfinite(current):
            raise ValueError("current must be finite")

        dt = self.sample_time
        x = self.x
        k1 = self.state_derivative(x, current)
        k2 = self.state_derivative(x + 0.5 * dt * k1, current)
        k3 = self.state_derivative(x + 0.5 * dt * k2, current)
        k4 = self.state_derivative(x + dt * k3, current)
        self.x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(self.x)):
            raise RuntimeError(f"integration produced invalid state: {self.x}")
        return self.x.copy()

    def step_feedback(self, feedback: Callable[[np.ndarray], float]) -> np.ndarray:
        """Advance one sample while evaluating feedback at every RK4 stage.

        This approximates continuous state feedback without imposing a zero-order
        hold on the LQR current. It is used only for the local LQR path.
        """
        dt = self.sample_time
        x = self.x

        def derivative(state: np.ndarray) -> np.ndarray:
            current = float(feedback(state))
            if not np.isfinite(current):
                raise ValueError("feedback current must be finite")
            return self.state_derivative(state, current)

        k1 = derivative(x)
        k2 = derivative(x + 0.5 * dt * k1)
        k3 = derivative(x + 0.5 * dt * k2)
        k4 = derivative(x + dt * k3)
        self.x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not np.all(np.isfinite(self.x)):
            raise RuntimeError(f"integration produced invalid state: {self.x}")
        return self.x.copy()

    def step_filtered_current(
        self,
        command: float,
        applied_current: float,
        cutoff_hz: float,
    ) -> tuple[np.ndarray, float]:
        """Advance SAC mechanics and a hidden first-order current response.

        The augmented dynamics are integrated together at every RK4 stage::

            d i / d t = 2 pi f_c (i_command - i)

        ``applied_current`` is deliberately kept outside the four mechanical
        states. This method accepts only SAC's scalar zero-order-held request;
        LQR uses :meth:`step_feedback` and bypasses this uncertainty model.
        """
        command = float(command)
        applied_current = float(applied_current)
        cutoff_hz = float(cutoff_hz)
        if not np.isfinite(command):
            raise ValueError("command current must be finite")
        if not np.isfinite(applied_current):
            raise ValueError("applied_current must be finite")
        if not np.isfinite(cutoff_hz) or cutoff_hz < 0.0:
            raise ValueError("cutoff_hz must be finite and nonnegative")

        # Zero disables the uncertainty model and applies the requested
        # current directly to the mechanical plant.
        if cutoff_hz == 0.0:
            state = self.step(command)
            return state, command

        angular_cutoff = 2.0 * np.pi * cutoff_hz
        augmented = np.concatenate((self.x, [applied_current]))

        def derivative(value: np.ndarray) -> np.ndarray:
            state = value[:4]
            actual = float(value[4])
            return np.concatenate(
                (
                    self.state_derivative(state, actual),
                    [angular_cutoff * (command - actual)],
                )
            )

        dt = self.sample_time
        k1 = derivative(augmented)
        k2 = derivative(augmented + 0.5 * dt * k1)
        k3 = derivative(augmented + 0.5 * dt * k2)
        k4 = derivative(augmented + dt * k3)
        augmented = augmented + (dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        if not np.all(np.isfinite(augmented)):
            raise RuntimeError(
                f"integration produced invalid augmented state: {augmented}"
            )
        self.x = augmented[:4]
        return self.x.copy(), float(augmented[4])
