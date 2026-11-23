"""Nonlinear current-controlled Furuta pendulum model.

State order
-----------
    x = [theta1, theta2, omega1, omega2]

``theta2 = 0`` is the downward equilibrium and ``theta2 = +/-pi`` is upright.
The input is motor current in ampere.
"""

from __future__ import annotations

import numpy as np


class FurutaPendulum:
    """Small deterministic simulator for the mechanical plant."""

    def __init__(self, sample_time: float = 0.001) -> None:
        self.sample_time = float(sample_time)
        if not np.isfinite(self.sample_time) or self.sample_time <= 0.0:
            raise ValueError("sample_time must be finite and positive")

        self.g = 9.80665

        # Nominal motor and mechanical parameters.
        self.km = 97.5e-3

        self.m1 = 122.2e-3
        self.J1zz = 11270.84e-9
        self.l1 = 0.027248
        self.L1 = 0.09 - 9e-3 / 2.0
        self.b1 = 1e-6

        self.m2 = 28e-3
        self.r2 = 9e-3 / 2.0
        self.L2 = 161e-3
        self.l2 = self.L2 / 2.0
        self.b2 = 4.4e-5

        self.x = np.zeros(4, dtype=np.float64)
        self.J2xx = 0.5 * self.m2 * self.r2**2
        self.J2yy = (
            0.25 * self.m2 * self.r2**2
            + (1.0 / 12.0) * self.m2 * self.L2**2
        )
        self.J2zz = self.J2yy

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
