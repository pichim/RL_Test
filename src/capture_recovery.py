"""Generate held-current recovery demonstrations and score saved capture starts."""

import csv
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from controllers import LQRController
from evaluate import run_prepared_episode
from furuta_env import FurutaSwingUpEnv, arm_position_error, observation_from_plant, upright_error
from furuta_model import FurutaPendulum


RECOVERY_CANDIDATES = 192
RECOVERY_SEED_STREAM = 20261123
STATE_COLUMNS = ("theta1_rad", "theta2_rad", "omega1_rad_s", "omega2_rad_s")


def recovery_examples(config, seed, teacher_bounds):
    """Label the approach to the local box on verified LQR recoveries."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, RECOVERY_SEED_STREAM]))
    start_bounds = np.array([2.5, config.capture_angle, 1.30 * config.capture_omega1,
                             1.10 * config.capture_omega2])
    teacher = LQRController(config.max_current)
    env = FurutaSwingUpEnv(config)
    observations, targets = [], []
    kept_count = 0
    try:
        for error in rng.uniform(-start_bounds, start_bounds, size=(RECOVERY_CANDIDATES, 4)):
            if np.all(np.abs(error) <= teacher_bounds):
                continue
            initial_state = error + [0.0, np.pi, 0.0, 0.0]
            _, info = env.reset(options={"initial_state": initial_state})
            previous = float(rng.uniform(-1.0, 1.0))
            env.last_normalized_action = previous
            observation = observation_from_plant(env.plant, config, previous)
            route_obs, route_actions = [], []
            entered = False
            while True:
                state = info["state"].copy()
                state[0] = arm_position_error(state[0], 0.0, config)
                command = teacher.current(state) / config.max_current
                error = np.array([state[0], upright_error(state), *state[2:]])
                if not entered:
                    route_obs.append(observation)
                    route_actions.append([command])
                    entered = bool(np.all(np.abs(error) <= teacher_bounds))
                observation, _, terminated, truncated, info = env.step([command])
                if terminated or truncated:
                    break
            if info["is_success"] and entered:
                observations.extend(route_obs)
                targets.extend(route_actions)
                kept_count += 1
    finally:
        env.close()
    if not kept_count:
        raise RuntimeError("The centered LQR did not recover any demonstration starts.")
    return np.asarray(observations), np.asarray(targets, dtype=np.float32), {
        "candidate_starts": RECOVERY_CANDIDATES,
        "recovered_starts_used": kept_count,
        "demonstration_transitions": len(targets),
        "sampling_bounds": start_bounds.tolist(),
        "seed_stream": [seed, RECOVERY_SEED_STREAM],
    }


def load_capture_starts(path: Path, config):
    """Read the fixed evaluation panel from a post-capture diagnostic."""
    starts = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            state = tuple(float(row[key]) for key in STATE_COLUMNS)
            previous = float(row["previous_command_A"]) / config.max_current
            remaining = float(row["time_remaining_s"])
            if (seed < 0 or not all(math.isfinite(value) for value in state)
                or not -1.0 <= previous <= 1.0
                or not config.balance_hold_time <= remaining <= config.episode_time):
                raise ValueError(f"Invalid capture benchmark start for seed {seed}.")
            starts.append((seed, state, previous, remaining))
    if not starts or len(starts) != len({start[0] for start in starts}):
        raise ValueError("The capture benchmark needs a nonempty set of unique seeds.")
    return starts


class _ActorWithPreviousCommand:
    def __init__(self, model, previous):
        self.model = model
        self.previous = previous

    def predict(self, observation, deterministic=True):
        if self.previous is not None:
            observation = observation.copy()
            observation[-1] = self.previous
            self.previous = None
        return self.model.predict(observation, deterministic=deterministic)


def score_capture_starts(model_path: Path, config, starts):
    model = SAC.load(model_path, device="cpu")
    results = []
    for seed, state, previous, remaining in starts:
        plant = FurutaPendulum(config.sample_time)
        plant.reset(np.asarray(state))
        controller = _ActorWithPreviousCommand(model, previous)
        result = run_prepared_episode(controller, plant, replace(config, episode_time=remaining), "sac")
        results.append({
            "seed": seed,
            "balanced": bool(result["success"]),
            "unsafe": bool(result["unsafe"]),
            "hold_finish_s": float(result["finish_time"]) if result["success"] else None,
        })
    return {
        "starts": len(results),
        "balanced": sum(row["balanced"] for row in results),
        "unsafe": sum(row["unsafe"] for row in results),
        "episodes": results,
    }
