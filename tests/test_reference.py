"""Focused checks for the single nominal training and evaluation path."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import REFERENCE, configure_runtime
configure_runtime()

import numpy as np
import torch
from stable_baselines3 import SAC

from evaluate import load_task, nominal_start, plot_episode, rollout
from furuta_env import FurutaSwingUpEnv, matlab_reward, unsafe
from guided_sac import (
    EnergyShaping, REFLECTION, ReflectedPolicy, lqr_observation_limits, teacher_examples,
)

REFERENCE_TASK = load_task(REFERENCE / "model.zip")


class ReferenceTests(unittest.TestCase):
    def test_reset_and_direct_current_transition_match_the_archived_task(self):
        env = EnergyShaping(FurutaSwingUpEnv(REFERENCE_TASK), .99, .25)
        first, info = env.reset(seed=0)
        np.testing.assert_allclose(info["state"], [
            0.8605556614246863, -1.4464727375963786, -5.508317712765663, -5.801668373657651,
        ], atol=1e-14, rtol=0)
        self.assertEqual(first.dtype, np.float32)
        observation, _, terminated, truncated, info = env.step([0.25])
        self.assertEqual(observation[-1], 0.25)
        self.assertEqual(info["command_current"], 0.25)
        self.assertEqual(env.unwrapped.plant_step_count, REFERENCE_TASK.action_repeat)
        self.assertFalse(terminated or truncated)

    def test_crossing_pi_preserves_current_reward_and_speed_guard(self):
        state = np.array([0.12, np.pi + 0.03, 0.10, 0.34])
        equiv = state + [2 * np.pi, -2 * np.pi, 0, 0]
        self.assertFalse(unsafe(equiv, REFERENCE_TASK))
        self.assertTrue(unsafe(equiv + [0, 0, 31, 0], REFERENCE_TASK))
        self.assertAlmostEqual(matlab_reward(equiv, .4, .1, REFERENCE_TASK), matlab_reward(state, .4, .1, REFERENCE_TASK))
        self.assertAlmostEqual(
            matlab_reward(state, .4, .4, REFERENCE_TASK) - matlab_reward(state, .4, .1, REFERENCE_TASK),
            1.5 * (.4 - .1)**2,
        )
        env = FurutaSwingUpEnv(REFERENCE_TASK)
        for sign in (1, -1):
            env.reset(options={"initial_state": [sign * (np.pi - .001), np.pi, sign * 6, 0]})
            _, _, terminated, truncated, info = env.step([0.0])
            self.assertFalse(terminated or truncated)
            self.assertGreater(abs(info["state"][0]), np.pi)

    def test_seeded_lqr_examples_and_region(self):
        observations, actions = teacher_examples(REFERENCE_TASK, 0)
        self.assertEqual(observations.shape, (991, 7))
        self.assertEqual(actions.shape, (991, 1))
        self.assertTrue(np.all(np.abs(actions) <= 1.0))
        self.assertGreater(np.max(np.abs(observations[:, -1])), 0.95)
        arm_cos, pendulum_cos, arm_speed, pendulum_speed = lqr_observation_limits(REFERENCE_TASK)
        self.assertTrue(np.all(observations[:, 1] >= arm_cos))
        self.assertTrue(np.all(observations[:, 4] >= pendulum_cos))
        self.assertTrue(np.all(np.abs(observations[:, 2]) <= arm_speed))
        self.assertTrue(np.all(np.abs(observations[:, 5]) <= pendulum_speed))

    def test_sac_actor_gradient_is_masked_inside_teacher_box(self):
        env = FurutaSwingUpEnv(REFERENCE_TASK)
        policy = ReflectedPolicy(
            env.observation_space, env.action_space, lambda _: 3e-4, net_arch=[16, 16],
            lqr_observation_limits=lqr_observation_limits(REFERENCE_TASK),
        )
        actor = policy.actor
        obs = torch.tensor([[0., 1., 0., 0., 1., 0., 0.], [1., 0., 0., 0., 1., 0., 0.]])
        self.assertEqual(actor.inside_lqr_region(obs).tolist(), [True, False])
        actions, log_prob = actor.action_log_prob(obs)
        (actions[0].sum() + log_prob[0]).backward(retain_graph=True)
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in actor.parameters()))
        actor.zero_grad(set_to_none=True)
        (actions[1].sum() + log_prob[1]).backward()
        self.assertTrue(any(p.grad is not None and p.grad.any() for p in actor.parameters()))

    def test_training_success_and_fixed_horizon_evaluation_are_distinct(self):
        initial = [0.0, np.pi, 0.0, 0.0]
        env = FurutaSwingUpEnv(REFERENCE_TASK)
        env.reset(options={"initial_state": initial})
        for _ in range(199):
            self.assertFalse(env.step([0.0])[2])
        self.assertTrue(env.step([0.0])[2])

        class PushAfterBalance:
            calls = 0

            def predict(self, _obs, deterministic=True):
                self.calls += 1
                return [0.0 if self.calls <= 200 else 1.0], None

        result, trace = rollout(PushAfterBalance(), REFERENCE_TASK, initial)
        self.assertFalse(result["balanced"])
        self.assertIsNotNone(result["first_balance_s"])
        self.assertGreater(trace[-1]["time_s"], 1.0)

    def test_frozen_checkpoint_and_export_agree(self):
        archived = json.loads((REFERENCE / "results.json").read_text())
        for file, digest in archived["sha256"].items():
            self.assertEqual(sha256((REFERENCE / file).read_bytes()).hexdigest(), digest)
        path = REFERENCE / "model.zip"
        task = load_task(path)
        self.assertEqual(task.episode_time, REFERENCE_TASK.episode_time)
        agent = SAC.load(path, device="cpu")
        actor = torch.export.load(REFERENCE / "actor.pt2").module()
        generator = torch.Generator().manual_seed(12)
        obs = torch.rand(32, 7, generator=generator) * 2 - 1
        with torch.no_grad():
            expected, _ = agent.predict(obs.numpy(), deterministic=True)
            np.testing.assert_allclose(actor(obs).numpy(), expected, atol=1e-6)
            torch.testing.assert_close(actor(obs * torch.tensor(REFLECTION)), -actor(obs))
        result, trace = rollout(agent, task, nominal_start(80000, task))
        self.assertTrue(result["balanced"])
        self.assertEqual(trace[-1]["time_s"], 5.0)

    def test_evaluation_figure_has_three_panels(self):
        class Zero:
            @staticmethod
            def predict(_obs, deterministic=True):
                return [0.0], None

        short = replace(REFERENCE_TASK, episode_time=.01, balance_hold_time=.005)
        outcome, trace = rollout(Zero(), short, [0, np.pi, 0, 0])
        with TemporaryDirectory() as tmp, patch("evaluate.plt.subplots", wraps=None) as subplots:
            from matplotlib.figure import Figure
            fig = Figure()
            subplots.return_value = fig, fig.subplots(3, 1)
            plot_episode(Path(tmp) / "episode.png", trace, outcome, REFERENCE_TASK.max_current)
            self.assertEqual(subplots.call_args.args[:2], (3, 1))
            self.assertTrue((Path(tmp) / "episode.png").is_file())


if __name__ == "__main__":
    unittest.main()
