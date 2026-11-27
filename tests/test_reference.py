"""Focused checks for the single nominal training and evaluation path."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import REFERENCE, Evaluation, Training, configure_runtime
configure_runtime()

import numpy as np
import torch
from stable_baselines3 import SAC

from evaluate import capture_panel, evaluate_rates, load_task, nominal_start, plot_episode, rollout
from furuta_env import FurutaSwingUpEnv, matlab_reward, unsafe
from train import train
from train_seeds import SEEDS, main as train_seeds
from guided_sac import (
    EnergyShaping, LQRController, REFLECTION, ReflectedPolicy, lqr_observation_limits, teacher_examples,
)

REFERENCE_TASK = load_task(REFERENCE / "model.zip")


class ReferenceTests(unittest.TestCase):
    def test_both_evaluations_use_the_same_checkpoint_seeds_and_capture_starts(self):
        model = REFERENCE / "model.zip"
        settings = Evaluation(episodes=2, base_seed=123)
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            panel = output / "capture_snapshot.csv"
            panel.touch()
            with patch("evaluate.SAC.load") as load, patch("evaluate._evaluate_rate") as evaluate, patch("builtins.print"):
                evaluate_rates(model, output, settings, capture_starts=panel)
        load.assert_called_once_with(model, device="cpu")
        self.assertEqual(evaluate.call_args_list, [
            call(load.return_value, model, REFERENCE_TASK, output / "evaluation", settings, panel),
            call(load.return_value, model, replace(REFERENCE_TASK, action_repeat=2),
                 output / "evaluation_500hz", settings, panel),
        ])

    def test_incompatible_plant_step_fails_before_starting_either_evaluation(self):
        model = REFERENCE / "model.zip"
        task = replace(REFERENCE_TASK, sample_time=0.005, action_repeat=1)
        with (
            patch("evaluate.load_task", return_value=task),
            patch("evaluate.SAC.load") as load,
            self.assertRaisesRegex(ValueError, "500 Hz"),
        ):
            evaluate_rates(model, Path("results"))
        load.assert_not_called()

    def test_post_capture_with_too_little_time_to_balance_is_scored(self):
        class Zero:
            @staticmethod
            def predict(_observation, deterministic=True):
                return [0.0], None

        with TemporaryDirectory() as tmp:
            panel = Path(tmp) / "starts.csv"
            panel.write_text(
                "seed,theta1_rad,theta2_rad,omega1_rad_s,omega2_rad_s,time_remaining_s,previous_command_A\n"
                f"0,0,{np.pi},0,0,0.01,0\n",
                encoding="utf-8",
            )
            result = capture_panel(Zero(), REFERENCE_TASK, panel)
        self.assertEqual((result["starts"], result["captured"], result["balanced"], result["unsafe"]), (1, 1, 0, 0))

    def test_training_rejects_500hz(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "200 Hz"):
                train(directory, replace(REFERENCE_TASK, action_repeat=2))
            self.assertFalse(directory.exists())

    def test_seed_launcher_changes_only_the_seed(self):
        settings = Training(decisions=125_000, gamma=0.995)
        with (
            patch("train_seeds.TRAINING", settings),
            patch("train_seeds.train_and_evaluate") as launch,
            patch("builtins.print"),
        ):
            train_seeds()
        self.assertEqual(launch.call_args_list, [call(replace(settings, seed=seed)) for seed in SEEDS])

    def test_integration_failure_removes_the_safe_reward(self):
        task = replace(REFERENCE_TASK, constraint_reward_weight=0.4)
        env = FurutaSwingUpEnv(task)
        _, initial = env.reset(options={"initial_state": [0, np.pi, 0, 0]})
        expected = matlab_reward(initial["state"], 0.0, 0.0, task) - task.constraint_reward_weight
        with patch.object(env.plant, "step", side_effect=RuntimeError):
            _, reward, terminated, truncated, info = env.step([0.0])
        self.assertAlmostEqual(reward, expected)
        self.assertTrue(terminated and not truncated and info["unsafe"])
        np.testing.assert_array_equal(info["state"], initial["state"])

    def test_500hz_holds_balance_for_exactly_one_second(self):
        task = replace(REFERENCE_TASK, action_repeat=2)
        env = FurutaSwingUpEnv(task)
        env.reset(options={"initial_state": [0, np.pi, 0, 0]})
        for _ in range(499):
            _, _, terminated, truncated, _ = env.step([0.0])
            self.assertFalse(terminated or truncated)
        _, _, terminated, truncated, info = env.step([0.0])
        self.assertTrue(terminated and not truncated and info["is_success"])
        self.assertEqual(env.plant_step_count, 1000)

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

    def test_current_change_starts_at_previous_command(self):
        class Hold:
            @staticmethod
            def predict(observation, deterministic=True):
                return [observation[-1]], None

        task = replace(REFERENCE_TASK, max_current=2.0, episode_time=.01, balance_hold_time=.005)
        result, trace = rollout(Hold(), task, [0, 0, 0, 0], previous_action=0.25)
        self.assertEqual(trace[0]["current_A"], 0.5)
        self.assertEqual(result["current_change_rms_A"], 0.0)

    def test_slew_uses_the_control_period(self):
        class Hold:
            @staticmethod
            def predict(_observation, deterministic=True):
                return [0.25], None

        task = replace(REFERENCE_TASK, action_repeat=2, episode_time=.002, balance_hold_time=.001)
        result, _ = rollout(Hold(), task, [0, 0, 0, 0])
        self.assertEqual(result["current_change_rms_A"], 0.25)
        self.assertEqual(result["current_slew_rms_A_per_s"], 125.0)

    def test_training_stops_at_a_partial_final_control_step(self):
        task = replace(REFERENCE_TASK, episode_time=.006, balance_hold_time=.002)
        env = FurutaSwingUpEnv(task)
        env.reset(options={"initial_state": [0, 0, 0, 0]})
        self.assertFalse(env.step([0.0])[3])
        _, _, terminated, truncated, _ = env.step([0.0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(env.plant_step_count, 6)

    def test_lqr_uses_periodic_arm_error(self):
        teacher = LQRController(REFERENCE_TASK.max_current)
        state = np.array([0.1, np.pi + 0.02, 0.1, -0.2])
        expected = teacher.current(state)
        for turns in (-2, -1, 1, 2):
            self.assertAlmostEqual(teacher.current(state + [turns * 2 * np.pi, 0, 0, 0]), expected)

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
