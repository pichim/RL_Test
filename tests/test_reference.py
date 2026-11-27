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

from config import ENCODER, REFERENCE, Evaluation, Training, configure_runtime
configure_runtime()

import numpy as np
import torch
from stable_baselines3 import SAC

from evaluate import capture_panel, evaluate_rates, load_task, nominal_start, plot_episode, rollout
from furuta_env import Encoder, FurutaSwingUpEnv, matlab_reward, unsafe
from train import train
from train_seeds import DECISIONS, available_cores, run_seed
from guided_sac import (
    EnergyShaping, LQRController, PeriodicValidation, REFLECTION, ReflectedPolicy,
    lqr_observation_limits, teacher_examples, validation_score,
)

REFERENCE_TASK = load_task(REFERENCE / "model.zip")


class ReferenceTests(unittest.TestCase):
    def test_encoder_wrap_and_filter_match_the_backward_difference(self):
        settings = replace(ENCODER, arm_noise_std_rad=0, pendulum_noise_std_rad=0)
        encoder = Encoder(settings, REFERENCE_TASK)
        state = np.array([np.pi - .001, -np.pi + .001, 0.0, 0.0])
        encoder.reset(state, seed=11)
        state[:2] = [-np.pi + .001, np.pi - .001]
        encoder.advance(state, 1)
        np.testing.assert_allclose(encoder.velocity, encoder.velocity_gain * np.array([.002, -.002]), atol=1e-10)
        encoder.advance(state, 2)
        np.testing.assert_allclose(encoder.velocity, encoder.alpha * encoder.velocity_gain * np.array([.002, -.002]),
                                   atol=1e-10)

    def test_encoder_noise_preserves_seeded_physics_and_previous_command(self):
        clean = FurutaSwingUpEnv(REFERENCE_TASK)
        measured = FurutaSwingUpEnv(REFERENCE_TASK, encoder_settings=ENCODER)
        normal, _ = clean.reset(seed=42)
        paired, _ = measured.reset(seed=42)
        np.testing.assert_array_equal(normal, paired["clean"])
        self.assertFalse(np.array_equal(paired["measured"], paired["clean"]))
        repeated, _ = measured.reset(seed=42)
        np.testing.assert_array_equal(repeated["measured"], paired["measured"])
        baseline, reward, terminated, truncated, _ = clean.step([.25])
        obs, new_reward, new_term, new_trunc, _ = measured.step([.25])
        np.testing.assert_array_equal(obs["clean"], baseline)
        self.assertEqual(obs["measured"][-1], .25)
        self.assertEqual((new_reward, new_term, new_trunc), (reward, terminated, truncated))

    def test_actor_and_both_critic_paths_consume_only_their_own_channel(self):
        env = FurutaSwingUpEnv(REFERENCE_TASK, encoder_settings=ENCODER)
        model = SAC(ReflectedPolicy, env, seed=0, device="cpu",
                    policy_kwargs={"net_arch": {"pi": [8], "qf": [8]}})
        a = torch.zeros((3, 7))
        b = torch.full_like(a, .5)
        action = torch.full((3, 1), .25)
        obs = {"measured": a, "clean": a}
        with torch.inference_mode():
            pi = model.actor(obs, deterministic=True)
            torch.testing.assert_close(pi, model.actor({"measured": a, "clean": b}, deterministic=True), rtol=0, atol=0)
            self.assertFalse(torch.equal(pi, model.actor({"measured": b, "clean": a}, deterministic=True)))
            for critic in (model.critic, model.critic_target):
                values = critic(obs, action)
                other_measurement = {"measured": b, "clean": a}
                torch.testing.assert_close(values, critic(other_measurement, action), rtol=0, atol=0)
                torch.testing.assert_close(critic.q1_forward(obs, action), critic.q1_forward(other_measurement, action),
                                           rtol=0, atol=0)
                self.assertFalse(torch.equal(values[0], critic({"measured": a, "clean": b}, action)[0]))

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

    def test_training_rejects_plant_step_that_cannot_evaluate_at_500hz(self):
        task = replace(REFERENCE_TASK, sample_time=.00125, action_repeat=4)
        encoder = replace(ENCODER, sample_time=task.sample_time)
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            with self.assertRaisesRegex(ValueError, "500 Hz must span a whole number"):
                train(run, task, encoder=encoder)
            self.assertFalse(run.exists())

    def test_seed_worker_pins_cpu_and_overrides_only_seed_and_budget(self):
        settings = Training(decisions=125_000, gamma=0.995)
        with (
            patch("train_seeds.TRAINING", settings),
            patch("train_seeds.os.sched_setaffinity") as pin,
            patch("train.main") as launch,
            patch("builtins.print"),
        ):
            run_seed(3, 9)
        pin.assert_called_once_with(0, {9})
        self.assertEqual(DECISIONS, 550_000)
        launch.assert_called_once_with(replace(settings, seed=3, decisions=DECISIONS))

    def test_seed_launcher_uses_distinct_physical_cores_within_cpu_affinity(self):
        siblings = {"cpu0": "0,8", "cpu8": "0,8", "cpu9": "1,9", "cpu12": "4,12"}

        def sibling_list(path):
            return siblings[path.parts[-3]] + "\n"

        with (
            patch("train_seeds.os.sched_getaffinity", return_value={0, 8, 9, 12}),
            patch("train_seeds.Path.read_text", autospec=True, side_effect=sibling_list),
        ):
            self.assertEqual(available_cores(), (0, 9, 12))

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

    def test_continuing_mode_pays_once_even_if_balance_is_lost_and_recaptured(self):
        task = replace(REFERENCE_TASK, episode_time=.030, balance_hold_time=.010)
        env = PeriodicValidation(FurutaSwingUpEnv(task, continue_after_success=True), .99)
        upright = np.array([0.0, np.pi, 0.0, 0.0])
        outside = np.array([0.30, np.pi, 0.0, 0.0])
        env.reset(options={"initial_state": upright})
        awarded = []
        with patch.object(env.unwrapped.plant, "step"):
            for state in (upright, upright, outside, upright, upright, outside):
                env.unwrapped.plant.x = state.copy()
                _, reward, terminated, truncated, info = env.step([0.0])
                self.assertFalse(terminated)
                awarded.append(round(reward - matlab_reward(state, 0.0, 0.0, task)))
                self.assertEqual(truncated, len(awarded) == 6)
        self.assertEqual(awarded, [0, 100, 0, 0, 0, 0])
        metrics = info["validation_metrics"]
        self.assertTrue(metrics["hold_lost_by_end"])
        self.assertTrue(metrics["timeout"])
        self.assertFalse(metrics["balanced"])
        self.assertAlmostEqual(metrics["longest_hold_s"], .010)

    def test_validation_measures_completion_of_the_final_uninterrupted_hold(self):
        task = replace(REFERENCE_TASK, episode_time=.030, balance_hold_time=.010)
        env = PeriodicValidation(FurutaSwingUpEnv(task, continue_after_success=True), .99)
        upright = np.array([0.0, np.pi, 0.0, 0.0])
        env.reset(options={"initial_state": upright})
        with patch.object(env.unwrapped.plant, "step"):
            for i in range(6):
                env.unwrapped.plant.x = upright.copy()
                if i == 2:
                    env.unwrapped.plant.x[0] = .30
                _, _, terminated, truncated, info = env.step([0.0])
        self.assertTrue(truncated and not terminated)
        metrics = info["validation_metrics"]
        self.assertTrue(metrics["balanced"])
        self.assertFalse(metrics["timeout"] or metrics["hold_lost_by_end"])
        self.assertAlmostEqual(metrics["balance_finish_s"], .025)
        self.assertAlmostEqual(metrics["longest_hold_s"], .015)

    def test_checkpoint_ties_use_safety_then_balance_time(self):
        hold = {"balanced": True, "unsafe": False, "balance_finish_s": 2.5}
        slow = dict(hold, balance_finish_s=3.0)
        unsafe = {"balanced": False, "unsafe": True, "balance_finish_s": None}
        timeout = dict(unsafe, unsafe=False)
        self.assertGreater(validation_score([hold, timeout]), validation_score([hold, unsafe]))
        self.assertGreater(validation_score([hold, timeout]), validation_score([slow, timeout]))
        self.assertGreater(validation_score([hold, unsafe]), validation_score([timeout, timeout]))

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
