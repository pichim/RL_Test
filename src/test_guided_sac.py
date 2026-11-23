"""Regression checks for guided SAC's symmetry and reward boundaries."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from capture_recovery import recovery_examples
from furuta_env import (
    PERIODIC_LEGACY_REWARD_TASK_VERSION, PERIODIC_STEP_COST, PERIODIC_TASK_VERSION, PERIODIC_UNSAFE_PENALTY,
    FurutaSwingUpEnv, balanced_state_ready, matlab_reward, unsafe,
)
from guided_sac import (
    DeterministicActor, EnergyShaping, LOCAL_EXPLORATION_STD, REFLECTION,
    PeriodicValidation, PeriodicValidationCallback, ReflectedPolicy, lqr_observation_limits, teacher_boundary_examples,
    teacher_bounds, teacher_examples,
)
from nominal import CONFIG, run


class EnergyFixture(gym.Env):
    def reset(self, **kwargs):
        return None, {"energy_error": 1.0}

    def step(self, action):
        terminated, truncated = action
        return None, 2.0, terminated, truncated, {"energy_error": 0.5}


class GuidedSACTests(unittest.TestCase):
    def test_exact_actor_and_critic_reflection(self):
        obs_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        policy = ReflectedPolicy(obs_space, act_space, lambda _: 0.001, net_arch=[8])
        rng = torch.Generator().manual_seed(1)
        observation = torch.rand(8, 7, generator=rng) * 2 - 1
        action = torch.rand(8, 1, generator=rng) * 2 - 1
        mirror = observation * torch.tensor(REFLECTION)
        mean, scale, _ = policy.actor.get_action_dist_params(observation)
        mirror_mean, mirror_scale, _ = policy.actor.get_action_dist_params(mirror)
        torch.testing.assert_close(mean, -mirror_mean, rtol=0, atol=0)
        torch.testing.assert_close(scale, mirror_scale, rtol=0, atol=0)
        torch.testing.assert_close(
            policy.critic(observation, action), policy.critic(mirror, -action), rtol=0, atol=0,
        )
        equilibrium = torch.tensor([[0., 1., 0., 0., 1., 0., 0.]])
        self.assertEqual(policy.actor(equilibrium, deterministic=True).item(), 0.0)

    def test_energy_potential_terminal_and_timeout(self):
        for terminated, truncated in ((False, False), (True, False), (False, True)):
            with self.subTest(terminated=terminated, truncated=truncated):
                env = EnergyShaping(EnergyFixture(), gamma=0.99, weight=0.25)
                env.reset()
                _, reward, *_ = env.step((terminated, truncated))
                successor = 0.0 if terminated else -0.25 * 0.25 / 1.25
                self.assertAlmostEqual(reward, 2.0 + 0.99 * successor + 0.125)

    def test_teacher_samples_are_local_and_current_limited(self):
        observations, actions = teacher_examples(CONFIG, 0)
        self.assertGreater(len(actions), 192)
        self.assertEqual(observations.shape, (len(actions), 7))
        self.assertTrue(np.isfinite(observations).all())
        self.assertTrue((np.abs(actions) <= 1.0).all())
        # The early samples deliberately vary the previous command.
        self.assertGreater(np.std(observations[:, -1]), 0.01)

    def test_guided_one_amp_preserves_penalties_and_command_units(self):
        with (
            TemporaryDirectory() as directory,
            patch("nominal.cuda_preflight", return_value=None),
            patch("nominal.train_guided", side_effect=RuntimeError("capture config")) as train,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture config"):
                run(Path(directory) / "pilot", execute=True, guided=True, max_current=1.0)
            config = train.call_args.args[0]
        self.assertEqual(config.max_current, 1.0)
        self.assertEqual(config.current_weight, 4.0)
        self.assertEqual(config.action_change_weight, 12.0)
        env = FurutaSwingUpEnv(config)
        for expected in (-1.0, 1.0):
            env.reset(options={"initial_state": [0.0, np.pi, 0.0, 0.0]})
            _, _, _, _, info = env.step(np.array([expected], dtype=np.float32))
            self.assertEqual(info["command_current"], expected)
            self.assertEqual(info["current"], expected)
        np.testing.assert_array_equal(teacher_bounds(config), [0.2, 0.1, 0.5, 1.0])
        upright = np.array([0.0, np.pi, 0.0, 0.0])
        baseline = matlab_reward(upright, 0.8, -0.4, CONFIG)
        one_amp = matlab_reward(upright, 0.4, -0.2, config)
        self.assertAlmostEqual(baseline, one_amp)

    def test_small_actor_options_leave_critics_and_guidance_unchanged(self):
        with (
            TemporaryDirectory() as directory,
            patch("nominal.cuda_preflight", return_value=None),
            patch("nominal.train_guided", side_effect=RuntimeError("capture setup")) as train,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture setup"):
                run(
                    Path(directory) / "pilot", execute=True, guided=True,
                    max_current=1.0, timesteps=500000, actor_network=[8, 8],
                )
        self.assertEqual(train.call_args.kwargs["actor_network"], [8, 8])
        self.assertEqual(train.call_args.kwargs["critic_network"], [64, 64])
        self.assertEqual(train.call_args.kwargs["energy_weight"], 0.25)
        self.assertEqual(train.call_args.kwargs["total_timesteps"], 500000)

    def test_actor_can_drop_previous_command_without_changing_critics(self):
        obs_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        policy = ReflectedPolicy(
            obs_space, act_space, lambda _: 0.001,
            net_arch={"pi": [8, 8], "qf": [64, 64]},
            actor_observes_previous_command=False,
        )
        self.assertEqual(policy.actor.latent_pi[0].in_features, 6)
        self.assertEqual(policy.critic.q_networks[0][0].in_features, 8)
        self.assertEqual(policy.critic.q_networks[0][0].out_features, 64)
        obs = torch.rand(8, 7, generator=torch.Generator().manual_seed(1))
        changed = obs.clone()
        changed[:, -1] *= -1
        mean, scale, _ = policy.actor.get_action_dist_params(obs)
        changed_mean, changed_scale, _ = policy.actor.get_action_dist_params(changed)
        torch.testing.assert_close(mean, changed_mean)
        torch.testing.assert_close(scale, changed_scale)
        actor = DeterministicActor(policy.actor)
        self.assertEqual(sum(p.numel() for p in actor.parameters()), 137)
        torch.testing.assert_close(actor(obs[:, :6]), policy.actor(obs, deterministic=True), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            actor(obs[:, :6] * torch.tensor(REFLECTION[:6])), -actor(obs[:, :6]), atol=1e-6, rtol=0,
        )

    def test_current_weight_variant_sets_physical_coefficients(self):
        with (
            TemporaryDirectory() as directory,
            patch("nominal.cuda_preflight", return_value=None),
            patch("nominal.train_guided", side_effect=RuntimeError("capture config")) as train,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture config"):
                run(
                    Path(directory) / "pilot", execute=True, guided=True, max_current=1.0,
                    current_weight=0.75, action_change_weight=3.75,
                )
        config = train.call_args.args[0]
        upright = np.array([0.0, np.pi, 0.0, 0.0])
        self.assertEqual(config.current_weight, 3.0)
        self.assertEqual(config.action_change_weight, 15.0)
        self.assertAlmostEqual(1 - matlab_reward(upright, 0.5, 0.5, config), 0.30 * 0.5**2)
        self.assertAlmostEqual(
            1 - matlab_reward(upright, 0.5, 0.3, config),
            0.30 * 0.5**2 + 1.50 * 0.2**2,
        )

    def test_lqr_region_masks_direct_sac_actor_gradients_but_keeps_teacher_gradients(self):
        config = replace(CONFIG, max_current=1.0)
        obs_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        policy = ReflectedPolicy(
            obs_space, act_space, lambda _: 0.001, net_arch=[16, 16],
            lqr_observation_limits=lqr_observation_limits(config),
        )
        local = torch.tensor([0.06, np.cos(0.06), 0.3/30, 0.04, np.cos(0.04), 0.6/30, 0.1])
        local = local.to(torch.float32)
        outside = local.clone()
        outside[5] = 1.1/30
        both = torch.stack((local, outside, local * torch.tensor(REFLECTION)))
        self.assertEqual(policy.actor.inside_lqr_region(both).tolist(), [True, False, True])
        _, log_std, _ = policy.actor.get_action_dist_params(both)
        torch.testing.assert_close(log_std[[0, 2]].exp(), torch.full((2, 1), LOCAL_EXPLORATION_STD))
        actions, log_prob = policy.actor.action_log_prob(both)
        parameters = tuple(policy.actor.parameters())
        local_gradient = torch.autograd.grad(
            actions[[0, 2]].sum() + log_prob[[0, 2]].sum(), parameters,
            retain_graph=True, allow_unused=True,
        )
        self.assertTrue(all(g is None or not torch.any(g) for g in local_gradient))
        outside_gradient = torch.autograd.grad(
            actions[1].sum() + log_prob[1], parameters, allow_unused=True,
        )
        self.assertTrue(any(g is not None and torch.any(g) for g in outside_gradient))
        teacher_gradient = torch.autograd.grad(
            policy.actor(local.unsqueeze(0), deterministic=True).sum(), parameters, allow_unused=True,
        )
        self.assertTrue(any(g is not None and torch.any(g) for g in teacher_gradient))

    def test_wide_reset_speeds_and_lqr_split_pass_through_guided_launch(self):
        with (
            TemporaryDirectory() as directory,
            patch("nominal.cuda_preflight", return_value=None),
            patch("nominal.train_guided", side_effect=RuntimeError("capture config")) as train,
        ):
            with self.assertRaisesRegex(RuntimeError, "capture config"):
                run(
                    Path(directory) / "job", execute=True, guided=True,
                    lqr_only_inside=True, reset_speed_rad_s=6.0, full_arm_travel=True,
                )
        config = train.call_args.args[0]
        self.assertEqual(config.reset_omega1_half_range, 6.0)
        self.assertEqual(config.reset_omega2_half_range, 6.0)
        self.assertEqual(config.randomized_reset_omega1_half_range, 6.0)
        self.assertEqual(config.randomized_reset_omega2_half_range, 6.0)
        self.assertEqual(config.reset_theta1_half_range, np.pi)
        self.assertEqual(config.arm_angle_limit, np.pi)
        self.assertFalse(unsafe(np.array([np.pi, 0.0, 6.0, 6.0]), config))
        self.assertTrue(unsafe(np.array([np.pi + 1e-4, 0.0, 0.0, 0.0]), config))
        self.assertTrue(train.call_args.kwargs["lqr_only_inside"])


    def test_periodic_pilot_crosses_the_arm_limit_and_scores_full_turns(self):
        config = replace(CONFIG, task_version=PERIODIC_TASK_VERSION, reset_theta1_half_range=np.pi)
        env = FurutaSwingUpEnv(config)
        for direction in (-1, 1):
            env.reset(options={"initial_state": [direction * (np.pi - 0.001), np.pi, direction * 6, 0]})
            _, _, terminated, truncated, info = env.step([0.0])
            self.assertFalse(terminated or truncated or info["unsafe"])
            self.assertGreater(abs(info["state"][0]), np.pi)

        upright = np.array([0.0, np.pi, 0.0, 0.0])
        equivalent = upright + [2 * np.pi, -2 * np.pi, 0, 0]
        self.assertFalse(balanced_state_ready(equivalent, 0.0, CONFIG))
        self.assertTrue(balanced_state_ready(equivalent, 0.0, config))
        self.assertAlmostEqual(matlab_reward(upright, 0, 0, config), -PERIODIC_STEP_COST)
        self.assertAlmostEqual(matlab_reward(equivalent, 0, 0, config), -PERIODIC_STEP_COST)
        self.assertTrue(unsafe(np.array([0, np.pi, 31, 0]), config))
        env.reset(options={"initial_state": [0, np.pi, 0, 0]})
        with patch.object(env.plant, "step_filtered_current", side_effect=RuntimeError):
            _, reward, terminated, truncated, info = env.step([0.0])
        self.assertTrue(terminated and not truncated and info["unsafe"])
        self.assertAlmostEqual(reward, -PERIODIC_UNSAFE_PENALTY - PERIODIC_STEP_COST)
        env.close()

    def test_periodic_validation_records_terminal_metrics(self):
        config = replace(CONFIG, task_version=PERIODIC_TASK_VERSION, balance_hold_time=0.005)
        env = PeriodicValidation(FurutaSwingUpEnv(config), gamma=0.9975)
        env.reset(options={"initial_state": [2 * np.pi, np.pi, 0.0, 0.0]})
        _, reward, term, trunc, info = env.step([0.0])
        self.assertTrue(term and not trunc)
        self.assertEqual(info["validation_metrics"], {
            "balanced": True, "captured": True, "genuine_swingup": False,
            "entered_teacher_box": True, "balanced_after_teacher_entry": True,
            "unsafe": False, "timeout": False, "discounted_reward": reward,
        })
        env.close()

    def test_boundary_examples_pair_previous_commands_with_the_same_target(self):
        observations, target = teacher_boundary_examples(replace(CONFIG, max_current=1.0))
        self.assertEqual(observations.shape, (400, 7))
        batch = observations.reshape(80, 5, 7)
        np.testing.assert_allclose(
            batch[..., :6], np.broadcast_to(batch[:, :1, :6], batch[..., :6].shape),
        )
        np.testing.assert_allclose(
            batch[..., 6], np.tile([-1.0, -0.5, 0.0, 0.5, 1.0], (80, 1)),
        )
        actions = target.reshape(80, 5)
        np.testing.assert_allclose(actions, np.broadcast_to(actions[:, :1], actions.shape))


    def test_periodic_scratch_preserves_the_prior_reward_and_wraps_the_arm(self):
        config = replace(CONFIG, task_version=PERIODIC_LEGACY_REWARD_TASK_VERSION)
        nominal = np.array([0.15, np.pi + 0.04, 0.24, -0.1])
        full_turn = nominal + [2 * np.pi, -2 * np.pi, 0, 0]
        expected = matlab_reward(nominal, 0.3, -0.1, CONFIG)
        self.assertAlmostEqual(matlab_reward(full_turn, 0.3, -0.1, config), expected)
        self.assertFalse(unsafe(full_turn, config))
        self.assertTrue(unsafe(full_turn, CONFIG))
        self.assertTrue(unsafe(full_turn + [0, 0, 31, 0], config))

    def test_capture_recovery_demos_are_local_approaches_with_command_history(self):
        config = replace(CONFIG, task_version=PERIODIC_LEGACY_REWARD_TASK_VERSION, max_current=1.0)
        with patch("capture_recovery.RECOVERY_CANDIDATES", 4):
            obs, actions, metadata = recovery_examples(config, 0, teacher_bounds(config))
        bounds = teacher_bounds(config)
        inside = (
            (obs[:, 1] >= np.cos(bounds[0])) & (obs[:, 4] >= np.cos(bounds[1]))
            & (np.abs(obs[:, 2]) <= bounds[2] / config.max_abs_omega1)
            & (np.abs(obs[:, 5]) <= bounds[3] / config.max_abs_omega2)
        )
        self.assertGreater(metadata["recovered_starts_used"], 0)
        self.assertEqual(obs.shape, (metadata["demonstration_transitions"], 7))
        self.assertTrue(inside[-1])
        self.assertFalse(inside[0])
        self.assertEqual(int(sum(inside)), metadata["recovered_starts_used"])
        index = np.flatnonzero(~inside[:-1]) + 1
        np.testing.assert_allclose(obs[index, -1], actions[index - 1, 0])
        self.assertTrue(np.all(np.abs(actions) <= 1.0))

    def test_scratch_checkpoint_ranks_balance_then_recovery(self):
        callback = PeriodicValidationCallback.__new__(PeriodicValidationCallback)
        callback.rank_recovery = True
        callback.last_mean_reward = 10.0
        callback.episodes = [{"balanced_after_teacher_entry": True}, {"balanced_after_teacher_entry": False}]
        earlier = callback.validation_score(.5)
        callback.last_mean_reward = 50.0
        callback.episodes[0]["balanced_after_teacher_entry"] = False
        self.assertLess(callback.validation_score(.5), earlier)
        self.assertGreater(callback.validation_score(1.0), earlier)
