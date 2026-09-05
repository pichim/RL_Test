"""Focused tests for the MATLAB-derived current-controlled Furuta project."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from controllers import HybridSupervisor, LQRController, Mode
from evaluate import (
    DEFAULT_BASE_SEED,
    DEFAULT_EPISODES,
    MODEL_PATH,
    RESULTS_DIR,
    load_run_config,
    main as evaluate_main,
    make_episode,
    maximum_available,
    percentile_available,
    run_episode,
    trajectory_shape_metrics,
)
from furuta_env import (
    DEFAULT_CONFIG,
    TASK_VERSION,
    FurutaConfig,
    FurutaSwingUpEnv,
    balanced_state_ready,
    capture_ready,
    lqr_handover_current,
    matlab_reward,
    mechanical_capture_ready,
    nominal_pendulum_energy_error,
    observation_from_plant,
    sample_action_dead_time_samples,
    sample_current_filter_cutoff_hz,
    sample_matlab_initial_state,
    sample_parameter_scales,
    unsafe,
    upright_error,
    within_matlab_constraints,
    wrap_angle,
)
from furuta_model import FurutaPendulum
from train import (
    algorithm_settings,
    atomic_write_json,
    curriculum_scale,
    find_run_file,
    resolve_resume_model,
    resolve_resume_replay_buffer,
    save_committed_state,
    validate_curriculum_config,
    validate_resume_config,
)
from reproduce import (
    DEFAULT_RECIPE,
    build_stage_command,
    load_recipe,
    stage_run_dir,
)
from qualify import (
    MODEL_PATH as QUALIFICATION_MODEL_PATH,
    build_suite_command,
    verify_frozen_evidence,
    verify_release,
)
from stress_evaluate import (
    corner_cases,
    long_duration_cases,
    parameter_profiles,
    prepare_plant,
    support_boundary_states,
)


class FurutaContinuousTests(unittest.TestCase):
    def test_matlab_timing_and_training_settings(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.task_version, TASK_VERSION)
        self.assertAlmostEqual(DEFAULT_CONFIG.sample_time, 1.0 / 1_000.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.balance_hold_time, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.constraint_reward_weight, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.arm_angle_weight, 1.5)
        self.assertAlmostEqual(DEFAULT_CONFIG.arm_velocity_weight, 0.015)
        self.assertAlmostEqual(DEFAULT_CONFIG.velocity_weight, 0.010)
        self.assertAlmostEqual(DEFAULT_CONFIG.action_change_weight, 3.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.current_filter_cutoff_hz, 0.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.current_filter_cutoff_min_hz, 0.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.current_filter_cutoff_max_hz, 0.0)
        self.assertFalse(DEFAULT_CONFIG.training_parameter_randomization)
        self.assertAlmostEqual(DEFAULT_CONFIG.arm_mass_randomization, 0.10)
        self.assertAlmostEqual(DEFAULT_CONFIG.pendulum_mass_randomization, 0.10)
        self.assertAlmostEqual(DEFAULT_CONFIG.arm_damping_randomization, 0.10)
        self.assertAlmostEqual(
            DEFAULT_CONFIG.pendulum_damping_randomization,
            0.10,
        )
        self.assertAlmostEqual(
            DEFAULT_CONFIG.sample_time * DEFAULT_CONFIG.action_repeat,
            1.0 / 200.0,
        )
        settings = algorithm_settings(DEFAULT_CONFIG, total_timesteps=1_000_000)
        self.assertEqual(settings["steps_per_episode"], 1_000)
        self.assertEqual(settings["total_timesteps"], 1_000_000)
        self.assertEqual(settings["learning_starts"], 40_000)
        self.assertEqual(settings["buffer_size"], 1_000_000)
        self.assertEqual(settings["batch_size"], 256)
        self.assertAlmostEqual(settings["learning_rate"], 3e-4)
        self.assertAlmostEqual(settings["gamma"], 0.99)
        self.assertAlmostEqual(settings["tau"], 0.005)
        self.assertEqual(settings["train_frequency"], 2)
        self.assertEqual(settings["target_update_interval"], 1)
        self.assertEqual(settings["entropy_coefficient"], "auto_0.1")
        self.assertEqual(settings["evaluation_seed"], 10_000)
        self.assertEqual(
            settings["best_model_selection"],
            "success_rate_then_longer_episode_then_mean_reward",
        )
        self.assertEqual(settings["actor_network"], [8, 8])
        self.assertEqual(settings["critic_network"], [64, 64])
        self.assertEqual(settings["initialization"], "from_scratch")
        self.assertIsNone(settings["initial_actor"])
        self.assertIsNone(settings["resume_model"])
        self.assertIsNone(settings["resume_replay_buffer"])
        self.assertEqual(settings["resume_snapshot_frequency"], 100_000)
        self.assertEqual(settings["resume_milestone_timesteps"], [])
        with self.assertRaisesRegex(ValueError, "must be even"):
            algorithm_settings(DEFAULT_CONFIG, total_timesteps=999_999)

    def test_full_resume_settings_and_paired_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_path = run_dir / "model.zip"
            replay_path = run_dir / "replay_buffer.pkl"
            model_path.touch()
            replay_path.touch()
            (run_dir / "config.json").write_text(
                json.dumps(asdict(DEFAULT_CONFIG)),
                encoding="utf-8",
            )
            (run_dir / "training.json").write_text(
                json.dumps({"initialization": "from_scratch"}),
                encoding="utf-8",
            )

            settings = algorithm_settings(
                DEFAULT_CONFIG,
                total_timesteps=50_000,
                resume_model_path=model_path,
                resume_replay_buffer_path=replay_path,
            )
            self.assertEqual(settings["initialization"], "full_resume")
            self.assertEqual(settings["resume_model"], str(model_path.resolve()))
            self.assertEqual(
                settings["resume_replay_buffer"],
                str(replay_path.resolve()),
            )
            self.assertEqual(
                resolve_resume_replay_buffer(model_path, None),
                replay_path,
            )
            self.assertEqual(
                validate_resume_config(model_path, DEFAULT_CONFIG),
                run_dir / "config.json",
            )
            self.assertEqual(
                find_run_file(model_path, "training.json"),
                run_dir / "training.json",
            )
            with self.assertRaisesRegex(ValueError, "exact saved task"):
                validate_resume_config(
                    model_path,
                    replace(DEFAULT_CONFIG, action_change_weight=5.0),
                )

    def test_full_resume_rejects_actor_transfer_at_the_same_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "exclusive"):
            algorithm_settings(
                DEFAULT_CONFIG,
                initial_actor_path=Path("actor.zip"),
                resume_model_path=Path("model.zip"),
            )

    def test_committed_model_replay_state_and_rolling_pointer(self) -> None:
        class FakeReplayBuffer:
            @staticmethod
            def size() -> int:
                return 7

        class FakeModel:
            num_timesteps = 12
            replay_buffer = FakeReplayBuffer()

            @staticmethod
            def save(path: Path) -> None:
                path.write_bytes(b"model")

            @staticmethod
            def save_replay_buffer(path: Path) -> None:
                path.write_bytes(b"replay")

        class FailingModel(FakeModel):
            @staticmethod
            def save_replay_buffer(path: Path) -> None:
                del path
                raise RuntimeError("interrupted replay save")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "final"
            save_committed_state(FakeModel(), final)
            self.assertEqual(resolve_resume_model(final), final / "model.zip")
            self.assertTrue((final / "replay_buffer.pkl").is_file())
            state = json.loads((final / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["num_timesteps"], 12)
            self.assertEqual(state["replay_buffer_size"], 7)
            self.assertEqual(
                state["model_sha256"],
                "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
            )
            self.assertEqual(
                state["replay_buffer_sha256"],
                "ac203c9843b5bd8c883e07039ff82820c94422010be6108bb82403ca25376a22",
            )

            rolling = root / "resume"
            snapshot = rolling / "snapshot_000000000012"
            save_committed_state(FakeModel(), snapshot)
            atomic_write_json(
                rolling / "latest.json",
                {"snapshot": snapshot.name, "num_timesteps": 12},
            )
            self.assertEqual(
                resolve_resume_model(rolling),
                snapshot / "model.zip",
            )

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                save_committed_state(FakeModel(), final)

            incomplete = root / "incomplete"
            with self.assertRaisesRegex(RuntimeError, "interrupted replay"):
                save_committed_state(FailingModel(), incomplete)
            self.assertFalse(incomplete.exists())
            self.assertEqual(list(root.glob(".incomplete-*")), [])

    def test_committed_state_retries_transient_directory_lock(self) -> None:
        class FakeReplayBuffer:
            @staticmethod
            def size() -> int:
                return 1

        class FakeModel:
            num_timesteps = 1
            replay_buffer = FakeReplayBuffer()

            @staticmethod
            def save(path: Path) -> None:
                path.write_bytes(b"model")

            @staticmethod
            def save_replay_buffer(path: Path) -> None:
                path.write_bytes(b"replay")

        real_rename = os.rename
        rename_calls = 0

        def transient_lock(source: Path, destination: Path) -> None:
            nonlocal rename_calls
            rename_calls += 1
            if rename_calls == 1:
                raise PermissionError(32, "file is being used by another process")
            real_rename(source, destination)

        with TemporaryDirectory() as directory:
            destination = Path(directory) / "final"
            with patch("train.os.rename", side_effect=transient_lock):
                with patch("train.time.sleep") as sleep:
                    save_committed_state(FakeModel(), destination)

            self.assertEqual(rename_calls, 2)
            sleep.assert_called_once_with(0.25)
            self.assertTrue((destination / "state.json").is_file())

    def test_model_equilibria_and_energy_diagnostic(self) -> None:
        plant = FurutaPendulum(DEFAULT_CONFIG.sample_time)
        for theta2 in (0.0, np.pi):
            derivative = plant.state_derivative(
                np.array([0.0, theta2, 0.0, 0.0]),
                current=0.0,
            )
            np.testing.assert_allclose(derivative, 0.0, atol=1e-10)
        self.assertAlmostEqual(plant.total_energy(np.zeros(4)), 0.0)
        self.assertAlmostEqual(
            plant.total_energy(np.array([0.0, np.pi, 0.0, 0.0])),
            plant.upright_energy,
        )
        self.assertAlmostEqual(nominal_pendulum_energy_error(np.zeros(4)), -1.0)
        self.assertAlmostEqual(
            nominal_pendulum_energy_error(np.array([0.0, np.pi, 0.0, 0.0])),
            0.0,
        )

    def test_current_filter_is_continuous_and_hidden(self) -> None:
        filtered_config = replace(DEFAULT_CONFIG, current_filter_cutoff_hz=100.0)
        plant = FurutaPendulum(DEFAULT_CONFIG.sample_time)
        plant.reset(np.zeros(4))
        _, applied = plant.step_filtered_current(
            0.5,
            applied_current=0.0,
            cutoff_hz=100.0,
        )
        expected = 0.5 * (
            1.0
            - np.exp(
                -2.0
                * np.pi
                * 100.0
                * DEFAULT_CONFIG.sample_time
            )
        )
        self.assertAlmostEqual(applied, expected, places=3)

        env = FurutaSwingUpEnv(filtered_config)
        try:
            observation, _ = env.reset(
                options={
                    "initial_state": np.zeros(4),
                    "parameter_scales": np.ones(7),
                }
            )
            self.assertEqual(observation.shape, (7,))
            self.assertAlmostEqual(observation[-1], 0.0)
        finally:
            env.close()

    def test_stage_one_applies_requested_current_directly(self) -> None:
        env = FurutaSwingUpEnv(DEFAULT_CONFIG)
        try:
            env.reset(
                options={
                    "initial_state": np.zeros(4),
                    "parameter_scales": np.ones(7),
                }
            )
            _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
            self.assertAlmostEqual(info["command_current"], 0.5)
            self.assertAlmostEqual(info["current"], 0.5)
        finally:
            env.close()

    def test_matlab_observation_order(self) -> None:
        plant = FurutaPendulum(DEFAULT_CONFIG.sample_time)
        plant.reset(np.array([0.3, np.pi + 0.4, 1.2, -2.3]))
        observation = observation_from_plant(
            plant,
            DEFAULT_CONFIG,
            previous_action=-0.25,
        )
        expected = np.array(
            [
                np.sin(0.3),
                np.cos(0.3),
                1.2 / DEFAULT_CONFIG.max_abs_omega1,
                np.sin(0.4),
                np.cos(0.4),
                -2.3 / DEFAULT_CONFIG.max_abs_omega2,
                -0.25,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observation, expected, rtol=1e-6, atol=1e-6)

    def test_matlab_reward_exact_terms(self) -> None:
        upright = np.array([0.0, np.pi, 0.0, 0.0])
        downward = np.zeros(4)
        self.assertAlmostEqual(matlab_reward(upright, 0.0, 0.0, DEFAULT_CONFIG), 1.0)
        self.assertAlmostEqual(
            matlab_reward(downward, 0.0, 0.0, DEFAULT_CONFIG),
            1.0 - 0.1 * np.pi**2,
        )
        held = matlab_reward(upright, 1.0, 1.0, DEFAULT_CONFIG)
        changed = matlab_reward(upright, 1.0, 0.0, DEFAULT_CONFIG)
        self.assertAlmostEqual(held, 0.9)
        self.assertAlmostEqual(
            changed,
            0.9 - 0.1 * DEFAULT_CONFIG.action_change_weight,
        )
        self.assertLess(changed, held)
        moving = np.array([0.0, np.pi, 5.0, -4.0])
        self.assertAlmostEqual(
            matlab_reward(moving, 0.0, 0.0, DEFAULT_CONFIG),
            1.0
            - DEFAULT_CONFIG.reward_scale
            * (
                DEFAULT_CONFIG.arm_velocity_weight * 5.0**2
                + DEFAULT_CONFIG.velocity_weight * 4.0**2
            ),
        )
        no_constraint_reward = replace(
            DEFAULT_CONFIG,
            constraint_reward_weight=0.0,
        )
        self.assertAlmostEqual(
            matlab_reward(upright, 0.0, 0.0, no_constraint_reward),
            0.0,
        )

    def test_arm_reward_weights_are_independent(self) -> None:
        shaped = replace(
            DEFAULT_CONFIG,
            arm_angle_weight=2.0,
            arm_velocity_weight=0.03,
        )
        upright = np.array([0.4, np.pi, 2.0, 3.0])
        expected_upright_cost = (
            2.0 * 0.4**2
            + 0.03 * 2.0**2
            + shaped.velocity_weight * 3.0**2
        )
        self.assertAlmostEqual(
            matlab_reward(upright, 0.0, 0.0, shaped),
            1.0 - shaped.reward_scale * expected_upright_cost,
        )

        downward = np.array([0.4, 0.0, 2.0, 3.0])
        expected_downward_cost = (
            2.0 * 0.4**2
            + np.pi**2
            + 0.03 * 2.0**2
            + shaped.velocity_weight * 3.0**2
        )
        self.assertAlmostEqual(
            matlab_reward(downward, 0.0, 0.0, shaped),
            1.0 - shaped.reward_scale * expected_downward_cost,
        )

    def test_environment_api_and_filtered_current_action(self) -> None:
        filtered_config = replace(DEFAULT_CONFIG, current_filter_cutoff_hz=100.0)
        env = FurutaSwingUpEnv(filtered_config)
        try:
            check_env(env, warn=True)
            observation, info = env.reset(seed=1)
            self.assertEqual(observation.shape, (7,))
            self.assertTrue(env.observation_space.contains(observation))
            np.testing.assert_allclose(info["parameter_scales"], np.ones(7))
            self.assertAlmostEqual(env.current_for_action(np.array([-1.0])), -0.5)
            self.assertAlmostEqual(env.current_for_action(np.array([0.25])), 0.125)
            self.assertAlmostEqual(env.current_for_action(np.array([2.0])), 0.5)
            next_observation, _, terminated, truncated, info = env.step(
                np.array([1.0], dtype=np.float32)
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertAlmostEqual(
                info["command_current"], filtered_config.max_current
            )
            self.assertGreater(info["current"], 0.0)
            self.assertLess(info["current"], filtered_config.max_current)
            self.assertAlmostEqual(next_observation[-1], 1.0)
            self.assertEqual(env.plant_step_count, filtered_config.action_repeat)
        finally:
            env.close()

    def test_training_reset_and_optional_randomized_plant(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(100):
            state = sample_matlab_initial_state(rng, DEFAULT_CONFIG)
            self.assertLessEqual(abs(state[0]), np.pi / 2.0)
            self.assertLessEqual(abs(state[1]), np.pi)
            self.assertLessEqual(abs(state[2]), 2.0)
            self.assertLessEqual(abs(state[3]), 2.0)
        env = FurutaSwingUpEnv(DEFAULT_CONFIG)
        try:
            _, info = env.reset(seed=0)
            np.testing.assert_allclose(info["parameter_scales"], np.ones(7))
        finally:
            env.close()

        randomized_env = FurutaSwingUpEnv(
            replace(DEFAULT_CONFIG, training_parameter_randomization=True)
        )
        try:
            for seed in range(20):
                _, info = randomized_env.reset(seed=seed)
                self.assertTrue(
                    np.all(
                        info["parameter_scales"]
                        >= [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
                    )
                )
                self.assertTrue(
                    np.all(
                        info["parameter_scales"]
                        <= [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10]
                    )
                )
        finally:
            randomized_env.close()

    def test_other_task_versions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported task_version"):
            FurutaConfig(task_version="matlab_hardware_sac_pure_v1")

    def test_evaluator_requires_the_current_config_schema(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_path = run_dir / "model.zip"
            config_path = run_dir / "config.json"
            model_path.touch()
            config_path.write_text(
                json.dumps(asdict(DEFAULT_CONFIG)),
                encoding="utf-8",
            )
            loaded, resolved = load_run_config(model_path, None)
            self.assertEqual(loaded, DEFAULT_CONFIG)
            self.assertEqual(resolved, config_path)

            incompatible = asdict(DEFAULT_CONFIG)
            incompatible["task_version"] = "matlab_hardware_sac_pure_v1"
            config_path.write_text(json.dumps(incompatible), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "current task version"):
                load_run_config(model_path, None)

            milestone = run_dir / "milestones" / "state_000000000100"
            milestone.mkdir(parents=True)
            nested_model = milestone / "model.zip"
            nested_model.touch()
            config_path.write_text(
                json.dumps(asdict(DEFAULT_CONFIG)),
                encoding="utf-8",
            )
            loaded, resolved = load_run_config(nested_model, None)
            self.assertEqual(loaded, DEFAULT_CONFIG)
            self.assertEqual(resolved, config_path)

    def test_parameter_randomization_ranges(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(100):
            scales = sample_parameter_scales(rng, DEFAULT_CONFIG)
            self.assertTrue(
                np.all(
                    scales >= [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90]
                )
            )
            self.assertTrue(
                np.all(
                    scales <= [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.10]
                )
            )

    def test_curriculum_scales_all_hidden_uncertainty(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            current_filter_cutoff_hz=100.0,
            current_filter_cutoff_randomization_hz=40.0,
            action_dead_time_max_samples=2,
            randomized_reset_omega1_half_range=4.0 * np.pi,
            randomized_reset_omega2_half_range=4.0 * np.pi,
        )
        rng = np.random.default_rng(33)
        np.testing.assert_allclose(
            sample_parameter_scales(rng, config, scale=0.0),
            np.ones(7),
        )
        self.assertEqual(sample_current_filter_cutoff_hz(rng, config, 0.0), 100.0)
        self.assertEqual(sample_action_dead_time_samples(rng, config, 0.0), 0)
        self.assertAlmostEqual(curriculum_scale(1_500_000, 1_500_000, 1_000_000), 0.0)
        self.assertAlmostEqual(curriculum_scale(2_000_000, 1_500_000, 1_000_000), 0.5)
        self.assertAlmostEqual(curriculum_scale(2_500_000, 1_500_000, 1_000_000), 1.0)

        full_states = np.array(
            [sample_matlab_initial_state(rng, config, 1.0) for _ in range(500)]
        )
        self.assertGreater(np.max(np.abs(full_states[:, 2:])), 2.0)
        self.assertLessEqual(np.max(np.abs(full_states[:, 2:])), 4.0 * np.pi)
        cutoffs = [sample_current_filter_cutoff_hz(rng, config) for _ in range(200)]
        self.assertGreaterEqual(min(cutoffs), 60.0)
        self.assertLessEqual(max(cutoffs), 140.0)
        delays = [sample_action_dead_time_samples(rng, config) for _ in range(200)]
        self.assertEqual(set(delays), {0, 1, 2})

    def test_asymmetric_current_filter_curriculum_range(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            current_filter_cutoff_hz=100.0,
            current_filter_cutoff_min_hz=50.0,
            current_filter_cutoff_max_hz=100.0,
        )
        rng = np.random.default_rng(44)
        self.assertEqual(sample_current_filter_cutoff_hz(rng, config, 0.0), 100.0)
        half_scale = [
            sample_current_filter_cutoff_hz(rng, config, 0.5)
            for _ in range(200)
        ]
        self.assertGreaterEqual(min(half_scale), 75.0)
        self.assertLessEqual(max(half_scale), 100.0)
        full_scale = [sample_current_filter_cutoff_hz(rng, config) for _ in range(200)]
        self.assertGreaterEqual(min(full_scale), 50.0)
        self.assertLessEqual(max(full_scale), 100.0)
        with self.assertRaisesRegex(ValueError, "either a symmetric or explicit"):
            replace(config, current_filter_cutoff_randomization_hz=10.0)
        with self.assertRaisesRegex(ValueError, "must lie inside"):
            replace(config, current_filter_cutoff_min_hz=110.0)

    def test_action_dead_time_delays_only_the_rl_plant_command(self) -> None:
        config = replace(DEFAULT_CONFIG, action_dead_time_max_samples=2)
        env = FurutaSwingUpEnv(config)
        try:
            env.reset(
                seed=0,
                options={
                    "initial_state": np.zeros(4),
                    "action_dead_time_samples": 2,
                },
            )
            _, _, _, _, first = env.step(np.array([1.0], dtype=np.float32))
            _, _, _, _, second = env.step(np.array([1.0], dtype=np.float32))
            _, _, _, _, third = env.step(np.array([1.0], dtype=np.float32))
            self.assertAlmostEqual(first["command_current"], config.max_current)
            self.assertAlmostEqual(first["plant_command_current"], 0.0)
            self.assertAlmostEqual(second["plant_command_current"], 0.0)
            self.assertAlmostEqual(third["plant_command_current"], config.max_current)
        finally:
            env.close()

    def test_curriculum_config_allows_only_uncertainty_changes(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model_path = run_dir / "model.zip"
            model_path.touch()
            source_config = replace(
                DEFAULT_CONFIG,
                current_filter_cutoff_hz=100.0,
            )
            (run_dir / "config.json").write_text(
                json.dumps(asdict(source_config)),
                encoding="utf-8",
            )
            target = replace(
                source_config,
                training_parameter_randomization=True,
                current_filter_cutoff_min_hz=50.0,
                current_filter_cutoff_max_hz=100.0,
                action_dead_time_max_samples=2,
                randomized_reset_omega1_half_range=4.0 * np.pi,
                randomized_reset_omega2_half_range=4.0 * np.pi,
            )
            self.assertEqual(
                validate_curriculum_config(model_path, target),
                run_dir / "config.json",
            )
            with self.assertRaisesRegex(ValueError, "Forbidden differences"):
                validate_curriculum_config(
                    model_path,
                    replace(target, action_change_weight=4.0),
                )

    def test_mass_inertia_and_damping_scales_are_independent(self) -> None:
        plant = FurutaPendulum(DEFAULT_CONFIG.sample_time)
        nominal_m1 = plant.m1
        nominal_m2 = plant.m2
        nominal_J1zz = plant.J1zz
        nominal_J2xx = plant.J2xx
        nominal_J2yy = plant.J2yy
        nominal_b1 = plant.b1
        nominal_b2 = plant.b2

        plant.set_parameter_scales(1.0, 1.05, 1.10, 0.95, 0.90, 0.80, 1.20)
        self.assertAlmostEqual(plant.m1, 1.05 * nominal_m1)
        self.assertAlmostEqual(plant.m2, 0.95 * nominal_m2)
        self.assertAlmostEqual(plant.J1zz, 1.10 * nominal_J1zz)
        self.assertAlmostEqual(plant.J2xx, 0.90 * nominal_J2xx)
        self.assertAlmostEqual(plant.J2yy, 0.90 * nominal_J2yy)
        self.assertAlmostEqual(plant.b1, 0.80 * nominal_b1)
        self.assertAlmostEqual(plant.b2, 1.20 * nominal_b2)

    def test_downward_rest_evaluation_is_distinct_from_training_reset(self) -> None:
        plant = make_episode(
            seed=10,
            config=DEFAULT_CONFIG,
            randomized=False,
            reset_mode="downward_rest",
        )
        self.assertLessEqual(abs(plant.x[0]), np.pi / 4.0)
        self.assertLessEqual(abs(plant.x[1]), np.pi / 4.0)
        np.testing.assert_allclose(plant.x[2:], 0.0)

    def test_matlab_constraints_and_numerical_guard(self) -> None:
        safe = np.array([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(within_matlab_constraints(safe, DEFAULT_CONFIG))
        self.assertFalse(unsafe(safe, DEFAULT_CONFIG))
        outside_arm = safe.copy()
        outside_arm[0] = DEFAULT_CONFIG.arm_angle_limit + 1e-6
        self.assertFalse(within_matlab_constraints(outside_arm, DEFAULT_CONFIG))
        self.assertTrue(unsafe(outside_arm, DEFAULT_CONFIG))
        excessive_arm_speed = safe.copy()
        excessive_arm_speed[2] = DEFAULT_CONFIG.max_abs_omega1 + 1e-6
        self.assertFalse(within_matlab_constraints(excessive_arm_speed, DEFAULT_CONFIG))
        self.assertTrue(unsafe(excessive_arm_speed, DEFAULT_CONFIG))
        excessive_pendulum_speed = safe.copy()
        excessive_pendulum_speed[3] = DEFAULT_CONFIG.max_abs_omega2 + 1e-6
        self.assertFalse(
            within_matlab_constraints(excessive_pendulum_speed, DEFAULT_CONFIG)
        )
        self.assertTrue(unsafe(excessive_pendulum_speed, DEFAULT_CONFIG))

    def test_constraint_failure_removes_reward_bonus_for_both_velocities(self) -> None:
        velocity_weights = {
            2: DEFAULT_CONFIG.arm_velocity_weight,
            3: DEFAULT_CONFIG.velocity_weight,
        }
        for velocity_index, velocity_weight in velocity_weights.items():
            state = np.array([0.0, np.pi, 0.0, 0.0])
            state[velocity_index] = 31.0
            expected_cost = DEFAULT_CONFIG.reward_scale * (
                velocity_weight * 31.0**2
            )
            self.assertAlmostEqual(
                matlab_reward(state, 0.0, 0.0, DEFAULT_CONFIG),
                -expected_cost,
            )

    def test_capture_and_supervisor_hysteresis(self) -> None:
        state = np.array([0.4, np.pi, 0.0, 0.0])
        self.assertTrue(mechanical_capture_ready(state, DEFAULT_CONFIG))
        candidate = lqr_handover_current(state, DEFAULT_CONFIG.max_current)
        self.assertTrue(capture_ready(state, candidate, DEFAULT_CONFIG))
        self.assertFalse(capture_ready(state, 0.5, DEFAULT_CONFIG))
        supervisor = HybridSupervisor(
            replace(DEFAULT_CONFIG, handover_hold_time=DEFAULT_CONFIG.sample_time)
        )
        self.assertIs(supervisor.update(state, candidate), Mode.LQR)
        state[1] = wrap_angle(np.pi + np.deg2rad(15.0))
        self.assertIs(supervisor.update(state, candidate), Mode.LQR)
        state[1] = wrap_angle(np.pi + np.deg2rad(21.0))
        self.assertIs(supervisor.update(state, candidate), Mode.SWING_UP)

    def test_supervisor_requires_sustained_capture(self) -> None:
        state = np.array([0.0, np.pi, 0.0, 0.0])
        current = lqr_handover_current(state, DEFAULT_CONFIG.max_current)
        supervisor = HybridSupervisor(DEFAULT_CONFIG)
        for _ in range(supervisor.required_capture_steps - 1):
            self.assertIs(supervisor.update(state, current), Mode.SWING_UP)
        self.assertIs(supervisor.update(state, current), Mode.LQR)

    def test_pure_rl_terminates_after_sustained_balance(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            episode_time=0.010,
            balance_hold_time=0.005,
        )
        env = FurutaSwingUpEnv(config)
        try:
            env.reset(
                options={
                    "initial_state": np.array([0.0, np.pi, 0.0, 0.0]),
                    "parameter_scales": np.ones(7),
                }
            )
            _, reward, terminated, truncated, info = env.step(np.array([0.0]))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["is_success"])
            self.assertAlmostEqual(reward, 1.0 + config.terminal_success_bonus)
        finally:
            env.close()

    def test_integration_failure_returns_finite_terminal_transition(self) -> None:
        env = FurutaSwingUpEnv(DEFAULT_CONFIG)
        try:
            initial_observation, _ = env.reset(seed=5)
            initial_state = env.plant.x.copy()

            def fail_with_invalid_state(command, applied_current, cutoff_hz):
                del command, applied_current, cutoff_hz
                env.plant.x[:] = np.inf
                raise RuntimeError("forced integration failure")

            env.plant.step_filtered_current = fail_with_invalid_state
            observation, reward, terminated, truncated, info = env.step(
                np.array([0.0])
            )
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["unsafe"])
            self.assertTrue(np.all(np.isfinite(observation)))
            self.assertTrue(np.isfinite(reward))
            np.testing.assert_allclose(env.plant.x, initial_state)
            np.testing.assert_allclose(observation[:6], initial_observation[:6])
        finally:
            env.close()

    def test_sac_only_evaluation_remains_available(self) -> None:
        class ZeroPolicy:
            @staticmethod
            def predict(observation, deterministic=True):
                del observation, deterministic
                return np.array([0.0], dtype=np.float32), None

        short_config = replace(DEFAULT_CONFIG, episode_time=0.010)
        result = run_episode(
            ZeroPolicy(),
            seed=10,
            randomized=False,
            reset_mode="training",
            config=short_config,
            controller="sac",
        )
        self.assertEqual(result["switches"], 0)
        self.assertEqual(result["lqr_fraction"], 0.0)
        self.assertTrue(all(row["mode"] == "swing_up" for row in result["trace"]))

    def test_hybrid_lqr_bypasses_rl_current_filter(self) -> None:
        class ZeroPolicy:
            @staticmethod
            def predict(observation, deterministic=True):
                del observation, deterministic
                return np.array([0.0], dtype=np.float32), None

        config = replace(
            DEFAULT_CONFIG,
            episode_time=0.005,
            handover_hold_time=0.001,
            balance_hold_time=0.001,
            current_filter_cutoff_hz=100.0,
        )
        plant = FurutaPendulum(config.sample_time)
        plant.reset(np.array([0.0, np.pi, 0.0, 0.0]))
        with (
            patch("evaluate.make_episode", return_value=plant),
            patch.object(
                plant,
                "step_filtered_current",
                wraps=plant.step_filtered_current,
            ) as filtered_step,
            patch.object(
                plant,
                "step_feedback",
                wraps=plant.step_feedback,
            ) as feedback_step,
        ):
            result = run_episode(
                ZeroPolicy(),
                seed=0,
                randomized=False,
                reset_mode="training",
                config=config,
                controller="hybrid",
            )
        self.assertGreater(feedback_step.call_count, 0)
        filtered_step.assert_not_called()
        self.assertGreater(result["lqr_fraction"], 0.0)

    def test_hybrid_uses_absolute_arm_reference_before_handover(self) -> None:
        class ZeroPolicy:
            @staticmethod
            def predict(observation, deterministic=True):
                del observation, deterministic
                return np.array([0.0], dtype=np.float32), None

        config = replace(
            DEFAULT_CONFIG,
            episode_time=0.005,
            handover_hold_time=0.010,
            balance_hold_time=0.001,
        )
        plant = FurutaPendulum(config.sample_time)
        plant.reset(np.array([0.5, np.pi, 0.0, 0.0]))
        with patch("evaluate.make_episode", return_value=plant):
            result = run_episode(
                ZeroPolicy(),
                seed=0,
                randomized=False,
                reset_mode="training",
                config=config,
                controller="hybrid",
            )
        self.assertEqual(result["switches"], 0)
        self.assertFalse(result["success"])
        self.assertTrue(
            all(row["theta1_reference_rad"] == 0.0 for row in result["trace"])
        )

    def test_evaluation_default_paths_and_controllers(self) -> None:
        self.assertEqual(run_episode.__defaults__[-1], "hybrid")
        self.assertEqual(evaluate_main.__defaults__[3], "both")
        self.assertEqual(evaluate_main.__defaults__[5], DEFAULT_EPISODES)
        self.assertEqual(evaluate_main.__defaults__[6], DEFAULT_BASE_SEED)
        self.assertIsNone(evaluate_main.__defaults__[7])
        self.assertEqual(
            MODEL_PATH.parent.name,
            "stage3c_half_rps_v0",
        )
        self.assertEqual(
            RESULTS_DIR.name,
            "evaluation_stage3c_half_rps_v0",
        )

    def test_half_rps_recipe_is_complete_and_resumable(self) -> None:
        recipe = load_recipe(DEFAULT_RECIPE)
        self.assertEqual(
            [stage["id"] for stage in recipe["stages"]],
            [
                "stage2_scratch_1m",
                "stage2_continue_1p5m",
                "stage3c_curriculum",
            ],
        )
        stage = recipe["stages"][-1]
        command = build_stage_command(recipe, stage, seed=2)
        self.assertIn("--curriculum-model", command)
        self.assertIn("--resume-milestone-timesteps", command)
        milestone_index = command.index("--resume-milestone-timesteps")
        self.assertEqual(
            command[milestone_index + 1 : milestone_index + 3],
            ["2400000", "2600000"],
        )
        omega_index = command.index(
            "--randomized-reset-omega1-half-range-rps"
        )
        self.assertEqual(command[omega_index + 1], "0.5")
        self.assertIn("seed2", str(stage_run_dir(stage, 2)))

    def test_selected_release_and_qualification_commands(self) -> None:
        manifest = verify_release()
        verify_frozen_evidence()
        self.assertEqual(manifest["selected_num_timesteps"], 2_600_000)
        self.assertEqual(
            manifest["model_sha256"],
            "ce2371e3f57107ae755fe5c6b8317129c5386e8cc96f9bc9f94d6f9282d63fbf",
        )
        command = build_suite_command("holdout")
        self.assertEqual(
            command[command.index("--model") + 1],
            str(QUALIFICATION_MODEL_PATH),
        )
        self.assertEqual(command[command.index("--base-seed") + 1], "70000")
        nominal_command = build_suite_command("nominal80")
        self.assertEqual(
            nominal_command[nominal_command.index("--base-seed") + 1],
            "90000",
        )

    def test_training_settings_validate_retained_resume_milestones(self) -> None:
        settings = algorithm_settings(
            DEFAULT_CONFIG,
            total_timesteps=100,
            resume_milestone_timesteps=[80, 40],
        )
        self.assertEqual(settings["resume_milestone_timesteps"], [40, 80])
        with self.assertRaisesRegex(ValueError, "positive even"):
            algorithm_settings(
                DEFAULT_CONFIG,
                total_timesteps=100,
                resume_milestone_timesteps=[41],
            )

    def test_evaluation_percentile_and_maximum_ignore_missing_values(self) -> None:
        results = [{"metric": 1.0}, {"metric": None}, {"metric": 3.0}]
        self.assertAlmostEqual(percentile_available(results, "metric", 50.0), 2.0)
        self.assertAlmostEqual(maximum_available(results, "metric"), 3.0)
        self.assertIsNone(percentile_available([{"metric": None}], "metric", 95.0))
        self.assertIsNone(maximum_available([{"metric": None}], "metric"))

    def test_deterministic_stress_profiles_cover_requested_endpoints(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            current_filter_cutoff_hz=100.0,
            current_filter_cutoff_min_hz=50.0,
            current_filter_cutoff_max_hz=100.0,
            randomized_reset_omega1_half_range=1.5 * 2.0 * np.pi,
            randomized_reset_omega2_half_range=1.5 * 2.0 * np.pi,
        )
        profiles = parameter_profiles()
        self.assertEqual(set(profiles), {"low_authority", "high_authority"})
        np.testing.assert_allclose(profiles["high_authority"][1:], 0.90)
        np.testing.assert_allclose(profiles["low_authority"][1:], 1.10)
        states = support_boundary_states(config)
        self.assertAlmostEqual(states["positive_arm_outward"][0], np.pi / 2.0)
        self.assertAlmostEqual(
            states["positive_velocity_edges"][2],
            1.5 * 2.0 * np.pi,
        )
        cases = corner_cases(config)
        self.assertEqual(len(cases), 48)
        self.assertEqual({case["cutoff_hz"] for case in cases}, {50.0, 100.0})
        self.assertEqual({case["delay_samples"] for case in cases}, {0, 1})
        self.assertEqual(len(long_duration_cases(config, 30.0)), 4)
        plant = prepare_plant(cases[0], config)
        self.assertAlmostEqual(
            plant.current_filter_cutoff_hz,
            cases[0]["cutoff_hz"],
        )
        self.assertEqual(plant.action_dead_time_samples, cases[0]["delay_samples"])

    def test_settling_metrics_use_post_capture_and_final_window(self) -> None:
        trace = [
            {
                "time_s": 0.0,
                "theta1_rad": 0.0,
                "theta1_relative_rad": 0.0,
                "omega1_rad_s": 0.0,
            },
            {
                "time_s": 1.0,
                "theta1_rad": 1.0,
                "theta1_relative_rad": 0.5,
                "omega1_rad_s": 2.0,
            },
            {
                "time_s": 2.0,
                "theta1_rad": 2.0,
                "theta1_relative_rad": 0.5,
                "omega1_rad_s": 4.0,
            },
        ]
        metrics = trajectory_shape_metrics(
            trace,
            capture_time=1.0,
            final_window_steps=1,
        )
        self.assertAlmostEqual(metrics["post_capture_arm_excursion"], 1.0)
        self.assertAlmostEqual(
            metrics["post_capture_omega1_rms"],
            np.sqrt(10.0),
        )
        self.assertAlmostEqual(metrics["final_arm_error_rms"], 0.5)
        self.assertAlmostEqual(metrics["final_omega1_rms"], np.sqrt(10.0))
        self.assertAlmostEqual(metrics["final_max_abs_omega1"], 4.0)

    def test_absolute_arm_angle_is_observed_and_penalized(self) -> None:
        plant_a = FurutaPendulum(DEFAULT_CONFIG.sample_time)
        plant_b = FurutaPendulum(DEFAULT_CONFIG.sample_time)
        state_a = np.array([0.0, np.pi, 0.0, 0.0])
        state_b = np.array([0.5, np.pi, 0.0, 0.0])
        plant_a.reset(state_a)
        plant_b.reset(state_b)
        self.assertFalse(
            np.allclose(
                observation_from_plant(plant_a, DEFAULT_CONFIG),
                observation_from_plant(plant_b, DEFAULT_CONFIG),
            )
        )
        self.assertLess(
            matlab_reward(state_b, 0.0, 0.0, DEFAULT_CONFIG),
            matlab_reward(state_a, 0.0, 0.0, DEFAULT_CONFIG),
        )

    def test_relative_lqr_is_shift_invariant(self) -> None:
        controller_a = LQRController(DEFAULT_CONFIG.max_current)
        controller_b = LQRController(DEFAULT_CONFIG.max_current)
        reference_a = np.array([0.2, np.pi, 0.0, 0.0])
        reference_b = reference_a.copy()
        reference_b[0] += 1.0
        controller_a.set_reference(reference_a)
        controller_b.set_reference(reference_b)
        state_a = np.array([0.3, np.pi + 0.05, 0.2, -0.3])
        state_b = state_a.copy()
        state_b[0] += 1.0
        self.assertAlmostEqual(
            controller_a.current(state_a),
            controller_b.current(state_b),
        )

    def test_lqr_stabilizes_randomized_capture_region(self) -> None:
        rng = np.random.default_rng(4)
        successes = 0
        for _ in range(30):
            plant = FurutaPendulum(DEFAULT_CONFIG.sample_time)
            plant.set_parameter_scales(
                rng.uniform(0.90, 1.10),
                rng.uniform(0.90, 1.10),
                rng.uniform(0.90, 1.10),
                rng.uniform(0.90, 1.10),
                rng.uniform(0.90, 1.10),
                rng.uniform(0.80, 1.20),
                rng.uniform(0.80, 1.20),
            )
            plant.reset(
                np.array(
                    [
                        rng.uniform(-0.5, 0.5),
                        wrap_angle(
                            np.pi
                            + rng.uniform(
                                -DEFAULT_CONFIG.capture_angle,
                                DEFAULT_CONFIG.capture_angle,
                            )
                        ),
                        rng.uniform(-2.0, 2.0),
                        rng.uniform(-2.0, 2.0),
                    ]
                )
            )
            controller = LQRController(DEFAULT_CONFIG.max_current)
            controller.set_reference(plant.x)
            for _ in range(int(round(5.0 / DEFAULT_CONFIG.sample_time))):
                plant.step_feedback(controller.current)
                plant.x[1] = wrap_angle(plant.x[1])
                if unsafe(plant.x, DEFAULT_CONFIG):
                    break
                if (
                    abs(upright_error(plant.x)) <= np.deg2rad(5.0)
                    and abs(controller.theta1_error(plant.x)) <= 0.25
                    and abs(plant.x[2]) <= 0.50
                    and abs(plant.x[3]) <= 1.00
                ):
                    successes += 1
                    break
        self.assertGreaterEqual(successes, 29)

    def test_short_sac_training_smoke_test(self) -> None:
        env = FurutaSwingUpEnv(DEFAULT_CONFIG)
        try:
            model = SAC(
                "MlpPolicy",
                env,
                learning_starts=10,
                buffer_size=200,
                batch_size=16,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs={"net_arch": {"pi": [8], "qf": [8]}},
                seed=0,
                verbose=0,
            )
            model.learn(total_timesteps=100)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
    curriculum_scale,
