# Stage-3c half-rps v0 selected controller

This directory freezes the selected seed-0 Stage-3c SAC controller at absolute
timestep 2,600,000. It is the only screened checkpoint that achieved 100/100
successes in each matched SAC evaluation mode with no unsafe episode. The
2.7M, 2.8M, and 2.9M checkpoints each reproduced one nominal unsafe seed; the
3.0M policy additionally lost one downward-rest episode.

`model.zip` is the evaluation and deployment candidate. Its SHA-256 is
`ce2371e3f57107ae755fe5c6b8317129c5386e8cc96f9bc9f94d6f9282d63fbf`.
`config.json` is the complete robust configuration used for training and the
full-envelope qualification. `training.json` preserves the original command,
source hashes, Git commit, Conda environment, and package versions.

The release actor does not contain a replay buffer. Exact continuation is
available after running the canonical reproduction recipe, which retains a
complete model/replay state at 2.6M. Do not pair this release model with the
3.0M replay buffer.

## Reproduce training

From a clean Git checkout with the `rl-env` Conda environment selected:

```bash
python src/reproduce.py --all --seed 0
python src/reproduce.py --all --seed 0 --execute
```

The first command prints the canonical three-stage chain. The second trains it.
The selected actor and its complete resumable state are then located at:

```text
src/runs/reproduce_stage3c_half_rps_selected_v0_seed0_stage3c/checkpoints/sac_2600000_steps.zip
src/runs/reproduce_stage3c_half_rps_selected_v0_seed0_stage3c/milestones/state_000002600000/
```

The historical recipe that produced this frozen actor remains unchanged at
`experiments/stage3c_half_rps_v0.json`. The canonical recipe changes no learning
setting; it uses fresh output directories and additionally retains the selected
2.6M replay state.

To continue the reproduced selected learner by another 500,000 decisions on
the unchanged task, use the VS Code profile
`Continue reproduced selected 2.6M: exact +500k` or run:

```bash
python src/train.py \
  --resume-model src/runs/reproduce_stage3c_half_rps_selected_v0_seed0_stage3c/milestones/state_000002600000 \
  --timesteps 500000 \
  --current-filter-cutoff-hz 100 \
  --current-filter-cutoff-range-hz 50 100 \
  --parameter-randomization \
  --motor-torque-randomization 0.10 \
  --arm-mass-randomization 0.10 \
  --arm-inertia-randomization 0.10 \
  --pendulum-mass-randomization 0.10 \
  --pendulum-inertia-randomization 0.10 \
  --arm-damping-randomization 0.10 \
  --pendulum-damping-randomization 0.10 \
  --action-dead-time-max-samples 1 \
  --randomized-reset-omega1-half-range-rps 0.5 \
  --randomized-reset-omega2-half-range-rps 1.5 \
  --run-dir src/runs/stage3c_half_rps_selected_v0_continued_500k
```

This restores actor, critics, target critics, optimizers, entropy state,
timestep count, and replay buffer. The resumed environment starts a new episode;
environment and callback RNG state are not serialized.

## Re-run qualification

Inspect the commands without executing them:

```bash
python src/qualify.py --suite all
```

After committing the intended code and release files, execute both frozen
protocols:

```bash
python src/qualify.py --suite all --execute
```

The runner verifies every release hash before evaluation and refuses canonical
execution from a dirty Git tree. It then checks the generated summary against
the frozen protocol. Use `--overwrite` only when deliberately replacing the
local canonical result directories.

The robust holdout uses seeds 70000--70299, both SAC and hybrid, full independent
mechanical `+/-10%` uncertainty, filter cutoff 50--100 Hz, action delay 0--1
policy sample, arbitrary pendulum angle, arm velocity `+/-0.5 rotations/s`, and
pendulum velocity `+/-1.5 rotations/s`.

The nominal protocol uses seeds 90000--90299, unit plant scales, a fixed 80 Hz
RL current filter, and zero action delay. Only initial positions and velocities
vary. The LQR current path remains direct continuous RK4-stage feedback.

## Evidence and status

- Selection set: SAC 300/300 and hybrid 300/300, zero unsafe.
- Unseen robust holdout: SAC 300/300 and hybrid 300/300, zero unsafe; all
  predeclared SAC trajectory-mean limits passed.
- Nominal 80 Hz/no-delay suite: SAC 300/300 and hybrid 300/300, zero unsafe.
- Deterministic corner/long-duration stress and physical hardware validation
  remain pending, so this release is not marked hardware-ready.

`selection_report.json` records every screened checkpoint. `acceptance.json`
is the predeclared unseen gate, and `holdout_report.json` records its result.
The `evidence/` directory contains the immutable raw summary tables used by the
reports. `manifest.json` hashes every release and evidence artifact.
