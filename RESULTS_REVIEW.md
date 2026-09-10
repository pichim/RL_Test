# Experiment review and staged plan

This document records the earlier staged experiments and release evidence.
For the current nominal actor64 reward experiments and the 2026-09-10 decision,
start with [HANDOFF.md](HANDOFF.md). The stage releases below are distinct from
the current1/delta3 nominal development reference; do not transfer qualification
claims between them. Historical plans below are not instructions to launch work.

The cleaned executable task and all three planned stages use schema `v0`
(`matlab_hardware_sac_pure_v0`). Stage numbers describe the experimental
sequence, while `v0` identifies the shared compatibility boundary. Future
structural schema-breaking task or plant changes advance the version; numeric
reward weights, seeds, training duration, and movement between the three
documented stages do not because they are recorded in each run configuration.

## Fixed successful comparisons

The `64 -> 64` actor baseline achieved 300/300 SAC-only and 300/300 hybrid
successes across the three evaluation scenarios, with no unsafe episode.

The successful `8 -> 8` actor run retained `64 -> 64` critics and the baseline
reward/dynamics settings. Its selected checkpoint achieved 99/100 SAC-only
nominal successes and 100/100 in both randomized scenarios. It reduced arm
excursion and peak arm speed, but increased pendulum-speed and current metrics.

## Failed combined robust run

The next run simultaneously changed the velocity weights, action-change weight,
hidden actuator response, and training-plant distribution. It achieved:

- zero successful episodes during training;
- 0/100 SAC-only successes in all three standalone scenarios;
- hybrid success of 89/100 nominal, 10/100 randomized broad-reset, and 6/100
  randomized downward-rest;
- zero unsafe standalone episodes;
- lower arm excursion, but substantially worse pendulum velocity, current RMS,
  and requested-current total variation.

The result is retained as a negative experiment. It should not be fully resumed.

## Failed equal-velocity stage-1 run

The next nominal run isolated the reward from the filter and randomizer but
changed both velocity weights: arm velocity `0.020 -> 0.015` and pendulum
velocity `0.010 -> 0.015`. It completed without a successful training episode
and achieved 0/100 SAC-only successes in every standalone mode. Nominal SAC
still reached capture in 99/100 episodes, showing that the failure was arm and
pendulum settling rather than swing-up acquisition. The result is retained at
`sac_200hz_matlab_pure_actor8_critic64_stage1_nominal_v0` as a negative
experiment.

## Current three-stage sequence

All stages retain arm-angle weight `1.5`, arm-velocity weight `0.015`, pendulum
velocity weight `0.010`, action-change weight `3.0`, an `8 -> 8` actor,
`64 -> 64` critics, pure-RL one-second balance termination, and the `+100`
terminal bonus.

1. Stage 1 trains from scratch on the nominal direct-current plant and changes
   only arm-velocity weight `0.020 -> 0.015` from the successful actor-8 run.
2. Stage 2 trains from scratch on the nominal plant with the hidden continuous
   100 Hz RL current response.
3. Stage 3 continues the complete selected Stage-2 1.5M learner and replay
   buffer. During its first additional million decisions, one curriculum scale
   linearly expands every uncertainty from the Stage-2 nominal setting to:
   independent `+/-10%` motor torque, both masses, both rotary inertias, and
   both damping coefficients; `50--100 Hz` RL current-filter cutoff; `0--1`
   hidden 200 Hz RL action-delay samples; and `+/-1.5 rotations/s` initial arm
   and pendulum velocities. It then trains one million decisions at full range,
   with learning rate reduced from `3e-4` to `1e-4` for the final 500,000.

The refined curriculum seed-0 pilot starts from the committed full-state source under
`sac_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0_selected_to1500k/resume`.
If the pilot passes the complete SAC-only and hybrid evaluation, seeds 1 and 2
will branch from the same source with otherwise identical settings. Those are
adaptation-seed confirmations; genuinely independent end-to-end seeds require
retraining the preceding stages as well.

## Completed Stage-3a curriculum and refinement

The first curriculum pilot used the same schedule and mechanical ranges, but
used `60--140 Hz`, `0--2` delay samples, and `+/-2 rotations/s` reset
velocities. Its 3.5M-total-step final policy achieved SAC-only success of
100/100 nominal, 76/100 randomized broad reset, and 81/100 randomized downward
rest. It had nine unsafe broad-reset episodes. Hybrid success was 100/100,
77/100, and 81/100.

The randomized episode table showed that every non-unsafe 0 or 5 ms broad-reset
episode succeeded. At 10 ms only 11/26 safe broad-reset episodes and 16/35
downward-rest episodes succeeded. All nine unsafe starts had an initial
velocity magnitude above `8 rad/s`; several were near the arm limit and moving
outward, leaving little recoverable travel. Nominal requested-current changes,
reversals, arm excursion, and arm-velocity RMS also regressed relative to the
selected Stage-2 source.

The refined Stage-3b run therefore starts again from the selected Stage-2 1.5M
state rather than continuing Stage 3a. It changes only the cutoff interval to
`50--100 Hz`, delay to `0--1` sample, and both reset-velocity bounds to
`+/-1.5 rotations/s`; all other curriculum and learner settings remain fixed.

## Selected Stage-3b controller and fast finalization

The Stage-3b final 3.5M policy regressed, so deterministic checkpoint screening
was followed by complete standalone evaluation. The selected absolute 2.4M
checkpoint had received 900,000 Stage-3b decisions and reached curriculum scale
0.9. It achieved 100/100 SAC-only and 100/100 hybrid successes in each of the
nominal, randomized broad-reset, and randomized downward-rest modes, with zero
unsafe episodes. Evaluation used the full configured uncertainty envelope even
though training had reached 90% of it.

Matched comparison on that same full Stage-3b envelope produced:

| controller | SAC nominal / broad / downward | unsafe |
|---|---:|---:|
| selected Stage-2 1.5M source | 100 / 62 / 54 | 0 |
| Stage-3a success-first best at 2.3M | 99 / 100 / 100 | 1 |
| selected Stage-3b checkpoint at 2.4M | 100 / 100 / 100 | 0 |

Stage 3b also reduced requested-current total variation, post-capture arm
motion, and pendulum-velocity RMS relative to Stage 3a. The chosen policy is
frozen under `models/stage3b_v0/` with its hashes and provenance. It has no
paired replay buffer and is not an exactly resumable learner.

The fast finalization path performs no more SAC training. It uses the untouched
holdout base seed 50000, first with SAC and then hybrid, followed by exact
parameter/filter/delay corners and 30-second drift cases. Adaptation seeds 1 and
2 are deferred research confirmation rather than a prerequisite for freezing
this particular controller.

The first untouched SAC holdout achieved 100/100 nominal, 99/100 randomized
broad-reset, and 100/100 randomized downward-rest success. Broad seed 50116 was
unsafe. It started at arm angle `-1.203 rad` and outward arm speed
`-8.181 rad/s`, with one-sample delay and an `83.4 Hz` cutoff, and crossed the
negative arm limit after `0.162 s`. All sampled mechanical scales were inside
approximately `+/-9%`, so this is not only a 90%-versus-100% curriculum issue.
The same state also fails under hybrid because the violation precedes a valid
handover. The predeclared hard gate therefore failed; the full hybrid holdout
and stress suite were not run. Seed 50116 is retained as a mandatory regression
case.

Run stage 1 from VS Code with `Train stage 1: arm velocity 0.015 (default)`
or:

```bash
python src/train.py
tensorboard --logdir src/runs/sac_200hz_matlab_pure_actor8_critic64_stage1_armv015_v0/tensorboard
```

After it finishes, use the matching stage-1 evaluation profile or:

```bash
python src/evaluate.py
```

Do not advance a stage unless SAC-only and hybrid evaluation retain reliability
and safety while improving the targeted physical metrics. Requested-current
smoothness remains the policy metric; filtered applied current must not hide a
rough command.

## Current project boundary

The executable project supports only the current pure-RL task schema. Historical
task versions, reset aliases, and configuration backfills have been removed.
Saved summaries remain valid comparisons, but old models are not expected to
load in the current evaluator. Scratch, actor-transfer, and full-state resume
remain available for current-schema experiments.

Final and rolling full-resume states publish the model and replay buffer as one
committed directory. Rolling continuation uses the atomically updated
`resume/latest.json` pointer, and timestep requests must be even to match SAC's
two-decision training frequency exactly.

Hardware deployment still requires measured uncertainty ranges, current and
thermal limits, travel protection, encoder/velocity validation, timing checks,
watchdogs, and an emergency stop outside the learned policy.

## Restricted-envelope qualification and reproducibility

The next predeclared qualification limits randomized initial arm speed to
`+/-0.5 rotations/s` while leaving pendulum speed at `+/-1.5 rotations/s` and
leaving all mechanical, filter, delay, reward, and controller settings
unchanged. Base seed `60000` is reserved in
`models/stage3b_v0/acceptance_half_rps_v1.json`. This is a new operational
envelope, not a reinterpretation of the failed broad-domain holdout.

The restricted SAC holdout achieved `99/100` nominal, `100/100` randomized
half-rps, and `100/100` randomized downward-rest success. Nominal seed `60066`
started near upright at arm angle `+0.985 rad` with outward arm speed
`+1.906 rad/s` (`0.303 rotations/s`) and crossed the positive travel limit
after `0.276 s`. Hybrid reproduced the failure before handover. This shows that
a scalar speed restriction is insufficient; deployment engagement must use a
position/direction/velocity viability rule or a narrower downward-start domain.

The selected controller's true ancestry is Stage 2 at 100 Hz trained from
scratch to `1.0M`, exact Stage-2 continuation to `1.5M`, and Stage-3b curriculum
adaptation. Stage 1 is a comparator rather than an ancestor. The tracked
historical and half-rps recipes under `experiments/` now make that chain
executable. New runs retain hashed model/replay pairs at selectable milestones,
so later training can use exact SAC continuation instead of actor-only transfer.
