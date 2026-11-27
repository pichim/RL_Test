# Furuta preservation notes

The last full repository is Git commit `a399b03`, **Archive Furuta RL before
repository simplification**. Inspect it in a separate checkout:

```bash
git worktree add --detach ../furuta-before-cleanup a399b03
```

That checkout retains the original launchers, replay and continuation
workflows, randomized and hybrid evaluation, recovery teaching, reports, and
historical model releases. Its original reference run was
`src/runs/nominal_sac_scratch_local_seed0_20261123_182519/`. Generated runs
were ignored by Git.

## Reference

The frozen local-teacher actor was selected at 225,000 of 250,000 decisions
using training seed 0. The nominal simulated arm is periodic and uses one
1 ms RK4 plant step repeated five times per current decision. Current is
limited to ±1 A. The actor observes sine and cosine of the arm angle, arm
speed, sine and cosine of pendulum error from upright, pendulum speed, and
the previous normalized command. Training uses stock SAC with current and
current-change costs, energy shaping, exact reflection, and the local held
LQR teacher. The source defaults are in [src/config.py](src/config.py).

For this reference run, the trainer reset the same 50-start validation panel
every 25,000 decisions using seed 10000. It ranked candidates by balance success, balance after
teaching-box entry, and mean validation return, in that order. Training
terminates an episode at one uninterrupted second of balance. Nominal
evaluation instead checks the last second of the five-second rollout.
The frozen reference passed 100/100 on seeds 80000–80099, 50/50 on seeds
91000–91049, 81/81 upright, and all 30 fixed post-capture starts. All four
panels recorded zero unsafe episodes. See
[reference results](models/reference/results.json) for validation history,
benchmark metrics, and artifact hashes.

The reference machine used Linux, an RTX 3080, Python 3.12.16, and the
PyTorch 2.14.0 CUDA 13.2 wheel. Exact direct Python dependencies are in
[requirements.txt](requirements.txt). Re-run the complete seed-0 recipe when
changing learning behavior. Check the validation history, selected checkpoint,
and final panels before replacing the frozen reference.

## Local archive

`archives/pre-simplification-evidence.tar.gz` holds the compact local archive
of old generated results. It contains configurations, aggregate and
per-episode evaluation tables, saved source snapshots, and selected models.
Its embedded `ARCHIVE_INDEX.json` lists every preserved path and SHA-256.
The local archive is ignored by Git. Its checksum is
`639df13c30fe9c4b3bc94132bd7b842394bcbd2de3a4d6ed6754a8fbee870b8b`.

Browse it with `tar -tzf archives/pre-simplification-evidence.tar.gz`.
To restore a file, extract that listed path into the archival worktree with
`tar --keep-old-files -xzf <archive-path> -C ../furuta-before-cleanup <member>`.
The compact archive omits replay buffers, intermediate checkpoints,
TensorBoard logs, and full rollout traces and plots. Git can restore the
historical code and tracked model releases.

## Simplification verification

The simplified trainer completed a fresh 250,000-decision CUDA run on the
reference RTX 3080. All ten validation records matched the archived run
exactly. The selected 225k checkpoint matched all 35 saved policy tensors,
and both standalone actors agreed exactly on 128 input probes. Final scores
were 100/100 nominal, 50/50 fresh, 81/81 upright, and 30/30 post-capture,
with zero unsafe episodes. The corrected launcher passed a separate
end-to-end CUDA smoke test after separating validation and final evaluation
directories. The full evaluation was run from the completed checkpoint.

The generated verification runs were removed during cleanup. The tracked
reference and local archive preserve the earlier baseline.

## CPU trainer

The CPU baseline uses stock SAC with seed 0 and 250,000 decisions. Its
effective current and current-change weights are 0.30 and 1.50 at the
nominal ±1 A limit. These results describe that recipe. For a later run,
check the `config.json` and `training.json` saved in its run directory.

For comparison, the archived trainer at `42532bf` was run with a minimal CPU
adaptation. Its ten validation checkpoints and all 35 selected policy tensors
matched the newer CPU run exactly. Both selected the 250k checkpoint and
passed 100/100 nominal, 81/81 upright, and 30/30 post-capture starts, with
zero unsafe episodes. Training took 16:11 for the archived CPU version
and 16:02–16:37 for the newer CPU version on the reference desktop.

The CPU-only trainer completed the full recipe with PyTorch 2.14.0+cpu.
All ten validation records, the selected policy and optimizer states,
and final evaluation metrics matched the previous CPU run exactly.
The exported actors also agreed exactly on 128 input probes. Training
took 15:38 and evaluation took 2:00.

The timeout, periodic LQR angle, and post-capture current reporting fixes
were checked with another full seed-0 CPU run. All ten validation records,
saved policy and optimizer tensors, and evaluation outcomes matched the
earlier CPU run exactly. The corrected post-capture mean RMS current
change was 12.24 mA, previously reported as 13.23 mA.

At the same baseline settings, seeds 1 and 2 passed all nominal, upright,
and post-capture tests with checkpoints selected at 250k and 175k,
respectively. Seed 3's selected 200k checkpoint passed 46/100 nominal,
81/81 upright, and 30/30 post-capture starts. All three recorded zero unsafe
episodes in final evaluation.

The CUDA Graph experiment cut training time from 24:14 on eager CUDA to
13:44, but its seed-0 controller passed 86/100 nominal, 73/81 upright,
and 25/30 post-capture starts. The eager CUDA controller passed all three
panels. The CPU timing comparison and GPU experiment used the same task
and training settings.


## Control-rate verification

Training uses 200 Hz. A fresh seed-2 run reproduced the
archived 200 Hz validation history, selected policy and optimizer tensors,
and existing evaluation scores exactly. The evaluator also reports RMS
command slew in A/s for comparisons across rates.

The historical 500 Hz training experiment kept the 1 ms plant integration
step and used a 2 ms control period. Its time-matched seed-2 run selected the 437,500
decision checkpoint out of 625,000. It scored 11/50 on validation, 22/100
nominal, 49/81 upright, and 4/30 post-capture. Six nominal episodes were
unsafe.

A separate 500 Hz seed-2 run changed only the effective current-change
weight from 3.75 to 1.50. It scored 1/50 on validation, 1/100 nominal,
49/81 upright, and 0/30 post-capture, with zero unsafe evaluation episodes.
This diagnostic used a temporary output directory.

The saved seed-1 and seed-2 actors trained at 200 Hz each passed 100/100
nominal, 81/81 upright, and 30/30 post-capture starts when evaluated
at 500 Hz in the same simulator. The current workflow evaluates each
selected 200 Hz checkpoint at both rates and exports one actor.


## Matched actor-width comparison

All three runs trained at 200 Hz on CPU for 300,000 decisions with seed 0.
Only the actor width varied. They used 64×64 critics, a learning rate of
0.0002, 50,000 warm-up decisions, one SAC update per decision, energy weight
0.25, and effective current and current-change costs of 0.20 and 1.50.
The limits, reward, and benchmark starts matched across all three runs.

| Actor | Local run suffix | Selected decision | 200 Hz nominal / upright / post-capture | 500 Hz nominal / upright / post-capture |
| ----- | ---------------- | ----------------: | --------------------------------------- | --------------------------------------- |
| 16×16 | 155301_196149    |           225,000 | 100/100, 81/81, 30/30                   | 100/100, 81/81, 30/30                   |
| 12×12 | 165726_990606    |           175,000 | 28/100, 55/81, 2/30                     | 32/100, 57/81, 3/30                     |
| 8×8   | 174839_845564    |           250,000 | 97/100, 81/81, 30/30                    | 98/100, 81/81, 30/30                    |

The local directories use the prefix `local_20261126_`. Superseded
runs are under `runs_old/`, outside Git. The 16×16 checkpoint passed all panels without a simulated
safety violation. The 8×8 passed both recovery panels and hit a safety
guard on nominal seeds 80051 and 80069 at both rates. The selected
checkpoints scored 50/50, 15/50, and 49/50 on validation respectively.
At 300,000 decisions those scores were 1/50, 2/50, and 21/50. This is one
seed per width. The [current mathematics](README_MATH.md) describes the
16×16 default and the shared recipe.

## Training experiments on 2026-11-27

The 16×16 seed sweep used the same CPU recipe with one million decisions
per seed. Seeds 0, 1, and 2 first achieved 50/50 validation balances by
375,000 decisions. At one million their scores were 26/50, 1/50, and 0/50,
even though all still reached mechanical capture on all 50 starts. On the
old selector seed 0's 350k checkpoint beat the 225k checkpoint on summed
validation reward despite finishing balance more slowly.

The current selector validates the full five-second episode and ranks final
one-second holds, fewer unsafe episodes, then earlier sustained balance.
With the baseline training reward, it recovered the 225k seed-0 checkpoint
and its perfect three-panel evaluation at both rates.

The episode-ending comparison kept the +100 completion bonus once per
episode. Continuing after that first hold produced 50/50 at 475k in seed 0,
then 0/50 from 500k to 550k. Its selected 350k actor balanced 98/100 at both
rates and exceeded the arm-speed guard on the same two starts. The early-stop
control's 225k actor passed all three panels and recorded zero violations.

The immediate-balance experiment added +0.25 per safe decision ending
inside the strict balance box. Matched runs differed only in that reward.
Mean validation successes across the five checkpoints from 450k through
550k were 15.0 → 50.0, 33.6 → 6.8, and 50.0 → 0.6 for seeds 0, 1, and 2,
respectively. Selected +0.25 actors passed all upright and post-capture
tests. Seed 0 scored 97/100 at 200 Hz with three unsafe starts and 100/100
at 500 Hz. Seeds 1 and 2 scored 100/100 at both rates with zero violations.
The baseline reward and early-stop training remain active. The two discarded
experiment launchers and their exact source snapshot are in the local
`runs/robustness_20261127/` directory alongside the experiment patch and
its base commit.

| Actor for stress evaluation | Local run                                   | Selected checkpoint |
| --------------------------- | ------------------------------------------- | ------------------: |
| Seed 0 control              | `balance_control_20261127_150536_321738`    |                225k |
| Seed 2 control              | `balance_control_s2_20261127_164233_625456` |                425k |
| Seed 1 +0.25                | `balance_dense_s1_20261127_164233_627566`   |                425k |
| Seed 2 +0.25                | `balance_dense_s2_20261127_164233_627004`   |                300k |

### Held-out robustness evaluation

Each frozen actor was tested at 200 and 500 Hz with the original 1 ms
plant integration, ±1 A command limit, and ±30 rad/s speed guards. The
fresh nominal panel used seeds 90000–90063 and ran for five seconds. Each
stress panel reused seeds 90000–90031 and ran for eight seconds. All actors
and rates shared the same start states and perturbation draws.

| Panel             | Perturbation                                                                                                                                                                                                                                                |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fresh nominal     | Original plant and observations                                                                                                                                                                                                                             |
| Plant variation   | Independent uniform ±10% changes in motor torque constant, arm inertia, and pendulum mass with its three inertias scaled together. Independent uniform ±30% changes in both damping coefficients. Each draw stays fixed during an episode.                  |
| Measurement noise | Independent zero-mean Gaussian noise with standard deviation 0.25° on each measured angle and 0.10 rad/s on each measured velocity. Both rates sample the same noise sequence at their own decision times. Safety and balance use the true simulated state. |
| Late kick         | A ±1 rad/s jump in pendulum velocity at four seconds. The sign alternates across starts.                                                                                                                                                                    |

These are illustrative simulation perturbations. Hardware measurements were
unavailable for choosing the ranges. The stressors were tested separately.
The standard score requires balance throughout the last second. A second
score uses the last three seconds, with the same strict balance bounds.
Arm-speed margin is 30 rad/s minus the largest absolute simulated arm
speed. Kick recovery time is measured from the impulse to the start of
the final uninterrupted balance, for starts that held balance immediately
before the kick and recovered by the end.

The local `runs/robustness_20261127/` directory contains the reproducible
`stress_eval.py`, the source used by the workers, per-episode CSV, summary
JSON, and SHA-256 checkpoint identifiers.

All four actors passed all 64 fresh nominal starts at both rates. Every
actor recovered from all 32 kicks at both rates and then met the final
three-second hold. The seed-2 control recovered from the kick in at most
0.514 s. The seed-1 reward actor took at most 0.795 s. No safety guard
tripped in any of the 1,280 rollouts.

| Actor                | Plant variation, 200 / 500 Hz | Measurement noise, 200 / 500 Hz | Minimum arm-speed margin, 200 / 500 Hz |
| -------------------- | ----------------------------- | ------------------------------- | -------------------------------------- |
| Seed 0 control, 225k | 31/32 · 30/32                 | 7/32 · 32/32                    | 10.49 · 10.90 rad/s                    |
| Seed 2 control, 425k | 32/32 · 32/32                 | 7/32 · 31/32                    | 9.01 · 9.25 rad/s                      |
| Seed 1 +0.25, 425k   | 32/32 · 32/32                 | 24/32 · 32/32                   | 10.79 · 11.30 rad/s                    |
| Seed 2 +0.25, 300k   | 31/32 · 31/32                 | 13/32 · 32/32                   | 10.01 · 10.35 rad/s                    |

Seed 2 control remains the main baseline. It finished fresh nominal starts
in 2.28 s at 200 Hz and 2.27 s at 500 Hz on average, with RMS current slew
of 5.06 and 6.50 A/s. Seed 1 +0.25 is the strongest saved noise challenger.
Its seven-feature actor passed 160/160 tests at 500 Hz and 152/160 at
200 Hz. Its noise-panel current slew still rose to 26.24 A/s at 500 Hz.
The controls and reward actors generally held balance better under noise
at 500 Hz, with a higher current slew at that rate. These are simulation
results for individual checkpoints and do not change the training default.

A follow-up separated the two measurement-noise channels on seeds
90000–90015 from the same panel. The combined results below reuse the
corresponding original rollouts. The two isolated conditions added 128
new rollouts with zero safety violations.

| Actor and rate         | Combined noise | Angle noise only | Velocity noise only |
| ---------------------- | -------------: | ---------------: | ------------------: |
| Seed 2 control, 200 Hz |           3/16 |            16/16 |                4/16 |
| Seed 1 +0.25, 200 Hz   |          11/16 |            16/16 |               14/16 |
| Seed 2 control, 500 Hz |          16/16 |            16/16 |               16/16 |
| Seed 1 +0.25, 500 Hz   |          16/16 |            16/16 |               16/16 |

Velocity measurement noise accounted for most of the 200 Hz balance
losses on this subset. With velocity noise alone, mean current slew at
500 Hz was about 24 A/s for both candidates. Measuring the hardware
velocity estimate and checking its filtering and delay are the next
steps before changing the training reward. The local evidence folder
contains `noise_diagnosis.py`, `noise_isolation.json`, and the per-episode
CSV for the follow-up.

### Encoder model screening

The new training pipeline simulates two independent white encoder-angle
errors. A 1 ms encoder period and 100 Hz velocity-filter pole feed a
wrapped finite-difference estimate to the actor. Both SAC critics receive
the corresponding clean simulator features. The encoder sampling period
is an assumption pending the hardware configuration.

The earlier seed-2 control at 425k, trained with ideal features, was
replayed with the estimator on 16 fresh starts (seeds 91100–91115). The
same encoder draws were used at both rates. The table includes stationary
velocity estimates from 20,000 simulated encoder ticks per level.

| Angle noise RMS per joint  | Stationary velocity RMS | Balance at 200 / 500 Hz | Nominal current slew at 200 / 500 Hz |
| -------------------------- | ----------------------: | ----------------------- | ------------------------------------ |
| Ideal features             |                       — | 16/16 · 16/16           | 5.63 · 7.23 A/s                      |
| Encoder filter, zero noise |                       0 | 16/16 · 16/16           | 5.75 · 7.36 A/s                      |
| 75 µrad                    |             0.040 rad/s | 16/16 · 16/16           | 7.04 · 12.21 A/s                     |
| 150 µrad                   |             0.080 rad/s | 12/16 · 16/16           | 9.87 · 20.67 A/s                     |
| 300 µrad                   |             0.160 rad/s | 0/16 · 12/16            | 17.27 · 39.37 A/s                    |

No simulated safety guard tripped in these 160 rollouts. The initial
training setting is 75 µrad so sensor corruption starts small at both
rates. These estimates characterize the sensor model and the previous actor.
The local encoder simulation artifacts are in
`runs/encoder_simulation_20261127/`.



### Encoder-trained seed sweep

Seeds 0–2 trained from scratch for 550,000 decisions each using the
16×16 actor and 75 µrad encoder noise. The three jobs took about
69 minutes in parallel, including both rate evaluations.

| Seed | Selected checkpoint | Nominal at 200 / 500 Hz | Upright / post-capture at both rates | Unsafe at 200 / 500 Hz | Validation at 550k |
| --- | ---: | --- | --- | --- | ---: |
| 0 | 550k | 100/100 · 100/100 | 81/81 · 30/30 | 0 · 0 | 50/50 |
| 1 | 175k | 100/100 · 95/100 | 81/81 · 30/30 | 0 · 0 | 1/50 |
| 2 | 350k | 98/100 · 98/100 | 81/81 · 30/30 | 2 · 2 | 50/50 |

Seeds 0 and 2 recovered at the last validation checkpoint after sharp
drops. On the same 16 fresh starts used in the encoder screening,
seed 0 and the ideal-trained reference both passed 16/16 at both rates
at 75 µrad. At 500 Hz both averaged about 12.2 A/s RMS slew on this
panel. At 150 µrad, the selected seed-0 actor scored 4/16 at 200 Hz,
the older reference scored 12/16, and the new seed-1 and seed-2 actors
each scored 15/16. The matched, per-episode replay results are in
`runs/encoder_simulation_20261127/matched_trained_actors.json`. These
results compare the whole encoder-training recipe with the earlier
reference. The sweep used the same privileged-critic design for all
three new seeds.

### Retained training runs

| Role | Run under `runs/` | Selected checkpoint |
| --- | --- | ---: |
| Encoder-trained candidate at 75 µrad | `local_20261127_185228_020464` | 550k |
| Ideal-trained control for matched comparisons | `balance_control_s2_20261127_164233_625456` | 425k |

The next controller comparison should use the hardware's measured
encoder and velocity-estimator characteristics. Completed runs outside
this shortlist were moved to `runs_old/`. Result files in archived
runs preserve their original model paths. The local robustness scripts
resolve checkpoints in both directories. The two evidence folders in
`runs/` contain test results and reproducibility files.
