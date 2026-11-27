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

Every 25,000 decisions the trainer resets the same 50-start validation panel
using seed 10000. It ranks candidates by balance success, balance after
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

The local directories use the prefix `runs/local_20261126_` and remain
outside Git. The 16×16 checkpoint passed all panels without a simulated
safety violation. The 8×8 passed both recovery panels and hit a safety
guard on nominal seeds 80051 and 80069 at both rates. The selected
checkpoints scored 50/50, 15/50, and 49/50 on validation respectively.
At 300,000 decisions those scores were 1/50, 2/50, and 21/50. This is one
seed per width. The [current mathematics](README_MATH.md) describes the
16×16 default and the shared recipe.
