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

The old generated directory was removed after these checks and the compact
archive verification. The retained verification run is
`runs/local_20261123_203957_207568/`. Its `verification.json` records these
checks and its `evaluation/` directory contains the new plots.
