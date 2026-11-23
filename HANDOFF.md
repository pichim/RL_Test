# Development Handoff

Updated 2026-11-23 after the two completed periodic 250k scratch runs.

## Current decision

Use the [frozen local 225k actor](models/nominal_scratch_actor16_v1/README.md)
as the periodic nominal reference. It passed every evaluation panel and
settled faster with smaller current changes than the recovery-trained actor.
Keep the recovery-trained 200k actor as a matched comparison. In this seed-0
pair, recovery teaching reached 50/50 validation at 200k and retained it
through 250k. The local run reached 50/50 at 225k and fell to 12/50 at 250k.

Keep periodic arm orientation, ±π angle starts, ±6 rad/s initial joint speeds,
±1 A, 5 ms, the symmetric seven-input 16×16 actor, 64×64 critics, current
weights of 0.30 / 1.50 in amperes, energy shaping, and validation every 25k.
The scratch profiles use discount 0.99, the legacy reward with periodic arm
error, and 500 local LQR warm-up updates. The two variants differ in the
additional recovery demonstrations applied during SAC updates.

The prior periodic time-cost pilot changed several settings at once. Its
best actor reached capture in 100/100 episodes and sustained balance in
70/100. Held-current LQR recovered all 30 of those failure states from the
next control tick. The previous 200k actor recovered 28/30 and both selected
scratch actors recovered 30/30. The saved 30-start panel is evaluation-only.
The recovery teachers for the scratch run used separately sampled states.

The [release](models/nominal_scratch_actor16_v1/README.md) records the model
files, timing, benchmark data, and source hashes. The historical
current1 / delta3 actor remains the reference for its bounded task. The
historical current1.5 / delta4 run achieved 3/20 final balance and had an
unstable slow pole pair.

The next planned comparison is a matched simulation stress test of the
frozen actors with current lag, command delay, and mechanical uncertainty.
The scratch releases were evaluated on the direct-current nominal plant.

## Linux CUDA Nominal Run (2026-11-22)

Run directory: `src/runs/nominal_sac_actor64_linux_cuda_seed0_20261122_095132`.
The Linux desktop completed the current1/delta3 nominal protocol with seed 0,
2,000,000 decisions, 64×64 actor and critic networks, and gamma 0.99. Training
used the RTX 3080 with PyTorch 2.14.0+cu132 in Miniforge `rl-env`, under the
desktop's balanced power profile. The final evaluation ran on CPU.

| Phase | Wall time |
|---|---:|
| CUDA training, including periodic validation and checkpoint saves | 2 h 04 min 33.4 s |
| Final CPU evaluation and artifact export | 1 min 24.0 s |
| Sum of the two timed phases | 2 h 05 min 57.4 s |

Training averaged **267.6 decisions/s** over the full timed training phase.
The matched monitor logs through the last completed training episode give
**268.3 decisions/s on the Linux desktop with CUDA** and **181.8 decisions/s
for the archived Windows CPU run**. The runs used different hardware and
software environments. The full Linux CPU comparison is still pending.

The **300,000-decision checkpoint** won selection with 50/50 on the fixed
validation episodes and the highest mean validation reward among tied
checkpoints. The evaluation artifact matches `training/best/best_model.zip`.
The 100 five-second SAC-only evaluation episodes (seeds 80000–80099) yielded
**98/100 final balance, 100/100 captures, and 0 unsafe episodes**. Seeds
80034 and 80044 stayed within all balance limits for the final 0.826 s and
0.711 s, respectively, short of the required 1 s. Both were within limits at
the end. The original current1/delta3 reference scored **100/100** on the
same evaluation seeds and remains the development reference.

The run's `FINDINGS.md` and `status.json` record the timing, and
`evaluation/summary.csv` plus `evaluation/sac/training_nominal/` hold the
aggregate results and per-episode traces. The validation history is at
`training/evaluation/evaluations.npz`. The best model is for evaluation.
The final model and replay buffer under `training/final/` are available
for continuation of this learner. These artifacts are local and Git-ignored.

When work resumes, use the [Linux CPU comparison in README.md](README.md#linux-cuda-setup-and-nominal-benchmark)
with the same evaluation and validation settings and balanced power profile.
Compare training wall time and evaluation results before changing the learner
or simulator.

### 16×16 current-penalty comparison (2026-11-23)

Two fresh 100k-decision CUDA runs used the same seed 0, seven-input actor,
64×64 critics, ±1 A limit, 5 ms decision period, teaching, and energy shaping.
Only the normalized current weights changed from 4/12 to 3/15, equivalent
to physical coefficients 0.40/1.20 and 0.30/1.50.

| Measure | Baseline | Proposed |
|---|---:|---:|
| Fixed validation passes | 49/50 | 50/50 |
| Final nominal balance, zero unsafe in both | 98/100 | 100/100 |
| Mean capture time on the 98 shared successes | 1.055 s | 0.692 s |
| Mean time to finish the hold on shared successes | 3.066 s | 2.392 s |
| Mean command-change RMS on shared successes | 0.02192 A | 0.01972 A |
| Largest sampled pole modulus at centered upright | 1.01064 | 0.99670 |

The proposed weights are the better candidate in this pilot. The comparison
uses one seed and the 100k checkpoint, so repeat before drawing conclusions
about the training method or changing the baseline. Logs, figures, the
reproduction commands, and `comparison.json` are under
`src/runs/compare_actor16_current_tradeoff_seed0_20261123_120136/`.

### Historical guided pilots

- The seed-0 500,000-decision ±1 A guided CUDA run completed. Its selected 64×64
  actor passed 50/50 validation and 100/100 final nominal episodes, with zero
  unsafe episodes. Training took 35 min 12.4 s and evaluation took 1 min 28.4 s.
  The largest sampled local pole modulus was 0.97525. The exact source snapshot,
  actor, SB3 checkpoint, configuration, evidence and hashes are in the
  local archive `models/nominal_guided_1a_actor64_v0/`.
- The guided 8×8 seven-input actor achieved 99/100 nominal balances at its
  selected 100k checkpoint. The six-input 8×8 comparison achieved 8/100 on
  the nominal panel. Both scored zero unsafe episodes. These narrower,
  bounded runs remain small-actor comparisons.
- The ±1 A centered LQR teacher recovered 273/273 tested starts within the
  0.40 rad arm, 0.20 rad upright error, 1.0 rad/s arm and 2.0 rad/s pendulum
  perturbation box in nominal simulation. The same 81-point grid gave 77/81
  at ±0.5 A. At the next refined box (0.45 rad, 0.225 rad, 1.125 rad/s,
  2.25 rad/s), ±1 A also recovered 273/273. At five times the original bounds
  it recovered 79/81 and two starts crossed the arm limit. See
  `src/runs/near_upright_current_limit_20261122_182243/` for the 5 ms
  held-command study and raw results. The ±1 A pilot uses a smaller teacher
  box (0.20 rad, 0.10 rad, 0.50 rad/s, 1.00 rad/s) and scales both normalized
  current penalties by four to preserve their physical costs. The default
  current limit remains ±0.5 A when the optional flag is omitted.

## Controller Map

All run paths below are relative to `src/runs/` and are ignored local artifacts.
`current` and `delta` mean reward weights, not current limits or sample periods.
Physical current remains limited to +/-0.5 A.

| Run directory | Controller | Evidence and disposition |
|---|---|---|
| `nominal_sac_actor64_seed0_v0` | current1 / delta3 | Reference: 284/284 successful in the paired current1.5 comparison, no unsafe episodes or late losses. |
| `nominal_sac_actor64_delta4_seed0_v0` | current1 / delta4 | 100/100 diagnostic, 81/81 neighborhood; fresh 97/100 at 5s, 100/100 at 30s. Smaller command changes but weak slow damping and higher total variation. Not rejected solely for being slow. |
| `nominal_sac_actor64_delta5_seed0_v0` | current1 / delta5 | 8/81 neighborhood cases unsafe; not recommended. |
| `nominal_sac_no_centering_delta4_seed0_v0` | Free-arm task | 284/284 free-task success, resting arm about 41.8 degrees. Different task; weak damping, not a demonstrated overall improvement. |
| `nominal_sac_current15_delta3_seed0_v0` | current1.5 / delta3 | 284/284 successful, no unsafe episodes or late losses. Higher current RMS/variation and faster, less-damped fast pair than the reference. |
| `nominal_sac_current15_delta4_seed0_v0` | current1.5 / delta4 | Completed 20 paired 30s cases; new actor 3/20 final balance, zero unsafe, 3 late losses. Do not promote. |

Reference model: `nominal_sac_actor64_seed0_v0/training/best/best_model.zip`.
Its standard 100-case plots are in `evaluation/sac/training_nominal/`.
The older delta4 and current1.5/delta3 standard plots are instead under
`candidate/evaluation/sac/training_nominal/`. All three standard evaluations
record 100/100 balance; they are not the broader comparison suites.

Latest delta4 plots are under `evaluation/modified/` (new current1.5/delta4) and
`evaluation/original/` (current1.5/delta3, NOT the current1/delta3 reference).
Use the validation-selected best actor, not an arbitrary final checkpoint.

## Latest Evidence

Source: `nominal_sac_current15_delta4_seed0_v0/REPORT.md`,
`evaluation/summary.json` and `evaluation/poles.json`.
Seeds 580000-580019, 30s each, 20 paired cases / 40 episodes.

| Metric | current1.5/delta3 | current1.5/delta4 |
|---|---:|---:|
| Final balance | 20/20 | 3/20 |
| Unsafe / late losses | 0 / 0 | 0 / 3 |
| First 5s current RMS (A) | 0.0951961 | 0.0828334 |
| First 5s command-change RMS (A) | 0.0272218 | 0.0243787 |
| First 5s total variation (A) | 5.8964 | 5.42718 |
| First 5s >=25Hz current RMS (A) | 0.0145933 | 0.0147919 |
| Last 5s current RMS (A) | 8.2784e-8 | 0.00310346 |

At the new actor's centered equilibrium (arm 9.1074 degrees), the fast pair is
`-120.3180 +/- j151.7246` (magnitude 193.6408, damping 0.6213), but the slow pair
is **`+0.4896 +/- j3.0268`**, locally unstable. The other discovered root is
also unstable; the root scan is non-exhaustive. Poles are principal-log
equivalents of the 5ms held-action map, including previous action as a fifth state.
They are not global stability or hardware guarantees.

For the original current1/delta3 reference, the slow pair is
`-3.871278 +/- j2.754986` (damping 0.814749), the fast pair is
`-81.045312 +/- j116.334845` (magnitude 141.782011, damping 0.571619),
and the remaining pole is `-15.571062`. In the paired fresh 100-case suite,
its current RMS was 0.0831724 A versus 0.099976 A for current1.5/delta3.
Evidence: `nominal_sac_current15_delta3_seed0_v0/FINDINGS.md` and `poles/`.

## Local Correction and Stopped Work

The simulation-only wrapper reduces the current1.5/delta3 actor's local
pendulum-velocity feedback derivative by 25%, tapered to zero outside a small
upright region. It is not a retrained actor or an LQR controller.
See [LOCAL_FEEDBACK_WORKFLOW.md](LOCAL_FEEDBACK_WORKFLOW.md).

`current15_local_feedback_probe_v0` completed 25 paired local 30s cases:
both controllers balanced in all cases, without unsafe episodes or late losses.
The correction improved local fast-pair damping to about 0.874 and reduced
first-5s command-change RMS by 7.39%. This is limited local evidence, not proof
of improvement over the original current1/delta3 reference.

**Do not resume `current15_feedback_swingup_v0`.** The user explicitly stopped
the 284-pair full swing-up comparison after 64 completed pairs to save time.
Those pairs passed final balance for both controllers, but the run is partial,
has no completion marker, and must not be described as 284-case qualification.
Stale running status after interruption is not evidence of a live process.

## Next Engineering Decision

Keep the original reference and avoid another blind scalar reward-weight increase.
Local gain corrections and LQR weight studies do not predict the gains learned
by SAC. If further pure-SAC work is requested, first investigate whether the
short training success hold and checkpoint selection favor brief balance over
persistent balance. This is a hypothesis, not an established cause or approved
change. Training currently ends after a 1s success hold; final evaluation does not.
Use a small explicitly agreed validation budget, not an automatic 284-case rerun.
One training seed and reused nominal resets do not establish robustness.

## Runtime and Reproduction

- Training: local Conda `rl-env`; numerical analysis: `src/runs/lqr_study_env/Scripts/python.exe`.
- The numerical venv inherits the Conda packages and pins local NumPy 2.2.6 /
  SciPy 1.15.3 because shared native SciPy linear algebra crashed on this machine.
- Follow [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) for environment creation,
  separate train/evaluate commands, hashes and output guards, and
  [OVERNIGHT_DELTA4.md](OVERNIGHT_DELTA4.md) for the completed compact experiment.
- VS Code tasks contain machine-specific interpreter paths; adjust them before
  using another machine. Nothing starts automatically. Existing completed output
  directories must not be reused. Do not delete guards to force a rerun.
- TensorBoard task has five runs on `http://127.0.0.1:6013/#scalars`; start it
  explicitly when needed. This URL is not a guarantee that a server is running.
- The prior Windows commit-memory exhaustion cause remains unresolved. See
  [MEMORY_INCIDENT.md](MEMORY_INCIDENT.md). Three interrupted attempts were
  deleted by user request; the current1.5/delta3 directory now holds the completed
  replacement, not those old attempts. Avoid duplicate trainers/dashboards.

## Git and Artifact Handoff

Git tracks source, tests, workflow docs and this result summary. `.gitignore`
excludes all of `src/runs/`, including models, replay buffers, frozen source,
protocols, hashes, traces, reports, event logs and the numerical venv.
**A source commit alone cannot transfer the selected nominal controller or
reproduce these comparisons on a clean clone.** The releases under `models/`
are earlier stage experiments, not these nominal actors.

For continuation elsewhere, transfer the referenced completed run directories
through artifact storage, retaining their protocol/source/config/model hashes
and reports. Include the baseline, current1.5/delta3, current1.5/delta4 and local
probe/sensitivity evidence; retain older delta4/delta5/free-arm runs if those
comparisons are needed. Preserve the stopped run as partial only. Recreate the
numerical environment; absolute interpreter paths in protocols need an explicit
relocation strategy. Replay snapshots are needed for learned-state continuation,
not merely inference, and do not serialize all environment/RNG/callback state.
No external artifact upload or availability is asserted here.

Before committing, review both staged and unstaged changes and include the new
workflow scripts/tests/docs explicitly. Do not force-add `src/runs/` or accidentally
commit large generated artifacts. Run the unit suite and check whitespace:

```powershell
& 'src/runs/lqr_study_env/Scripts/python.exe' -m unittest discover -s src -p 'test*.py' -v
git diff --check
git diff --cached --check
```

Unit tests check implementation and workflow guards, not hardware readiness or
repeatability of training. Do not restart training as a commit prerequisite.

Verification on 2026-09-10: 61 workflow/analysis tests (`test_*.py`) and 48 core
tests (`test.py`) passed in the existing numerical environment, 109 total.
The combined pattern above includes both; `test_*.py` alone misses the core suite.
Core tests include tiny training/resume smoke tests in temporary directories.
No experiment training or full evaluation was restarted for this handoff.
Local links in all seven updated documents and task JSON/dependency checks passed,
as did staged and unstaged Git whitespace checks. Git staging is a separate step:
do not assume the current index includes the latest working-tree edits or new files.