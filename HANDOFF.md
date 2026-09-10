# Development Handoff

Updated 2026-09-10 after the completed current1.5/delta4 experiment.
This is the current decision record; older assessments and generated pass/fail
labels retain their original criteria. No controller is automatically promoted.

## Current Decision

Use the original **current1 / delta3** pure-SAC actor as the development reference.
It has the strongest overall nominal evidence, not just the fastest settling.
The goal is reliable swing-up, sustained balance, smooth current and good damping
in a small teaching project. Slower settling is acceptable. Arm centering is not
intrinsically required, but the recent reward comparisons kept the centered task.
Do not change task criteria silently between comparisons.

The latest current1.5/delta4 run completed normally but is a behavioral regression:
only 3/20 cases achieved final balance versus 20/20 for current1.5/delta3.
Its centered equilibrium has an unstable slow pole pair. Do not use it as the
new reference, and do not interpret its failure as merely slow settling.
No further training or evaluation is authorized by this handoff.

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