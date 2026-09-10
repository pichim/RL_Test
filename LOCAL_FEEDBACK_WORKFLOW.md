# Nonlinear Local Feedback Probe

## Current Status (2026-09-10)

The 25-pair local probe completed: both controllers achieved final balance in
all cases, with no unsafe episodes or late losses. The correction reduced
first-5s command-change RMS by 7.39%; this is local evidence, not qualification
against the original current1/delta3 reference. The default output already exists.
The subsequent full swing-up comparison was stopped by user request after
64/284 pairs. **Do not resume it or report it as complete.**
See [HANDOFF.md](HANDOFF.md) for current decisions and evidence locations.

This standalone simulation compares the completed current1.5 SAC controller
against the same actor with a local feedback correction. It does not train,
rewrite a model, change the reward, or use LQR. Slower settling is acceptable:
there is no settling-speed nonregression gate and no automatic promotion.

## Run It

Open this workspace and choose **Terminal > Run Task**:

1. **Local feedback: preflight** validates prerequisites and the local poles.
2. **Local feedback: run** repeats preflight, then runs the comparison.

These tasks use the existing isolated numerical environment under
`src/runs/lqr_study_env`. Nothing starts automatically. Do not use
`Current1.5: train` for this experiment; the completed training is its input.

Equivalent PowerShell commands, from the repository root:

```powershell
& 'src/runs/lqr_study_env/Scripts/python.exe' -X faulthandler -u src/local_feedback_probe.py
& 'src/runs/lqr_study_env/Scripts/python.exe' -X faulthandler -u src/local_feedback_probe.py --execute
```

Use tasks OR commands, not both. Preflight creates no run output. The full
comparison runs 25 initial conditions for both controllers, each for 30 simulated
seconds (50 episodes). Keep its terminal open until it exits. This is simulation
time, not a promise of 25 minutes of wall-clock execution.

The output directory is `src/runs/current15_local_feedback_probe_v0`. Existing
output is refused, even if a prior run was interrupted. For a repeat:

```powershell
& 'src/runs/lqr_study_env/Scripts/python.exe' src/local_feedback_probe.py --execute --output src/runs/current15_local_feedback_probe_v1
```

No TensorBoard server is needed: this is deterministic evaluation, not training.
Inspect the generated PNG plots and CSV traces directly.

## What Is Modified

At the studied upright equilibrium, arm angle 9.3499 degrees, the normalized
action derivative with respect to pendulum velocity is -1.98330. The correction
targets 75% of that derivative, -1.48748, without changing the other local
derivatives:

```text
modified_action = clip(original_action
                       + gate * (0.75 - 1) * (-1.98330) * omega2, -1, 1)
```

The actual code uses the full-precision derivative from the verified sensitivity
study. Both controllers observe the actual previous held action. Applied current
remains limited to +/-0.5 A, with 200 Hz action updates and the existing nominal
nonlinear 1 kHz RK4 plant. There is no filter, delay or hidden handover state.

The gate is a product of four smoothstep tapers. It is one inside all inner
bounds, zero outside any outer bound, and smoothly transitions between:

| Coordinate (absolute value) | Inner | Outer |
|---|---:|---:|
| Arm displacement from the studied equilibrium | 0.05 rad | 0.15 rad |
| Pendulum upright error | 0.05 rad | 0.15 rad |
| Arm velocity | 0.5 rad/s | 1.5 rad/s |
| Pendulum velocity | 0.5 rad/s | 1.5 rad/s |

These bounds are explicit experimental choices, not a certified region of
attraction. The taper is an added controller design; it is not part of the
original linear gain scan. Outside it, the action equals the original actor's
action at the same observation. Whole trajectories can still differ after
entering the taper or because their previous actions differ.

The expected local continuous-equivalent poles at the expansion point are:

```text
-186.4535 +/- 103.4900j
-39.7741
-3.2677 +/- 1.3572j
```

These predict local behavior only. Preflight reconstructs the modified nonlinear
map's Jacobian and checks its discrete poles against the sensitivity study.
The saved float32 actor has a small equilibrium residual; the model is not
rebased or edited to remove it.

## Cases and Interpretation

The paired cases are the equilibrium itself and both signs of each individual
coordinate perturbation at three amplitudes:

- Arm and pendulum angles: 0.02, 0.10, 0.16 rad.
- Arm and pendulum velocities: 0.2, 1.0, 1.6 rad/s.

This covers the gate interior, transition and just outside its support. Both
controllers start with identical physical state and previous action zero.
It is not a random-reset swing-up test, a general region-of-attraction study,
or hardware qualification.

Safety, final balance, late balance losses, state/current traces, and paired
first/last five-second current metrics are reported. Finish times remain visible
but are not a pass/fail gate. Final balance still means balance within the existing
tolerances during the final one-second hold, within the finite 30-second horizon.
Five-second success and late-loss columns are diagnostic, not speed requirements.

The output includes final offsets from the expansion point, because meeting the
balance tolerances does not mean exact return to that point. Inspect ringdown
and any taper transitions in the plots. The report does not fit a nonlinear
damping ratio or claim that lower RMS necessarily means better damping.

## Outputs

Paths below are relative to the output directory:

| Path | Contents |
|---|---|
| `REPORT.md` | Readable comparison with safety, balance and current metrics |
| `summary.json` | Counts and paired current means; speed gate explicitly disabled |
| `episodes.csv` | Per-controller initial states, finish times, final offsets, safety |
| `current_windows.csv` | Per-case complete-window RMS, action changes, variation and HF RMS |
| `plots/<case>/<controller>/` | State/current PNG and full CSV trace for every episode |
| `protocol.json` | Cases, configuration, taper bounds, gain checks, source hashes, runtime versions |
| `candidate.zip`, `sensitivity.json`, `source/` | Frozen model, sensitivity input and Python source copies |
| `completed.json` | Written only after the full comparison and report finish |
| `status.json`, `failed.json` | Progress and caught Python exceptions |

Incomplete windows are excluded from both controllers' paired current averages;
unsafe episodes remain in the safety counts. The script never overwrites the
original training or sensitivity outputs. Ctrl+C stops the run; interruption or
native process loss can leave stale status without a failure marker.

## Prerequisites and Tests

Retain the completed job `src/runs/nominal_sac_current15_delta3_seed0_v0` and
`src/runs/current15_pole_sensitivity_v1/analysis.json`. They are local generated
artifacts and are not provided by a clean Git clone. Preflight checks their
configuration/model identity and relevant current source against the completed
experiment. A mismatch is a stop condition, not a reason to bypass the check.

The `--job-dir` and `--sensitivity` options can point to relocated copies;
`--output` must always be new. Archive outputs together with their source and
environment details. If rerunning from a saved source directory, pass these
three paths explicitly because imported default paths are source-relative.

Environment setup and the earlier memory precautions are in
[TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md). Keep one simulation process running
at a time and watch Windows committed memory if the earlier issue recurs.

```powershell
& 'src/runs/lqr_study_env/Scripts/python.exe' -m unittest discover -s src -p test_local_feedback_probe.py -v
```

The setup was checked with unit tests, a real read-only pole preflight and a
20 ms integration smoke check. The full local comparison subsequently completed;
see the status above. Commands here document reproduction into a new output,
not an instruction to restart the stopped swing-up comparison.
