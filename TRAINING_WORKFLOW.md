# Manual Current-Penalty Experiment

Current status and decisions: [HANDOFF.md](HANDOFF.md). The replacement
current1.5/delta3 experiment and the subsequent current1.5/delta4 experiment
have both completed. These are reproduction instructions, not pending work.
For the latter's compact 20-pair protocol, see [OVERNIGHT_DELTA4.md](OVERNIGHT_DELTA4.md).

For the subsequent simulation-only 25% local feedback correction, use
[LOCAL_FEEDBACK_WORKFLOW.md](LOCAL_FEEDBACK_WORKFLOW.md), not the training tasks below.

Run training, TensorBoard, and evaluation independently. Nothing starts when
you open the workspace. This experiment starts fresh; it does not resume any
of the deleted interrupted attempts.

## Setup

Use the repository root as your working directory. On this machine, keep the
existing Conda `rl-env` and select it using **Python: Select Interpreter** in
VS Code. The interpreter is `C:/Users/pmic/.conda/envs/rl-env/python.exe`.
The tasks use that explicit executable path because VS Code incorrectly resolved
the Python extension's interpreter variable during launch on this machine.
On another machine, update the machine-specific interpreter `command` fields in
[.vscode/tasks.json](.vscode/tasks.json) to its environment's Python executable.

In an Anaconda Prompt or Conda-enabled PowerShell:

```powershell
conda activate rl-env
python -c "import sys; print(sys.executable)"
```

If Conda activation is unavailable in PowerShell, replace `python` in the
commands below with `& 'C:/Users/pmic/.conda/envs/rl-env/python.exe'`.
Bare `python` without activation may open the Windows Store on this machine.

For a new machine only, create the environment from [environment.yml](environment.yml):

```powershell
conda env create -f environment.yml
conda activate rl-env
```

The environment specification permits package-version ranges, so it is not a
bit-for-bit lockfile. Record `python -m pip freeze` and `conda list` with your
experiment records when reproducing on another machine. Matching a seed alone
does not guarantee identical results across library versions or hardware.

Full comparison requires the retained baseline directory
[src/runs/nominal_sac_actor64_seed0_v0](src/runs/nominal_sac_actor64_seed0_v0),
including its `training`, `source`, and best-model files. These generated
artifacts are not supplied by a clean Git clone; retain or transfer them.
The delta4/delta5 directories are optional historical TensorBoard comparisons.

Pole analysis uses the existing isolated interpreter
`src/runs/lqr_study_env/Scripts/python.exe` because native SciPy linear algebra
failed in the shared environment on this machine. Only if that venv is absent,
create it from the activated `rl-env`:

```powershell
python -m venv --system-site-packages src/runs/lqr_study_env
& 'src/runs/lqr_study_env/Scripts/python.exe' -m pip install --ignore-installed --no-deps numpy==2.2.6 scipy==1.15.3
& 'src/runs/lqr_study_env/Scripts/python.exe' -m unittest discover -s src -p test_lqr_weight_study.py -v
```

This venv inherits other packages from `rl-env`; it is not a fully independent
environment. Do not reinstall shared packages while Python jobs are running.

## VS Code Tasks

Choose **Terminal > Run Task** and run these tasks yourself:

1. **Current1.5: preflight** checks prerequisites without creating a run.
2. **Current1.5: train** starts fresh training and stops before final evaluation.
3. **Current1.5: TensorBoard** opens one dashboard server alongside training.
4. **Current1.5: evaluate** runs the frozen evaluation code after training finishes.

Alternatively, **Current1.5: run experiment** explicitly sequences preflight,
training, then evaluation, stopping if a stage fails. Start TensorBoard separately
alongside it. Do not also launch the individual training or evaluation tasks.
This optional sequence starts only when requested, never on workspace opening.

The task terminals remain visible in VS Code; there are no external console
windows, background training supervisors, or automatic restarts.
Keep VS Code and the training terminal open until training exits. Do not run
both a task and its equivalent terminal command; choose one launch method.

## Equivalent Commands

All commands below assume the repository root and activated `rl-env`.

### Preflight and Training

```powershell
python src/current_penalty.py --train-only
python -X faulthandler -u src/current_penalty.py --train-only --execute
```

Default output is `src/runs/nominal_sac_current15_delta3_seed0_v0`.
Preflight and training both refuse an existing output directory. For a later
independent run, add `--job-dir src/runs/<new-name>` and update the evaluation
and TensorBoard commands to point to that same directory. Do not reuse old logs.

The recipe is seed 0, 2,000,000 policy decisions at 200 Hz, actor/critics 64x64,
gamma 0.99, current weight 1.5 and action-change weight 3. The plant, centered
success criteria, current limit, and nominal task remain unchanged. Training
includes its own 50-episode validation every 100k decisions; this is necessary
for selecting the best checkpoint and is separate from final evaluation.

Successful training exits normally and writes `training_completed.json`, with
frozen candidate/baseline model hashes. `status.json` then says `trained` and
`awaiting_evaluation`. A checkpoint alone is not proof of completed training.

### TensorBoard

In another terminal:

```powershell
python -m tensorboard.main --logdir_spec "delta3_current1:src/runs/nominal_sac_actor64_seed0_v0/training/tensorboard,delta4_current1:src/runs/nominal_sac_actor64_delta4_seed0_v0/candidate/training/tensorboard,delta5_current1:src/runs/nominal_sac_actor64_delta5_seed0_v0/candidate/training/tensorboard,delta3_current1.5:src/runs/nominal_sac_current15_delta3_seed0_v0/candidate/training/tensorboard,delta4_current1.5:src/runs/nominal_sac_current15_delta4_seed0_v0/candidate/training/tensorboard" --host 127.0.0.1 --port 6013
```

Open http://127.0.0.1:6013/#scalars. For only the new run, replace
`--logdir_spec "..."` with
`--logdir src/runs/nominal_sac_current15_delta3_seed0_v0/candidate/training/tensorboard`.
The new run appears after training writes its first events. Compare success
rates and later evaluation metrics; reward curves use different cost weights.

Use one dashboard process. Stop it with Ctrl+C in its terminal. If port 6013
is occupied, use the existing dashboard or choose another port explicitly;
do not start duplicate dashboards repeatedly.

### Evaluation

After successful training, use the source snapshot saved with that run:

```powershell
python src/runs/nominal_sac_current15_delta3_seed0_v0/source/current_penalty.py --evaluate-only --job-dir src/runs/nominal_sac_current15_delta3_seed0_v0
python -X faulthandler -u src/runs/nominal_sac_current15_delta3_seed0_v0/source/current_penalty.py --evaluate-only --execute --job-dir src/runs/nominal_sac_current15_delta3_seed0_v0
```

The first command checks configuration, frozen source/model hashes, training
completion, and the recorded analysis-interpreter path without evaluating.
The second runs 100 standard nominal episodes, then 284 paired cases for each
model, local pole analysis, and the report. It never trains or changes the
checkpoint selection. The analysis interpreter path is recorded in
`protocol.json`; moving the run to another machine requires that path to be
made available or an explicitly documented relocation.

Evaluation creates `evaluation_started.json` exclusively and refuses a second
attempt in the same directory, including after interruption. Preserve partial
evidence and diagnose the failure before arranging a separate evaluation copy;
do not simply delete the guard and mix old and new results.

## Outputs and Reproducibility

Paths below are relative to the job directory:

| Output | Meaning |
|---|---|
| `protocol.json`, `source/` | Declared task, settings, cases, hashes, and frozen Python source |
| `candidate/training/config.json`, `training.json` | Training environment and learner settings |
| `candidate/training/tensorboard/` | Training and validation curves |
| `candidate/training/best/best_model.zip` | Best validation checkpoint, not necessarily the final actor |
| `candidate/training/resume/` | Rolling model/replay recovery snapshot |
| `reference/`, `model_hashes.json` | Frozen models used by final evaluation |
| `training_completed.json` | Training complete; evaluation may now start |
| `candidate/evaluation/` | Standard 100-case evaluation |
| `comparison/`, `summary.json` | Paired trajectories, reliability and current metrics |
| `poles/`, `pole_analysis.log` | Local stability, damping and small-signal analysis |
| `review.json`, `FINDINGS.md`, `completed.json` | Final screens, readable report, complete evaluation marker |
| `failed.json`, `status.json` | Python exception/status evidence, not a live process heartbeat |

Cases include 100 reused diagnostic seeds, 81 boundary-neighborhood cases,
100 fresh 30-second seeds (480000-480099), and three earlier slow cases.
These are fixed for reproduction; after inspecting results they are no longer
unseen evidence for future tuning. One training seed and nominal simulation
do not establish robustness or hardware safety. Local poles are not a global
stability guarantee. No model is automatically promoted.

## Stopping and Memory

Ctrl+C stops the chosen terminal process. A native crash, terminal closure,
reboot, or interruption may leave stale status without `failed.json`. Do not
rerun training over that directory. Preserve it and choose a fresh job name;
resume is not automatic and this guide's training command always starts fresh.

The earlier machine-wide memory exhaustion is documented in
[MEMORY_INCIDENT.md](MEMORY_INCIDENT.md). Rebooting cleared the memory state
but did not identify or fix its cause. Watch **Task Manager > Performance >
Memory > Committed**, not only the trainer's RAM usage. Before and during
training, this read-only PowerShell command gives a snapshot:

```powershell
Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | Select-Object AvailableMBytes, CommittedBytes, CommitLimit, PoolPagedBytes, PoolNonpagedBytes
```

No automatic memory monitor or process killer is installed. If committed memory
keeps rising toward the limit (for example, above 85%), stop the experiment
and capture diagnostics before another failure. Avoid parallel training jobs,
duplicate dashboards, sleep, and reboot while training is intended to run.

## Focused Tests

```powershell
python -m unittest discover -s src -p test_current_penalty.py -v
python -m unittest discover -s src -p test_current_weight.py -v
```

These test workflow separation, protocol/reward rules, current metrics and
guards. They do not run a full 2M-decision experiment or establish its outcome.
