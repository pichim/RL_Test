# Furuta swing-up: direct-current SAC

## Current stage (2026-11-23)

The [16×16 local-teacher actor](models/nominal_scratch_actor16_v1/README.md)
at 225,000 decisions is the reference for the periodic nominal task. Its
recovery-teaching counterpart reached perfect validation earlier and stayed
there for three consecutive checks in the seed-0 pilot. Both selected actors
passed 100/100 nominal starts, 50/50 fresh starts, 81/81 upright starts, and
30/30 fixed post-capture starts with zero unsafe episodes.

See [HANDOFF.md](HANDOFF.md) for the current decisions and
[the scratch run commands](#matched-250k-scratch-runs-with-lqr-recovery) for
reproduction. The release directory contains both selected actors, the common
task configuration, and compact evidence. Full run outputs in `src/runs/`
are local artifacts. The reference is for nominal simulation.

The project trains SAC to swing up and balance a nonlinear Furuta pendulum at
200 Hz. Guided profiles use LQR to teach the actor during training and export
a single neural actor. The older bounded profiles and optional hybrid
evaluation are still available. The observation and current-change reward
follow the structure of the [MathWorks Quanser QUBE example](https://www.mathworks.com/help/reinforcement-learning/ug/train-agent-offline-to-control-quanser-qube-pendulum.html);
this project uses its own Furuta plant model and commands motor current.

## Versioning

Every saved `config.json` identifies the task it uses.

| Task version | Behavior |
|---|---|
| `matlab_hardware_sac_pure_v0` | Bounded arm travel and the original reward in the legacy profiles |
| `periodic_arm_sac_v1` | Periodic arm and experimental time-cost reward |
| `periodic_arm_legacy_reward_v1` | Periodic arm and the original reward structure with wrapped arm error |

The two periodic tasks allow arm crossings at ±π and score arm orientation
modulo a turn. Evaluation reads the saved task. Training seed, network size,
and budget are recorded separately. Scripts that require byte-for-byte source
matches to an older run need that run's source snapshot.

## Controller structure

```text
stage 1: encoder state -> SAC at 200 Hz -> held current -> nominal plant
stage 2: encoder state -> SAC -> hidden continuous 100 Hz response -> nominal plant
stage 3c: encoder state -> SAC -> hidden 0--1 sample delay
                               -> hidden 50--100 Hz response -> randomized plant

evaluation: SAC-only full episode
            SAC -> sustained compatible capture -> fast-feedback LQR
```

The mechanical simulation uses a 1 kHz RK4 grid. Each SAC current command is
held for five plant steps. Stage 1 applies it directly. Stages 2 and 3 use
`di/dt = 2*pi*f_c*(i_command - i)`: Stage 2 fixes `f_c = 100 Hz`, while
Stage 3c samples it from the active curriculum interval. This state is
integrated jointly with the mechanics at every RK4 stage but is not added to
the SAC observation. It represents unobserved actuator uncertainty, not a
deployment slew limiter. Hardware LQR current control is intended to run at
10 kHz, so evaluation recomputes LQR feedback directly at the RK4 stages and
always bypasses both uncertainty models. In hybrid evaluation, delay and filtering
are active only during the SAC portion and are removed at LQR handover. A
delay sample is one 200 Hz policy period (`5 ms`), not one 1 kHz integration
step. The delay and actuator state are deliberately absent from the SAC
observation.

## Current-penalty 1.5 experiment

`src/current_penalty.py` trains a fresh seed-0 nominal SAC actor for 2M decisions,
changing only `current_weight` from 1 to 1.5. The action-change weight stays 3;
the centered success definition, safety limits, network and learner settings
remain unchanged. Old configurations without `current_weight` default to 1.

```powershell
conda activate rl-env
python src/current_penalty.py --train-only
python -X faulthandler -u src/current_penalty.py --train-only --execute
```

Use **Terminal > Run Task** for `Current1.5: preflight`, `Current1.5: train`,
`Current1.5: TensorBoard`, and `Current1.5: evaluate`. Select the `rl-env`
interpreter first. Tasks never start automatically.
See [the manual reproduction guide](TRAINING_WORKFLOW.md) for setup, separate
commands, output locations, completion checks, and interruption handling.

The dry run verifies the frozen weight-3 baseline and training runtime.
Training refuses an existing job directory. It saves a protocol and source
snapshot, trains, and stops at `training_completed.json`. Run evaluation
separately to test the best training-validation checkpoint on the standard
100 cases, then compare both models on 284 paired cases each.
These include 100 fresh 30-second cases (480000-480099), 100 reused diagnostic
cases, 81 boundary-neighborhood cases and three previous slow cases.

The final stage uses the isolated numerical environment described below for
local poles and +/-0.02 rad small-signal tests. `summary.json`, `review.json`
and `FINDINGS.md` report reliability, paired current windows and local damping.
No holdout-based checkpoint selection or automatic model promotion is performed.
`completed.json` is written only after evaluation; exceptions write `failed.json`.
Status-file locks fall back to uniquely named status events.
The legacy `--execute` command without `--train-only` still runs the full pipeline.

Default output: `src/runs/nominal_sac_current15_delta3_seed0_v0`.
Training TensorBoard logs are under `candidate/training/tensorboard`.

## Local actor-gain sensitivity

[src/pole_sensitivity.py](src/pole_sensitivity.py) examines hypothetical local
changes to the completed current1.5 actor's four physical-state derivatives
and previous-action derivative. It verifies the saved model/source hashes,
reproduces the original poles, tracks all five discrete modes, and screens
fast-pair improvements against stability and slow-mode degradation. It writes
only a new analysis directory; it does not train or modify a model.

Results and interpretation:
[src/runs/current15_pole_sensitivity_v1/FINDINGS.md](src/runs/current15_pole_sensitivity_v1/FINDINGS.md).
These generated results require the completed experiment and are not part of
a clean Git clone. To reproduce from the repository root, choose a new output:

```powershell
& 'src/runs/lqr_study_env/Scripts/python.exe' -m unittest discover -s src -p test_pole_sensitivity.py -v
& 'src/runs/lqr_study_env/Scripts/python.exe' src/pole_sensitivity.py --output src/runs/current15_pole_sensitivity_v2
```

The scan holds the equilibrium fixed and changes local feedback derivatives,
not reward weights. It does not predict what SAC will learn after retraining.

## Local reward-weight study

`src/lqr_weight_study.py` is a read-only, 200 Hz upright LQR surrogate for the
SAC running cost. It includes previous normalized action, the action-change
penalty, and the post-action state-cost timing. It sweeps one weight at a time
at gamma 1.0 and 0.99, without training SAC or modifying controllers.

The completed study and interpretation are in
[`src/runs/lqr_weight_study_v1/FINDINGS.md`](src/runs/lqr_weight_study_v1/FINDINGS.md)
(local generated artifacts, not tracked). Full poles, numerical checks,
runtime versions, and source snapshots accompany the findings.

SciPy native linear algebra crashed in the shared `rl-env` on this machine.
The study was verified in `src/runs/lqr_study_env`, an isolated venv with local
NumPy 2.2.6 and SciPy 1.15.3 that inherits other packages from `rl-env`.
It does not change the training environment. To rerun locally in PowerShell:

```powershell
& 'src/runs/lqr_study_env/Scripts/python.exe' -m unittest discover -s src -p test_lqr_weight_study.py -v
& 'src/runs/lqr_study_env/Scripts/python.exe' src/lqr_weight_study.py --output src/runs/lqr_weight_study_v2
```

The output directory must be new. Pole proximity in this unconstrained local
model is not evidence of improved nonlinear SAC swing-up, safety, or smoothness.

## Observation and reward

The seven-element observation is:

```text
[
    sin(theta1),
    cos(theta1),
    omega1 / 30,
    sin(upright_error),
    cos(upright_error),
    omega2 / 30,
    previous normalized current command,
]
```

Here, `theta2 = 0` is downward, `theta2 = +/-pi` is upright, and
`upright_error = wrap(theta2 - pi)`.

The running reward is:

```text
reward = F - 0.1 * (
    arm_angle_weight*theta1^2
    + upright_error^2
    + arm_velocity_weight*omega1^2
    + pendulum_velocity_weight*omega2^2
    + current_weight*action^2
    + action_change_weight*(action - previous_action)^2
)
```

The current staged experiments use `arm_angle_weight = 1.5`,
`arm_velocity_weight = 0.015`, `pendulum_velocity_weight = 0.010`, and
`action_change_weight = 3.0`. Relative to the completed `8 -> 8` actor run,
this changes only the arm-velocity weight from `0.020` to `0.015`. The absolute
normalized action and its change remain penalized.

`F` is one while `|theta1| <= 5*pi/8`, `|omega1| <= 30 rad/s`, and
`|omega2| <= 30 rad/s`. It supplies the dense safe-operation reward used by the
completed run; a violation sets `F` to zero and terminates the episode.

The action-change weight is deliberately larger than the MathWorks value. The
MathWorks example commands voltage and its `0.3` value did not
sufficiently discourage rapid reversals for this project's direct-current
actuator. It is exposed as `--action-change-weight` for controlled experiments.

## Training success

SAC is trained to swing up and balance without LQR. Training
success requires all of the following continuously for `1.0 s`:

- upright error no greater than 5 degrees;
- arm displacement from zero no greater than 0.25 rad;
- `|omega1| <= 0.5 rad/s`;
- `|omega2| <= 1.0 rad/s`.

Success terminates the episode and adds the continuation-value bonus of `100`
for `gamma = 0.99`. This trains complete swing-up, settling, and a sustained
pure-SAC balance rather than terminating at the wider LQR capture condition.
The five-second limit remains the maximum for unsuccessful episodes. Training
resets cover a broad safe state distribution: the arm angle is uniform in
`[-pi/2, pi/2]`, the pendulum angle is uniform over its full wrapped circle
`[-pi, pi]`, and both velocities are uniform in `[-2, 2] rad/s`. The arm range
retains margin to its `+/-5*pi/8` travel limit.

Stages 1 and 2 train on the nominal plant. Stage 3c is a continuation curriculum
whose uncertainty fraction ramps linearly for one million decisions and then
stays at full range for the remaining 500,000 decisions. At full range, motor
torque, both masses, both rotary inertias, and both damping coefficients vary independently
by `+/-10%`; the RL current-filter cutoff varies uniformly from `50` to
`100 Hz`; and the hidden RL action delay is sampled from `0` or `1`
policy period. The arm reset-velocity half-range widens from the nominal
`2 rad/s` to `0.5 rotations/s` (`pi rad/s`); the pendulum range widens to
`1.5 rotations/s` (`3*pi rad/s`). The final 500,000
decisions use learning rate `1e-4` instead of `3e-4`. The deterministic
checkpoint callback remains nominal in every stage so scores are comparable;
the standalone evaluator always tests nominal and full-range randomized cases.

The default SAC setup remains intentionally small:

- one million decisions and 40,000 warm-up decisions;
- an `8 -> 8` actor and `64 -> 64` critics;
- replay capacity one million and batch size 256;
- one gradient update every two decisions;
- automatic entropy coefficient initialized at `0.1`;
- TensorBoard, checkpoints, deterministic evaluation, and one rolling
  resumable model/replay snapshot every 100,000 steps.

Checkpoint selection first maximizes deterministic final-balance success. For
equal success rates it prefers mean reward, which combines the task errors,
control effort, and action-change penalty defined above. Training validation
uses 50 fixed nominal episodes by default; `--evaluation-episodes` changes that
count. Final selection must still use unseen seeds in the standalone evaluator,
which reports current smoothness and arm-angle/velocity trajectory shape.

## Evaluation

The default evaluator runs both complete five-second controller variants with
identical seeds:

```text
SAC-only swing-up and balance
SAC -> 0.10 s sustained compatible capture -> relative-angle LQR balance
```

Success requires tight balance throughout the final `1.0 s`. The hybrid uses
the same sustained capture duration as the supervisor definition; it does not
switch on a single compatible sample.

Evaluation runs 100 episodes in each genuinely distinct mode:

- broad training reset with the nominal plant;
- broad training reset with randomized parameters;
- near-downward, zero-velocity hardware start with randomized parameters.

Use `--mode nominal` to run only the first scenario, as in the compact
nominal-only teaching workflow. The default `--mode all` preserves the complete
three-scenario evaluation used by existing qualification protocols.

The nominal mode uses arm angle in `[-pi/2, pi/2]`, arbitrary wrapped pendulum
angle, and both velocities in `[-2, 2] rad/s`. In a curriculum run, the
randomized broad-reset mode instead uses the saved full reset-velocity range,
up to `+/-1.5 rotations/s`. The third mode uses arm and pendulum angles in
`[-pi/4, pi/4]`
around the hanging equilibrium with zero initial velocity. This avoids treating
two identical broad-reset seed batches as different robustness scenarios.

The randomized cases use every range saved with the run, including independent
mechanical parameter scales, current-filter cutoff, RL action dead time, and
reset velocities. Reports distinguish the 200 Hz requested-current sequence,
the delayed plant command, and the 1 kHz applied current; these are identical
in stage 1 and can differ during the robust curriculum.
They include magnitude, saturation, applied-current slew/variation,
command changes, command total variation, second differences, reversals,
high-frequency command power, velocity RMS and peaks, arm excursion,
capture-to-handover delay, supervisor switches, LQR fraction, and handover
current mismatch. Settling diagnostics additionally report the beginning and
completion of the final uninterrupted one-second balance interval,
post-capture arm excursion and arm-velocity RMS, and final-window arm-error and
arm-velocity metrics. Detailed traces and plots contain both requested and
applied current.

`--episodes` changes the number of episodes per mode and `--base-seed` selects
the first nominal seed. The three modes use consecutive non-overlapping seed
ranges. Every result directory records the exact ranges in `evaluation.json`.
Summary files report mean, 95th-percentile, and worst episode values for the
principal command-smoothness, arm, velocity, and final-settling metrics.

For SAC-only episodes, current statistics cover the complete five-second run.
For hybrid episodes, fields prefixed with `swingup_` cover only the SAC segment
before handover. The hybrid arm-error criterion is relative to the handover arm
angle, whereas SAC is evaluated relative to absolute arm angle zero.

## Files

```text
src/furuta_model.py  nonlinear mechanics and continuous actuator response
src/furuta_env.py    observation, reward, randomized plant, and training success
src/controllers.py   relative-angle LQR and hybrid supervisor
src/train.py         scratch, actor-transfer, exact resume, or curriculum training
src/evaluate.py      matched SAC-only and hybrid evaluation
src/qualify.py       hash-checked frozen Stage-3c qualification protocols
src/stress_evaluate.py deterministic corner and long-duration evaluation
src/test.py          focused dynamics, task, handover, LQR, and smoke tests
models/stage3c_half_rps_v0/ selected controller, protocols, and evidence
models/stage3b_v0/   superseded historical controller and failed holdouts
.vscode/launch.json  one-click scratch, transfer, resume, and evaluation profiles
```

## Install and test

Use Python 3.12, as specified in `environment.yml`.

If the project Conda environment already exists, activate it and run the tests:

```bash
conda activate rl-env
python src/test.py
```

To create an isolated environment without Conda instead:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/test.py
```

## Linux CUDA setup and nominal benchmark

For the RTX 3080 desktop with Miniforge and an NVIDIA driver compatible with CUDA
13.2, create the Python and pip environment, install the CUDA build of PyTorch, then
install the project requirements. Run the creation and installation commands only
when setting up a new environment. The desktop setup uses the PyTorch 2.14.0
CUDA 13.2 wheel. The generic `environment.yml` leaves PyTorch to dependency
resolution.

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda create -n rl-env python=3.12 pip
conda activate rl-env
python -m pip install 'torch==2.14.0+cu132' --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
python -m pip check
```

On the desktop, activate the existing environment from a VS Code Remote SSH
terminal at the repository root. Use `tmux new -s furuta-gpu` first if the run
should survive an SSH disconnect. The commands below use a new output directory
and the non-interactive plotting backend for SSH.

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rl-env
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_dir="src/runs/nominal_sac_actor64_linux_cuda_seed0_$(date +%Y%m%d_%H%M%S)"
python -u src/nominal.py --require-cuda --job-dir "$run_dir"
```

Check that preflight reports the RTX 3080, then start training and automatic
evaluation in the same terminal.

```bash
python -u src/nominal.py --require-cuda --job-dir "$run_dir" --execute
```

The trainer should print `Using cuda device`. The preflight performs a real CUDA
matrix multiplication and stops before creating a run directory if CUDA is
unavailable. The nominal simulation and final evaluation still use the CPU.
Training and final evaluation wall times are written separately to
`status.json` and `FINDINGS.md`. Training time includes periodic checkpoint
validation and saving. TensorBoard events are under
`training/tensorboard/`, and the selected model is
`training/best/best_model.zip`. Detach from tmux with **Ctrl+B**, then **D**.
Reconnect with `tmux attach -t furuta-gpu`.

For a CPU comparison on the same desktop, run this separately after the CUDA job
finishes. Both commands use the same nominal training and evaluation settings
and one CPU math thread.

```bash
cpu_dir="src/runs/nominal_sac_actor64_linux_cpu_seed0_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES= python -u src/nominal.py --job-dir "$cpu_dir" --execute
```

### Guided nominal SAC on CUDA

Use `--guided` for energy shaping, reflection symmetry, and LQR teaching on
CUDA. The 500,000-decision pilot below enables ±1 A. The 5 ms control period,
1 ms physics step, and centered balance targets are the same as the reference.
The command without `--max-current` retains the original ±0.5 A limit.

From the repository root in a VS Code Remote SSH shell, start a dedicated
terminal session with `tmux new -s furuta-guided-1a`. In that session run:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rl-env
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_dir="src/runs/nominal_sac_guided_cuda_1a_seed0_$(date +%Y%m%d_%H%M%S)"
python -u src/nominal.py --guided --max-current 1 --timesteps 500000 --job-dir "$run_dir"
```

After preflight reports the RTX 3080 and ±1 A, launch from the same terminal:

```bash
python -u src/nominal.py --guided --max-current 1 --timesteps 500000 --job-dir "$run_dir" --execute
```

The pilot validates every 100,000 decisions and evaluates the best checkpoint
on the standard 100 nominal resets. Detach with **Ctrl+B**, then **D**, and
reconnect with `tmux attach -t furuta-guided-1a`.

The 1 A option scales the normalized current and current-change weights
from 1 and 3 to 4 and 12, preserving their cost for the same physical
current and current change. The 1 A teacher samples starts within
±0.20 rad arm angle, ±0.10 rad upright error, ±0.50 rad/s arm speed and
±1.00 rad/s pendulum speed. This is inside the tested recovery envelope.
The centered LQR labels short nonlinear recovery trajectories using 5 ms
held commands. The actor gets 500 supervised updates before SAC rollout
collection, then a supervised command and near-origin slope loss alongside
each SAC actor update. The actor's mean is antisymmetric under reflection, its
exploration scale is invariant, and the critics are invariant under simultaneous
observation and action reflection. Training adds a bounded discounted
pendulum-energy potential difference with a default weight of 0.25.
`--energy-weight 0` disables shaping while retaining the same reflection and
LQR teaching for the control comparison. Use fresh output directories and matched
`--seed` values for comparisons.

Checkpoint selection and the final nominal evaluation use the original
unshaped task. `training/actor.pt2` is the standalone deterministic neural
controller. It takes an N×7 float32 batch in the existing SAC observation
order and returns normalized commands, which are multiplied by the current
limit. Load it with `torch.export.load(path).module()`.
`training/local_diagnostics.json` gives the actor and teacher gains
and the actor's finite-difference 5 ms closed-loop poles near upright.
The LQR teacher operates during training. An exactly symmetric deterministic
actor outputs zero at a perfectly centered, motionless hanging-down start.
A separate startup decision would be needed if exact-rest startup is required.

### Matched 8×8 actor pilot

The local archive `models/nominal_guided_1a_actor64_v0/` contains the
64×64 checkpoint, standalone actor, source, and results. To test
a smaller actor, use the same CUDA session setup above and a fresh output path.

```bash
run_dir="src/runs/nominal_sac_guided_cuda_1a_actor8_seed0_$(date +%Y%m%d_%H%M%S)"
python -u src/nominal.py --guided --max-current 1 --timesteps 500000 \
  --actor-hidden-sizes 8 8 --job-dir "$run_dir"
python -u src/nominal.py --guided --max-current 1 --timesteps 500000 \
  --actor-hidden-sizes 8 8 --job-dir "$run_dir" --execute
```

Preflight should display `actor_network: [8, 8]` and `critic_network: [64, 64]`.
This is a fresh seed-0 run with the same energy shaping, symmetry, LQR teaching,
current limit, evaluation and decision budget as the 64×64 pilot. The exported
deterministic 7 → 8 → 8 → 1 network has 145 parameters.

### Six-input 8×8 actor comparison

Use the guided CUDA setup above. The comparison uses the first 100,000
decisions, matching the selected checkpoint of the seven-input 8×8 actor.

```bash
run_dir="src/runs/nominal_sac_guided_cuda_1a_actor8_no_history_seed0_$(date +%Y%m%d_%H%M%S)"
python -u src/nominal.py --guided --max-current 1 --timesteps 100000 \
  --actor-hidden-sizes 8 8 --actor-no-previous-command --job-dir "$run_dir" --execute
```

The standalone actor takes the first six observation features. The training
critics still see the full seven-feature observation and the candidate action,
with two 64-unit hidden layers each. The environment keeps the previous command
for its current-change penalty. Reflection symmetry and LQR teaching apply
as before. The exported model has 137 parameters and takes an N×6 float32 tensor.

### 16×16 LQR-region pilot with full arm travel

With the guided CUDA environment active, prepare a fresh seed-0 run for 500,000
control decisions. Run the first Python command for preflight and the second
when ready to start.

```bash
run_dir="src/runs/nominal_sac_guided_cuda_1a_actor16_lqr_region_seed0_$(date +%Y%m%d_%H%M%S)"
args=(--guided --max-current 1 --timesteps 500000 --actor-hidden-sizes 16 16
      --current-weight 0.75 --action-change-weight 3.75
      --reset-speed-rad-s 6 --full-arm-travel --lqr-only-inside --job-dir "$run_dir")
python -u src/nominal.py "${args[@]}"
python -u src/nominal.py "${args[@]}" --execute
```

Preflight should show the CUDA device, 16×16 actor, 64×64 critics, ±1 A,
and a 5 ms control period. The two CLI weights use the 0.5 A reference
convention. At 1 A the run stores 3 and 15, which the reward scale of 0.10
turns into the physical costs `0.30 * I² + 1.50 * (I - I_previous)²`
with commanded current in amperes. The actor keeps all seven inputs and
pendulum-energy shaping retains its weight of 0.25.

The simulator samples both angles over ±π and both initial angular velocities
over ±6 rad/s. Arm excursions beyond ±π end the episode as unsafe. A reset
near that limit with outward velocity can end before the controller has time
to brake. Training validation and the 100-episode final evaluation use this
wider reset distribution, so scores from the older distribution describe a
different test.

Inside the LQR box, the actor receives direct updates only from LQR examples.
The box is ±0.20 rad arm angle, ±0.10 rad upright error, ±0.50 rad/s arm speed,
and ±1.00 rad/s pendulum speed. The teacher supplies commands and near-origin
slope targets during every actor update. Outside the box, SAC also updates
the actor using replay. The critics train on the full replay buffer. After
the standard random replay warm-up, rollout actions come from the actor with
reduced exploration inside the box.

The deployed artifact is the single symmetric neural actor in
`training/actor.pt2`. Inspect `training/local_diagnostics.json` for the
local gains and closed-loop poles of the validation-selected actor and
`evaluation/summary.txt` for its swing-up and balance results.

### Periodic-arm 16×16 pilot

The 500,000-decision pilot uses the same 5 ms control period, ±1 A current
limit, seven-input symmetric 16×16 actor, 64×64 critics and current weights
of 0.30 and 1.50 in amperes. Both angles start within ±π and both velocities
within ±6 rad/s. Arm orientation is periodic. A full turn counts as zero
orientation, and crossing ±π can continue through the simulation. Traces
retain unwrapped arm angle. The velocity and numerical guards still apply.

A smooth periodic arm cost, a 0.01 time cost per decision, a 250 balance
bonus, and a 250 unsafe penalty replace the old arm-position and survival
terms. Pendulum energy shaping stays at 0.25. Discount is 0.999 per control
decision. This reward favored successful balance over timeout on all 34
paired trajectories audited from the earlier 200k and 500k checkpoints.

Every 25,000 decisions, the run evaluates the same 50 validation starts and
saves a checkpoint. It selects by balance success first. The LQR target
holds across five previous-current values at sampled faces, edges, and
corners of the upright box. The actor gets 4,000 teacher-only updates before
rollout collection. The slope anchors and trajectory examples are also used.

From the repository root with `rl-env` active, preflight the run and then
add `--execute` in the same shell when ready.

```bash
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_dir="src/runs/nominal_sac_guided_cuda_periodic_actor16_seed0_$(date +%Y%m%d_%H%M%S)"
baseline="src/runs/nominal_sac_guided_cuda_1a_actor16_lqr_region_seed0_20261123_131002/training/best/best_model.zip"
python -u src/nominal.py --periodic-pilot --comparison-model "$baseline" --job-dir "$run_dir"
python -u src/nominal.py --periodic-pilot --comparison-model "$baseline" --job-dir "$run_dir" --execute
```

`training/validation_metrics.json` records balance, capture, teaching-box
entry, unsafe terminations, timeouts, and discounted return at each check.
The selected model and the previous actor are evaluated using the same
100 final seeds in `evaluation/` and `baseline_evaluation/`. This panel
includes all 15 starts that crossed the old arm limit. The
`upright_benchmark.json` report compares LQR command error and recovery on
81 local starts, and `training/actor.pt2` contains the selected standalone
actor. The previous actor scored 100/100 on the periodic final panel and
81/81 on the local recovery panel, with a mean LQR command error of 0.033 A.
The `--comparison-model` argument can be omitted if the earlier checkpoint is
unavailable.

Legacy configs retain the bounded task rules. Workflows that require
byte-for-byte core source matches to historical runs need the corresponding
historical source snapshot.

### Matched 250k scratch runs with LQR recovery

Both variants start from random weights and empty replay. They use the
periodic simulator, ±π angle and ±6 rad/s speed resets, ±1 A, 5 ms, a symmetric
seven-input 16×16 actor, 64×64 critics, and the 0.30 / 1.50 physical current
weights. They use the previous reward with wrapped arm error and discount
0.99, plus energy shaping with weight 0.25. Pretraining uses 500 updates from
the local teacher, including the near-origin slope anchors.

The `local` control uses the existing LQR teaching box. The `recovery` variant
adds held-current LQR approaches during SAC updates. It samples new starts
near mechanical capture with a reproducible separate RNG, keeps the
trajectories that reach the small teaching box and sustain balance, and
adds the approach up to the first entry. Both profiles use the same
local data and pretraining. They require no earlier model.

Activate `rl-env` and run from the repository root. The two jobs are
sequential on the same GPU and use seed 0 unless you supply `--seed`.

```bash
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
capture_panel="models/nominal_scratch_actor16_v1/capture_starts.csv"
stamp=$(date +%Y%m%d_%H%M%S)
for variant in local recovery; do
  python -u src/nominal.py --scratch-pilot "$variant" --capture-benchmark "$capture_panel" \
    --job-dir "src/runs/nominal_sac_scratch_${variant}_seed0_$stamp" || break
done
```

Once both preflights pass, use the same loop with `--execute` appended to the
Python command. The fixed 30-state capture panel is optional and used only
after training. Each run also scores 100 final periodic starts and an
81-state upright recovery panel. Checkpoints are ranked every 25,000
decisions on fixed validation starts by sustained balance, then balance
after entering the teaching box, then mean return.

Read `training/validation_metrics.json`, `evaluation/summary.txt`,
`upright_benchmark.json`, and `capture_benchmark.json` in each job directory.
`training/training.json` records the recovery sampling and dataset counts.
Each best checkpoint exports a single neural controller as `training/actor.pt2`.

## One simple nominal SAC run

For a single swing-up-and-balance experiment without a filter, action delay,
plant randomization, curriculum, or LQR, run:

```bash
python src/nominal.py --execute
```

This trains SAC from scratch with seed 0, a `64 -> 64` actor and critics,
`gamma = 0.99`, and two million decisions. The existing reward, five-second
episode limit, and strict one-second balance criterion are unchanged. Initial
states still vary over the existing broad reset distribution; the plant is
fixed. This is more demanding than the linked MathWorks example's near-downward,
zero-velocity reset, and its arm-position success criterion is stricter than
the behavior illustrated there.

Training validation uses 50 fixed nominal episodes. After training, the
validation-selected policy is automatically evaluated with SAC alone on 100
nominal episodes, seeds 80000-80099. No evaluation-based checkpoint selection
is performed. The final report records success, safety, and capture counts;
learning success is not guaranteed by completing the training budget.

Outputs are under `src/runs/nominal_sac_actor64_seed0_v0/`:
`training/` contains checkpoints, full resume states, and TensorBoard events;
`evaluation/` contains summaries, PNG plots, and CSV traces; `status.json`
records the active phase or failure; `FINDINGS.md` is written after evaluation.
Existing output is never overwritten. Omit `--execute` for a dry run, or use
`--job-dir` to choose a fresh output location.

```bash
tensorboard --logdir src/runs/nominal_sac_actor64_seed0_v0/training/tensorboard --port 6007
```

The matching VS Code profile is `Train and evaluate simple nominal SAC (64x64)`.

### Current-jitter experiment

After the successful nominal baseline exists locally, run the isolated
action-change-weight comparison:

```bash
python src/smooth_current.py --execute
```

This starts a fresh seed-0, 2M-decision nominal run with only the action-change
penalty increased from `3.0` to `5.0`. It does not resume the baseline learner
with a changed reward. Filter, delay, and plant randomization stay off;
networks, gamma, balance criteria, and all other reward weights stay unchanged.
The runner checks the baseline configuration, learner settings, and core-code
snapshot before starting. No existing results are overwritten or models promoted.

Training, 100-episode nominal SAC evaluation, and a matched current comparison
run sequentially without prompts. The comparison replays both validation-selected
policies on seeds 80000-80099 and reports mean, p95, and maximum command-change
RMS, total variation, and >=25 Hz power fraction. Windows cover the whole episode,
the first second, and 0.2 seconds before/after first mechanical capture. Capture
windows can differ between policies and exclude cases without capture; failures
must be considered before interpreting smoother commands. These previously
inspected seeds are a diagnostic comparison, not a new untouched holdout.

Outputs are under `src/runs/nominal_sac_actor64_delta5_seed0_v0/`:
`candidate/` holds training and evaluation, `comparison/` holds paired CSV metrics
and PNG traces for seeds 80000 and 80015, and `reference/` preserves the baseline
model and metadata. `status.json` tracks the overall phase, `candidate/status.json`
distinguishes training from evaluation, and `FINDINGS.md` plus `comparison.json`
are written after comparison. An execution failure stops the job and records
the error; it does not silently retry or overwrite partial runs.

Omit `--execute` for preflight only. For individual nominal experiments,
`src/nominal.py` also accepts `--action-change-weight` and `--job-dir`.

### Weight-4 follow-up

With the nominal weight-3 and weight-5 runs available locally:

```bash
python src/weight4.py --execute
```

This trains weight 4 from scratch with the same seed-0, 2M-decision nominal
settings. It automatically runs the usual 100-episode evaluation, then compares
the three validation-selected policies on 100 old diagnostic seeds, a fixed
81-state neighborhood around failure seed 180051, and 100 fresh seeds
280000-280099 for 30 seconds each. Neighborhood offsets are +/-0.035 rad in
each angle and +/-0.25 rad/s in each velocity; combinations outside reset
support are omitted. No plant uncertainty, filtering, or delay is introduced.

The fresh run is under `src/runs/nominal_sac_actor64_delta4_seed0_v0/`.
`candidate/` contains training and nominal evaluation; `reference/` preserves
the compared checkpoints; `source/` and `protocol.json` record the experiment.
`comparison/` holds episode/current CSVs and representative or flagged traces
and plots; `summary.json` and `FINDINGS.md` are written after all comparisons.
Outer `status.json` tracks the full job, while `candidate/status.json` separates
training from nominal evaluation. Failures stop the pipeline and record
`ERROR.txt`; existing output is never overwritten. Omit `--execute` for preflight.

Success at five seconds, final success, unsafe episodes, later balance loss,
settling time, and paired current-window metrics are reported separately.
Unsafe or truncated episodes are not silently treated as smoother successes.
Evaluation seeds never select checkpoints, and no model is automatically promoted.

### No-centering task

```bash
python src/no_centering.py --execute
```

This is a NEW task variant, not a relabeling of earlier results. The arm-angle
reward weight becomes zero, and the balance arm-position tolerance becomes
the existing travel limit (5*pi/8). Upright and velocity tolerances, the
one-second hold, hard safety limits, current range, seven observations, weight-4
action-change penalty, gamma, networks, and seed-0 2M-decision training remain
unchanged. No extra boundary penalty is introduced. Stationary offsets near
travel limits can pass; travel margins must be assessed before deployment.

The three slow saved weight-4 cases were delayed by arm SPEED, not position.
`python src/diagnose_centering.py` reproduces that diagnosis without altering
saved results. Removing centering does not retrospectively make those cases pass.

The runner trains from scratch and evaluates automatically. Compared models
are the original centered weights 3 and 4 and the new no-centering weight 4.
All are scored under both original and new success definitions, with paired
current metrics and minimum travel margins. The comparison covers old100
diagnostic seeds,81 neighborhood cases,100 fresh30-second seeds380000-380099,
and the three slow30-second diagnostic cases. Training validation uses the
new objective; none of these comparison cases select the checkpoint.

Outputs are under `src/runs/nominal_sac_no_centering_delta4_seed0_v0/`:
`candidate/`, `comparison/`, `reference/`, source/protocol snapshots,
`summary.json`, and `FINDINGS.md`. Outer status updates fall back to unique
`status_event_*.json` files if Windows locks `status.json`; `completed.json`
is written only after the final report. `failed.json` records execution errors.
Existing outputs and models are never overwritten. Omit `--execute` for preflight.

## Run the three-stage experiment

### Stage 1: isolate the arm-velocity weight on the nominal plant

```bash
python src/train.py
```

By default, the new run is saved under:

```text
src/runs/sac_200hz_matlab_pure_actor8_critic64_stage1_armv015_v0
```

The default initializes the complete SAC model from scratch: actor, critics,
target critics, optimizers, replay buffer, and entropy state are all new. This
isolates the arm-velocity weight change on the same nominal direct-current
setting as the successful actor-8 experiment. The trainer refuses to use a nonempty
output directory. `final/model.zip` contains the complete final SB3 learner
state and `final/replay_buffer.pkl` contains its matching experience. The
`final/` directory is published only after both files are complete.
`best/best_model.zip` remains the success-first policy-selection checkpoint.
`config.json` and `training.json` record the physical, reward, architecture,
seed, initialization, and final timestep choices.

For the compact nominal-only teaching workflow, train for two million decisions
while retaining every 100,000-step checkpoint:

```bash
python src/train.py \
  --timesteps 2000000 \
  --current-filter-cutoff-hz 0 \
  --evaluation-episodes 50 \
  --seed 0 \
  --run-dir src/runs/nominal_direct_seed0_2m
```

Then evaluate the validation-selected model once on 100 unseen nominal seeds:

```bash
python src/evaluate.py \
  --model src/runs/nominal_direct_seed0_2m/best/best_model.zip \
  --controller sac \
  --mode nominal \
  --episodes 100 \
  --base-seed 20000 \
  --results-dir src/runs/evaluation_nominal_direct_seed0_2m_best
```

Repeat the training run with independent seeds before drawing a final comparison.
Do not use the unseen evaluation set to choose checkpoints.

### Stage 2: add only the hidden current response

Run stage 2 from scratch only after stage 1 passes evaluation:

```bash
python src/train.py \
  --current-filter-cutoff-hz 100 \
  --run-dir src/runs/sac_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0
```

This changes only the RL current path. The plant remains nominal.

### Stage 3c: canonical gradual uncertainty curriculum

The canonical runner restores the complete Stage-2 learner and replay buffer,
ramps all uncertainty from zero to full range over one million decisions, then
uses learning rate `1e-4` for the final 500,000 decisions. It retains complete
resume states at the 2.4M comparison point and the selected 2.6M point:

```bash
python src/reproduce.py --all --seed 0
python src/reproduce.py --all --seed 0 --execute
```

The first command is a dry run that prints all three exact training commands.
The second executes them. The recipe is
`experiments/stage3c_half_rps_selected_v0.json`; it uses a nominal 100 Hz Stage
2, then independent `+/-10%` mechanical uncertainty, 50--100 Hz RL filter
uncertainty, zero/one policy-sample delay, arm reset velocity up to
`+/-0.5 rotations/s`, and pendulum reset velocity up to
`+/-1.5 rotations/s`.

TensorBoard records the curriculum scale and learning rate:

```bash
tensorboard --logdir src/runs/reproduce_stage3c_half_rps_selected_v0_seed0_stage3c/tensorboard
```

### Longer discount-horizon experiment

The separate recipe `experiments/stage3c_half_rps_gamma09975_v0.json` runs the
same three-stage chain from scratch with `--gamma 0.9975`. At 200 Hz, the
approximate discount horizon `policy_period / (1 - gamma)` increases from
0.5 seconds to 2 seconds. The five-second episode limit, reward weights,
`+100` terminal bonus, and uncertainty curriculum remain unchanged. Keeping
the bonus fixed isolates gamma; it is no longer the `1 / (1 - gamma)`
continuation-value approximation for unit reward.

Inspect the new seed-0 chain without starting training:

```bash
python src/reproduce.py --recipe experiments/stage3c_half_rps_gamma09975_v0.json --all --seed 0
```

Add `--execute` after reviewing and committing the changes to launch training.
The recipe uses fresh `reproduce_stage3c_half_rps_gamma09975_v0_seed0_*`
directories and retains the same 2.4M and 2.6M model/replay milestones. These
are comparison checkpoints, not preselected winners for the new experiment.
The frozen release and original recipes are unchanged.

`--gamma` defaults to `0.99` for scratch or actor-transfer training. Full
resume and curriculum continuation inherit the saved learner's gamma when
omitted and reject an explicitly different value. The effective value is
recorded in `training.json`. Compare with a fresh default-gamma run using the
same current code and validation settings, not just the historical release.

### Selected Stage-3c 2.6M controller

Matched checkpoint screening selected absolute timestep 2.6M. It was the only
screened checkpoint with 100/100 successes in nominal, randomized broad, and
randomized downward-rest SAC modes and zero unsafe episodes. Every 2.7M--3.0M
checkpoint regressed on at least one matched seed. Both SAC and hybrid achieved
300/300 selection successes and 300/300 unseen robust-holdout successes, with
zero unsafe episodes. The fixed nominal-plant, 80 Hz, no-delay suite also passed
300/300 for each controller.

The immutable actor, exact configuration, source training metadata, acceptance
gate, reports, and raw summary tables are frozen under
`models/stage3c_half_rps_v0/`. Re-run both qualification protocols with:

```bash
python src/qualify.py --suite all
python src/qualify.py --suite all --execute
```

The runner verifies all release hashes, uses the frozen seed ranges and
configurations, and checks the resulting summaries against the predeclared
requirements. The selected release actor itself has no replay buffer. Exact
continuation uses the complete 2.6M milestone created by the canonical recipe;
never combine this actor with the 3.0M replay buffer.

The older `models/stage3b_v0/` release and its failed holdouts remain preserved
as historical regression evidence. The current release has passed stochastic
simulation qualification but not deterministic stress or hardware validation,
so it is not yet marked hardware-ready.

The four initialization modes are deliberately separate:

| mode | flag | restored state | intended use |
|---|---|---|---|
| scratch | none | nothing | independent experiment |
| actor transfer | `--initial-actor MODEL` | actor weights only | changed task or dynamics |
| full resume | `--resume-model MODEL` | complete SAC learner plus replay | more training on the identical task |
| curriculum | `--curriculum-model MODEL` | complete SAC learner plus replay | gradual transition to changed uncertainty ranges |

For a later actor-transfer experiment, explicitly provide a model and a fresh
run directory:

```bash
python src/train.py \
  --initial-actor src/runs/sac_200hz_matlab_pure_actor8_critic64_v1/best/best_model.zip \
  --run-dir src/runs/my_actor_transfer_experiment
```

This copies only the actor. Critics, optimizers, entropy state, and replay
buffer remain new, so it is transfer initialization rather than an exact SAC
resume. Because this is still a new SB3 learner, its first 40,000 decisions are
uniformly random replay collection; the copied actor is retained but is not
used to choose those warm-up actions. Full resume does not repeat that warm-up
because it retains the timestep counter and replay buffer.

To continue the complete SAC learner on the unchanged task, use its final
model and a fresh output directory:

```bash
python src/train.py \
  --resume-model src/runs/previous_run/final \
  --timesteps 500000 \
  --run-dir src/runs/previous_run_resume1
```

Here `--timesteps` means additional decisions and must be even because SAC
collects two decisions per training rollout. A committed `final/` or rolling
`resume/` directory resolves its matching model and replay buffer
automatically; a direct model path and unusual replay location can still be
supplied explicitly. SB3 restores the actor, critics,
target critics, their optimizer states, automatic-entropy state, and timestep
counter from `model.zip`; the trainer then loads the paired replay buffer. It
requires `config.json` to match the current task exactly. Use actor transfer or
scratch training if reward, dynamics, observation, or randomization settings
change. If the source used nondefault task flags, repeat those flags on the
resume invocation so the validation is explicit. Network and seed flags do not
replace properties of a fully loaded learner.

The committed `final/` state and the rolling state under `resume/` are
resumable. Each rolling save is first completed in a new generation directory;
only then is `resume/latest.json` replaced atomically and the older generation
removed. Passing `--resume-model RUN/resume` therefore resolves only a complete
model/replay pair. Set `--resume-snapshot-frequency 0` to disable rolling
states. Files under `checkpoints/` and `best/` do not have paired replay buffers
and are intended for inspection, evaluation, or actor transfer rather than
full resume. Runs completed before replay-buffer saving was added also cannot
be fully resumed retrospectively; use their selected model for actor transfer.

Pass `--resume-milestone-timesteps` with absolute, even SAC timesteps to retain
additional complete model/replay states permanently. Unlike the rolling state,
these `milestones/state_*` directories are not deleted. This prevents a policy
selected retrospectively from becoming an actor-only checkpoint.

This is a full learned-state continuation, but not a bit-for-bit restart at the
middle of an environment step. A resumed invocation begins with a newly reset
environment and new callback histories; Python/NumPy environment RNG state is
not serialized. The retained timestep count prevents SB3 from repeating the
40,000-decision random-action warm-up.

`--curriculum-model` is the one intentional exception to exact configuration
matching: it permits only the saved uncertainty fields to change. Reward,
nominal dynamics, observation, termination, controller rate, and network
structure must still match the source. An interrupted curriculum is continued
with `--resume-model` and the target uncertainty flags repeated exactly; its
original absolute ramp and learning-rate boundaries are restored from
`training.json` rather than restarted.

### Reproduce the complete ancestry

The selected-policy ancestry is Stage 2 at 100 Hz from scratch to `1.0M`, an
exact Stage-2 resume to `1.5M`, then the Stage-3c uncertainty curriculum to
3.0M. Stage 1 is a comparison baseline, not an ancestor of Stage 3c.

Three tracked recipes describe the history and canonical reconstruction:

- `experiments/reproduce_stage3b_v0.json` reproduces the historical Stage-3b
  settings and retains a paired state at the selected absolute `2.4M` step.
- `experiments/stage3c_half_rps_v0.json` is the immutable executed recipe that
  produced the selected 2.6M actor and the evaluated 3.0M final learner.
- `experiments/stage3c_half_rps_selected_v0.json` is the default canonical
  reconstruction. Its learning settings are identical to the executed recipe,
  but it uses fresh directories and retains complete model/replay states at
  both 2.4M and the selected 2.6M boundary.

Display the full seed-0 chain without training:

```bash
python src/reproduce.py --all --seed 0
```

Execute all stages by adding `--execute`, or execute them individually:

```bash
python src/reproduce.py --stage stage2_scratch_1m --seed 0 --execute
python src/reproduce.py --stage stage2_continue_1p5m --seed 0 --execute
python src/reproduce.py --stage stage3c_curriculum --seed 0 --execute
```

The runner refuses to overwrite or silently adopt an unverified directory.
It also refuses to execute from a dirty Git tree unless `--allow-dirty` is
provided explicitly. Commit the reviewed recipe and implementation before a
canonical reproduction run.
After each stage it hashes the parent and resulting model/replay pair and writes
`pipeline_stage.json`. Training also records the command, Git commit and dirty
flag, Python/Conda environment, core package versions, parent hashes, and final
artifact hashes in `training.json`. `environment.yml` recreates the supported
Conda environment; the per-run provenance captures the exact versions used.

After reproduction, compare the generated 2.6M actor with the frozen release:

```bash
sha256sum \
  src/runs/reproduce_stage3c_half_rps_selected_v0_seed0_stage3c/checkpoints/sac_2600000_steps.zip \
  models/stage3c_half_rps_v0/model.zip
```

The reference seed-0 release hash is
`ce2371e3f57107ae755fe5c6b8317129c5386e8cc96f9bc9f94d6f9282d63fbf`.
Exact byte equality is expected in the recorded environment and commit; other
hardware or library builds may not be bitwise deterministic and must still pass
the frozen qualification protocols.

For independent evidence, repeat the entire recipe with seeds `1` and `2`.
Sharing one Stage-2 source between all seeds tests only curriculum adaptation;
it is not an independent end-to-end reproduction.

### Run from VS Code

Select the `rl-env` Conda interpreter once. Open **Run and Debug** and choose:

- `Train stage 1: arm velocity 0.015 (default)`;
- `Train nominal direct-current 2M (seed 0)` and its matching 100-episode
  unseen nominal evaluation profile;
- `Train stage 2: add 100 Hz filter`;
- `Train stage 3b: refined curriculum seed 0` is retained as a historical
  experiment profile;
- `Train: actor transfer (edit paths)` to copy actor weights;
- `Train: full resume (edit paths)` to restore the complete learner;
- `Evaluate selected Stage3c 2.6M: selection seeds`;
- `Qualify selected Stage3c: show frozen protocols` or `execute all`;
- `Stress selected Stage3c: corners and 30 s`;
- the `Reproduce Stage3c half-rps` profiles to inspect or execute each tracked
  training stage;
- `Continue reproduced selected 2.6M: exact +500k` to resume the retained
  selected learner without changing the task.

For editable transfer or resume profiles, check the source and fresh destination
paths in `.vscode/launch.json` before starting. The launch profiles are simply
the documented command-line flags in a form that does not require typing a terminal
command. **Run Python File** on `src/train.py` still runs the scratch default.

Trace learning with:

```bash
tensorboard --logdir src/runs/sac_200hz_matlab_pure_actor8_critic64_stage1_armv015_v0/tensorboard
```

For a short software check:

```bash
python src/train.py --timesteps 100 \
  --run-dir /tmp/furuta_matlab_pure_smoke
```

Parameters can be changed without editing code. For example:

```bash
python src/train.py \
  --arm-velocity-weight 0.0125 \
  --pendulum-velocity-weight 0.02 \
  --action-change-weight 5 \
  --parameter-randomization \
  --motor-torque-randomization 0.08 \
  --run-dir src/runs/my_next_experiment
```

Run `python src/train.py --help` for reward, actuator, delay, reset,
randomization, network, seed, actor-transfer, curriculum, and full-resume
controls. Always choose a fresh run
directory.

RL results vary by seed. Each stage begins as a seed-0 from-scratch pilot.
A candidate is promoted to a three-seed confirmation only after the standalone
evaluation retains reliability, safety, current smoothness, and improved
pendulum/arm behavior. This follows
[Stable-Baselines3 guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
to use normalized continuous actions and observations, separate deterministic
evaluation, quantitative hyperparameter comparisons, and multiple random
seeds.

## Evaluate

The default evaluator uses the frozen `models/stage3c_half_rps_v0/model.zip`
and its full-scale configuration:

```bash
python src/evaluate.py
```

The combined SAC-only and hybrid results are written to:

```text
src/runs/evaluation_stage3c_half_rps_v0
```

Generated training and evaluation directories under `src/runs/` are ignored
by Git because models, replay buffers, plots, and traces are large generated
artifacts. The small frozen controller under `models/stage3c_half_rps_v0/` is
the intentional canonical exception and contains no replay buffer. Keep full runs
and result archives in dedicated artifact storage (or Git LFS if chosen
explicitly).

Evaluation refuses to replace an existing result set. To rerun it deliberately,
pass `--overwrite`. A single controller can still be requested explicitly:

```bash
python src/evaluate.py --controller sac \
  --results-dir src/runs/evaluation_200hz_matlab_pure_actor8_critic64_stage1_armv015_v0_sac
```

## Experimental rationale and next steps

### Completed `64 -> 64` baseline

The completed seed-0 arm-weight run selected its 800k checkpoint because it
achieved 100% deterministic checkpoint success and settled more slowly than
the also-successful 900k checkpoint. The full standalone evaluation then
achieved 100/100 SAC-only and 100/100 hybrid successes in each of the three
scenarios, with no unsafe episode.

The SAC-only baseline for comparison is:

| scenario | success | current RMS | current-change RMS | HF power | arm excursion | worst arm speed | final arm-error RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| training nominal | 100/100 | 0.0937 A | 0.0252 A | 3.0% | 1.514 rad | 21.69 rad/s | 0.1891 rad |
| training randomized | 100/100 | 0.0909 A | 0.0241 A | 3.7% | 1.451 rad | 20.70 rad/s | 0.1891 rad |
| downward rest randomized | 100/100 | 0.1064 A | 0.0250 A | 1.3% | 1.386 rad | 18.51 rad/s | 0.1890 rad |

These values come from
`src/runs/evaluation_200hz_matlab_pure_arm_weights_v1/summary.csv`. They are the
fixed acceptance baseline for the network-capacity experiment; raw rewards are
secondary to matched physical metrics.

### Completed `8 -> 8` actor-capacity result

The selected actor-8 checkpoint achieved 99/100 SAC-only nominal successes and
100/100 in both randomized scenarios; the hybrid showed the same single
nominal unsafe case. The failure started near the positive arm limit and was
not unique to the selected checkpoint. Relative to the actor-64 baseline, the
smaller actor reduced arm excursion, peak arm speed, and post-capture arm
motion, but increased pendulum-speed RMS, current RMS, current-command changes,
reversals, and total variation. Its final successful SAC arm offset remained
about `0.194 rad`. This is why the actor remains `8 -> 8` while the next run
changes trajectory shaping and robustness rather than network capacity.

### Failed combined robust experiment and staged recovery

The combined seed-0 robust run changed both velocity weights, raised the
action-change weight to `4.0`, enabled the hidden 100 Hz current response, and
enabled parameter randomization simultaneously. It produced zero successful
training episodes and zero SAC-only evaluation successes. Hybrid evaluation
reached `89%` nominal success but only `10%` randomized broad-reset and `6%`
randomized downward-rest success. It remained safe and reduced arm excursion,
but pendulum velocity, current magnitude, and total variation deteriorated.

Because that run changed too many factors at once, the current task version
started a controlled three-stage sequence. A subsequent nominal run changed both
velocity weights to `0.015`; it reached the capture region in 99/100 nominal
episodes but achieved zero pure-SAC successes because it did not settle the
arm. The restarted stage 1 therefore changes only the arm-velocity weight from
`0.020` to `0.015`, restores the successful pendulum-velocity weight of `0.010`,
and keeps the action-change weight at `3.0`.
Stage 2 added only the 100 Hz response. After selecting and reconstructing the
exactly resumable 1.5M Stage-2 controller, Stage 3a added the first agreed
uncertainty envelope through a full-state curriculum. Its evaluation then
motivated the narrower Stage-3b envelope documented above. The current
randomizer varies motor torque, both masses, both rotary inertias, and the two
joint dampings independently; it also varies cutoff, action delay, and reset
velocity bounds. Historical randomized results used an older
coupled parameterization, so their nominal modes remain the cleanest
cross-version comparison; the three new stages use one common schema.

### Completed path to the selected stage-2 controller

The seed-0 staged pilot required checkpoint selection rather than assuming
that the final policy was the best policy. The table reports SAC-only
successes for nominal broad resets, randomized broad resets, and randomized
downward-rest resets, in that order. Each entry contains 100 episodes. The
hybrid controller was evaluated separately with the same seeds.

| experiment artifact | decisions | SAC successes | unsafe | conclusion |
|---|---:|---:|---:|---|
| `sac_200hz_matlab_pure_actor8_critic64_stage1_armv015_v0` | 1.0M | 99 / 85 / 76 | 1 | direct-current stage needed more training |
| stage-1 continuation, final policy | 2.0M | 100 / 98 / 99 | 0 | reliable but not the strongest checkpoint |
| stage-1 continuation, 1.5M checkpoint | 1.5M | 100 / 100 / 100 | 0 | selected direct-current reference; hybrid also 300/300 |
| `sac_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0` selected 600k policy | 1.0M run | 93 / 81 / 62 | 4 | adding only the hidden 100 Hz response required more training |
| stage-2 continuation, final 2.0M policy | 2.0M | 99 / 100 / 100 | 1 | later training recovered reliability but introduced one unsafe reset |
| stage-2 continuation, selected 2.0M policy | 2.0M | 99 / 100 / 100 | 1 | reproduced the same unsafe seed and approximately `0.218 rad` final arm offset |
| stage-2 continuation, 1.5M checkpoint | 1.5M | 100 / 100 / 100 | 0 | selected filtered controller; hybrid also 300/300 |
| Stage-3a curriculum final | 3.5M total | 100 / 76 / 81 | 9 | nominal retained; 10 ms delay and extreme independent resets were too demanding |

The selected filtered policy is
`sac_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0_continued_to2m/checkpoints/sac_1500000_steps.zip`.
Its complete evaluation is stored under
`evaluation_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0_continued_1500000`.
It achieved 600/600 successes across SAC-only and hybrid evaluation with no
unsafe episode. Compared with the 2.0M policy, its nominal SAC arm excursion
fell from `1.599` to `1.278 rad`, post-capture arm excursion from `1.013` to
`0.538 rad`, and final arm error from `0.218` to `0.051 rad`.

This checkpoint was originally saved without a replay buffer. To create an
exactly resumable curriculum source, the original stage-2 1.0M final state was
resumed for 500,000 decisions into
`sac_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0_selected_to1500k`.
The committed rolling source is:

```text
src/runs/sac_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0_selected_to1500k/resume/snapshot_000001500000/
```

It contains the complete SAC learner and its one-million-transition replay
buffer. Its serialized `policy.pth` SHA-256 is
`7ea043548107ba9a075e46a85e497b0508e9e17363f093ecc9299cfb0508551c`,
identical to the fully evaluated historical 1.5M checkpoint. Use this rolling
snapshot, rather than the run's post-callback `final/model.zip`, as the
curriculum source.

This historical selected Stage-2 policy was a curriculum starting point, not a
hardware-ready controller. Relative to the direct-current stage-1 reference it
improves arm
excursion, arm-speed RMS, post-capture arm motion, and final arm speed, but has
higher current RMS, requested-current changes and total variation, and higher
pendulum-speed RMS. The 100 Hz response smooths applied current; it must not be
used to hide those requested-current differences.

Acceptance requires at least 100/100 successes and zero unsafe episodes in all
three SAC-only modes before considering hardware. It must also improve
pendulum-speed RMS and requested-current roughness without giving back the
actor-8 arm excursion and arm-speed gains. Applied-current smoothness is
reported separately; the 100 Hz response must not be used to excuse a rough
requested-current policy.

The MathWorks hardware example uses a 5 ms policy sample, a normalized action,
a `64`-unit initialization, replay capacity `1e6`, batch size `256`, entropy
weight `0.1`, target entropy `-1`, and an action-change term. Stable-Baselines3
uses the same normalized-action convention and supports the current SAC
settings, but explicitly warns that custom environments require quantitative
hyperparameter testing and multiple seeds.

Stage 3a used `60--140 Hz`, `0--2` action-delay samples, and reset velocities
up to `+/-2 rotations/s`. Its final policy retained 100/100 nominal success,
but achieved only 76/100 randomized broad-reset and 81/100 randomized
downward-rest success, with nine unsafe broad resets. Stratifying identical
evaluation episodes isolated two issues: every non-unsafe 0 or 5 ms broad-reset
episode succeeded, while only 11/26 safe 10 ms episodes succeeded; all nine
unsafe episodes began above `8 rad/s`, often near a travel limit and moving
outward. Stage 3b therefore restarted from the same selected Stage-2 source
with `50--100 Hz`, `0--1` delay samples, and `+/-1.5 rotations/s`; Stage 3c
subsequently reduced only the randomized arm-velocity envelope to
`+/-0.5 rotations/s` and was reproduced end to end.

The selected Stage-3c 2.6M controller passed matched selection and untouched
holdout evaluation for both SAC and hybrid, including the fixed nominal 80 Hz,
no-delay suite. Adaptation-only seeds would share the pretraining history and
would therefore not prove the complete chain. Independent end-to-end seeds 1
and 2 remain the appropriate statistical confirmation before making a broad
training-repeatability claim. Deterministic stress and hardware validation also
remain required before deployment.

For sim-to-real work, prioritize measured system identification over widening
randomization blindly. Estimate the current-loop response, torque constant,
inertias, damping/friction, encoder-derived velocity noise, computation delay,
and sample-time jitter, then center and size the distributions from those
measurements. Randomizing delay, sensor noise/filtering, Coulomb friction,
dead zone, and current gain is more defensible once their hardware ranges are
known. Excessively broad randomization can produce a needlessly conservative
policy.

The hidden actuator state deliberately makes the policy input slightly
partially observable. The previous command helps with the refined one-sample
delay, while Stage 3a showed that an unknown two-sample delay was not learned
reliably with only one previous action in the observation. Measured longer
delay or a slower current loop would justify longer observation/action history
or a recurrent policy. Policy-level
temporal and spatial regularization such as
[Conditioning for Action Policy Smoothness (CAPS)](https://arxiv.org/abs/2012.06644)
remains a later experiment if the action-change penalty is insufficient. A
deployment current limiter remains a separate safety layer.

Before interpreting simulation robustness as hardware readiness, validate the
plant against MATLAB or measured trajectories, including motor sign and torque
constant, inertias, damping, current-loop dynamics, encoder filtering, delay,
and the 10 kHz current-control timing. Parameter randomization does not replace
that validation.

## Before real hardware

This project intentionally has no learned-current command limiter. Simulation
success is not a hardware safety certificate. Before deployment, validate
encoder signs, current and thermal limits, current-loop behavior, velocity
estimation, sampling delay, travel protection, watchdogs, and emergency-stop
behavior on a mechanically safe setup.
