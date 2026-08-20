# Furuta swing-up: direct-current SAC

This is a deliberately small teaching project. SAC learns the complete
swing-up-and-balance task. The retained LQR is not used during training; it is
only an evaluation and deployment comparison. The project follows the
observation and action-change
structure of the MathWorks
[Quanser QUBE example](https://www.mathworks.com/help/reinforcement-learning/ug/train-agent-offline-to-control-quanser-qube-pendulum.html),
with these intentional differences:

- the existing nonlinear Furuta model and its measured parameters;
- SAC instead of TD3;
- motor current in `[-0.5, +0.5] A` instead of motor voltage;
- optional comparison against a deterministic LQR after SAC swing-up.

The first and most recent successful experiments remain fixed comparison
references in the experiment review. The executable code now supports only
the current pure-RL task schema; historical compatibility branches have
deliberately been removed.

## Versioning

The cleaned, executable experiment schema starts at `v0`. The version is
stored in every run's `config.json` as
`task_version = "matlab_hardware_sac_pure_v0"` and is repeated at the end of
the canonical run-directory names. Stages 1, 2, and 3 are controlled
experiments within the same `v0` schema; the stage number is not a schema
version.

Increment the suffix to `v1`, `v2`, and so on only for a breaking structural
change—for example, changing the observation or action shape, reward formula,
termination logic, plant equations, or the meaning of saved randomization
fields. Additive uncertainty fields with nominal defaults remain compatible
with the selected `v0` source. Numeric
hyperparameters such as reward weights, seeds, training duration, and stage
options are recorded in `config.json` and the run-directory name; they do not
by themselves advance the schema version. Historical result names that already
contain another suffix are fixed artifact labels and are not part of this new
numbering sequence.

## Controller structure

```text
stage 1: encoder state -> SAC at 200 Hz -> held current -> nominal plant
stage 2: encoder state -> SAC -> hidden continuous 100 Hz response -> nominal plant
stage 3b: encoder state -> SAC -> hidden 0--1 sample delay
                               -> hidden 50--100 Hz response -> randomized plant

evaluation: SAC-only full episode
            SAC -> sustained compatible capture -> fast-feedback LQR
```

The mechanical simulation uses a 1 kHz RK4 grid. Each SAC current command is
held for five plant steps. Stage 1 applies it directly. Stages 2 and 3 use
`di/dt = 2*pi*f_c*(i_command - i)`: Stage 2 fixes `f_c = 100 Hz`, while
Stage 3b samples it from the active curriculum interval. This state is
integrated jointly with the mechanics at every RK4 stage but is not added to
the SAC observation. It represents unobserved actuator uncertainty, not a
deployment slew limiter. Hardware LQR current control is intended to run at 10 kHz, so
evaluation recomputes LQR feedback directly at the RK4 stages and always
bypasses both uncertainty models. In hybrid evaluation, delay and filtering
are active only during the SAC portion and are removed at LQR handover. A
delay sample is one 200 Hz policy period (`5 ms`), not one 1 kHz integration
step. The delay and actuator state are deliberately absent from the SAC
observation.

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
    + action^2
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

Stages 1 and 2 train on the nominal plant. Stage 3b is a continuation curriculum
whose uncertainty fraction ramps linearly for one million decisions and then
stays at full range for another million. At full range, motor torque, both
masses, both rotary inertias, and both damping coefficients vary independently
by `+/-10%`; the RL current-filter cutoff varies uniformly from `50` to
`100 Hz`; and the hidden RL action delay is sampled from `0` or `1`
policy period. The arm and pendulum reset-velocity half-ranges widen from the
nominal `2 rad/s` to `1.5 rotations/s` (`3*pi rad/s`). The final 500,000
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
equal success rates it deliberately prefers later termination, then mean reward.
This encodes the preference for slower settling but does not claim that duration
alone measures smoothness. Faster termination is reported but is not an
objective. Final deployment selection must use the separate two-controller
evaluation, including current smoothness and arm-angle/velocity trajectory
shape.

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
src/stress_evaluate.py deterministic corner and long-duration evaluation
src/test.py          focused dynamics, task, handover, LQR, and smoke tests
models/stage3b_v0/   frozen selected simulation controller and manifest
.vscode/launch.json  one-click scratch, transfer, resume, and evaluation profiles
```

## Install and test

Use Python 3.10 or newer.

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

### Stage 2: add only the hidden current response

Run stage 2 from scratch only after stage 1 passes evaluation:

```bash
python src/train.py \
  --current-filter-cutoff-hz 100 \
  --run-dir src/runs/sac_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0
```

This changes only the RL current path. The plant remains nominal.

### Stage 3b: refined gradual uncertainty curriculum

Stage 3b restores the complete selected Stage-2 learner and replay buffer. It
ramps the complete uncertainty set from zero to full range during the first
million additional decisions, holds full range for the second million, and
uses the reduced learning rate only during the last 500,000 decisions:

```bash
python src/train.py \
  --curriculum-model src/runs/sac_200hz_matlab_pure_actor8_critic64_stage2_filter100_v0_selected_to1500k/resume \
  --timesteps 2000000 \
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
  --randomized-reset-omega1-half-range-rps 1.5 \
  --randomized-reset-omega2-half-range-rps 1.5 \
  --curriculum-ramp-steps 1000000 \
  --curriculum-final-learning-rate 0.0001 \
  --curriculum-final-learning-rate-steps 500000 \
  --seed 0 \
  --run-dir src/runs/sac_200hz_matlab_pure_actor8_critic64_stage3b_curriculum_filter50to100_delay1_vel15rps_seed0_v0
```

At curriculum scale zero, the source controller sees its original nominal
Stage-2 task: fixed `100 Hz` response, zero delay, nominal mechanics, and
`+/-2 rad/s` reset velocities. At scale one it sees the full ranges above.
Mechanical scales, cutoff, delay, and initial state are independently sampled
for every episode. SAC's actor, critics, target critics, optimizers, entropy
state, replay, and timestep count all continue from the selected source.
This refined run starts from the Stage-2 source again, not from the completed
Stage-3a final learner. Relative to Stage 3a, only the cutoff interval, maximum
delay, and two reset-velocity bounds change; reward, network, seed, mechanical
ranges, ramp, hold, and learning-rate schedule remain fixed.
TensorBoard records `curriculum/randomization_scale` and
`curriculum/learning_rate`. Start it with:

```bash
tensorboard --logdir src/runs/sac_200hz_matlab_pure_actor8_critic64_stage3b_curriculum_filter50to100_delay1_vel15rps_seed0_v0/tensorboard
```

After completion, evaluate the full-range final learner first:

```bash
python src/evaluate.py \
  --model src/runs/sac_200hz_matlab_pure_actor8_critic64_stage3b_curriculum_filter50to100_delay1_vel15rps_seed0_v0/final/model.zip \
  --results-dir src/runs/evaluation_200hz_matlab_pure_actor8_critic64_stage3b_curriculum_filter50to100_delay1_vel15rps_seed0_v0_final
```

The nominal best-model callback cannot by itself select the most robust
curriculum checkpoint. If the final learner regresses, evaluate the saved
1.6M/1.7M/... absolute-timestep curriculum checkpoints with separate result
directories before choosing a controller.

### Frozen Stage-3b controller and finalization

The final 3.5M learner regressed, so checkpoint screening selected the absolute
2.4M policy: 900,000 Stage-3b decisions after the Stage-2 source. It had reached
curriculum scale `0.9`, corresponding to `+/-9%` mechanical scales, a
`55--100 Hz` filter interval, reset velocities of approximately
`+/-1.38 rotations/s`, and a one-sample delay in approximately 45% of training
episodes. Standalone evaluation deliberately used the complete configured
`+/-10%`, `50--100 Hz`, `+/-1.5 rotations/s`, and `0--1` sample envelope.

The selected policy achieved 300/300 SAC-only and 300/300 hybrid successes with
no unsafe episode on selection seeds `10000--10299`. It is frozen under
`models/stage3b_v0/`; the manifest records its source, hashes, training scale,
and evaluation evidence. It has no replay buffer and is an evaluation/deployment
candidate, not an exactly resumable learner. No further SAC training is part of
the fast finalization path.

The selection seeds are no longer an unbiased final test. Run the untouched
holdout once, SAC first:

```bash
python src/evaluate.py \
  --controller sac \
  --episodes 100 \
  --base-seed 50000 \
  --results-dir src/runs/evaluation_stage3b_v0_holdout_sac
```

If that passes all three modes with zero unsafe episodes, run the hybrid on the
identical states:

```bash
python src/evaluate.py \
  --controller hybrid \
  --episodes 100 \
  --base-seed 50000 \
  --results-dir src/runs/evaluation_stage3b_v0_holdout_hybrid
```

Then run deterministic endpoint and 30-second drift tests:

```bash
python src/stress_evaluate.py \
  --profile both \
  --controller both \
  --long-duration 30 \
  --results-dir src/runs/stress_stage3b_v0
```

The first untouched SAC holdout has now been run. It achieved `100/100`
nominal, `99/100` randomized broad, and `100/100` randomized downward-rest
success. Broad seed `50116` was unsafe and therefore failed the predeclared
hard gate. It began at arm angle `-1.203 rad` with outward arm velocity
`-8.181 rad/s`, one-sample delay, and an `83.4 Hz` cutoff. The arm crossed its
negative limit after `0.162 s`; hybrid reproduces the same failure before
handover. The full hybrid holdout and stress suite were intentionally not run
after the hard gate failed. This seed is now a permanent regression case and
must not be removed from future evaluation.

### Restricted half-revolution/s operational envelope

The first hardware-facing qualification restricts randomized initial arm speed
to `+/-0.5 rotations/s` (`+/-pi rad/s`). This is an evaluation and engagement
envelope; it does not rewrite the frozen Stage-3b model or erase the failed
broad holdout. Pendulum reset velocity remains `+/-1.5 rotations/s`, and the
mechanical, filter, and delay ranges remain unchanged.

The second protocol was frozen before execution in
`models/stage3b_v0/acceptance_half_rps_v1.json`. Its unseen SAC command is:

```bash
python src/evaluate.py \
  --controller sac \
  --episodes 100 \
  --base-seed 60000 \
  --randomized-arm-velocity-half-range-rps 0.5 \
  --results-dir src/runs/evaluation_stage3b_v0_half_rps_v1_sac
```

If SAC passes all predeclared gates, run the identical seed ranges with
`--controller hybrid` and a new results directory. The deployment engagement
guard must still combine arm position and outward velocity: a speed limit alone
does not guarantee adequate stopping distance near the travel boundary.

This restricted SAC holdout has now been executed. Randomized half-rps and
randomized downward-rest modes both passed `100/100` with no unsafe episode,
but nominal seed `60066` was unsafe, giving `99/100` nominal. It started near
upright at arm angle `+0.985 rad`, arm velocity `+1.906 rad/s`
(`0.303 rotations/s`), and pendulum velocity `-1.981 rad/s` on the nominal
plant. SAC drove outward and crossed the positive arm limit after `0.276 s`.
Hybrid reproduces the failure before handover. The full hybrid holdout and
stress suite were therefore not run. The result is frozen in
`models/stage3b_v0/holdout_half_rps_v1_report.json`.

Consequently, `+/-0.5 rotations/s` is a useful reset bound but not a sufficient
hardware engagement rule. A valid rule must include arm position and outward
velocity and should distinguish the intended downward-rest hardware start from
arbitrary near-upright states.

The corner profile combines both adverse `+/-10%` parameter endpoints with
exactly `50/100 Hz`, exactly zero/one delay sample, downward starts, near-reset
arm-angle starts at nominal outward velocity, and maximum reset velocities at
the arm center. It intentionally does not combine a near-travel-limit arm with
maximum outward speed, because that may have insufficient physical stopping
distance and should not be mislabeled as an ordinary controller failure.

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

The actual selected-policy ancestry is Stage 2 at 100 Hz from scratch to
`1.0M`, an exact Stage-2 resume to `1.5M`, then the Stage-3b uncertainty
curriculum. Stage 1 is a comparison baseline, not an ancestor of Stage 3b.

Two immutable recipes now describe the training chain:

- `experiments/reproduce_stage3b_v0.json` reproduces the historical Stage-3b
  settings and retains a paired state at the selected absolute `2.4M` step.
- `experiments/stage3c_half_rps_v0.json` is the proposed definitive chain. It
  changes only the Stage-3 randomized initial arm velocity to
  `+/-0.5 rotations/s`, retains pendulum velocity at `+/-1.5 rotations/s`,
  reaches the full uncertainty envelope after `1.0M` curriculum decisions,
  then refines for `0.5M` decisions at learning rate `1e-4`.

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

For independent evidence, repeat the entire recipe with seeds `1` and `2`.
Sharing one Stage-2 source between all seeds tests only curriculum adaptation;
it is not an independent end-to-end reproduction.

### Run from VS Code

Select the `rl-env` Conda interpreter once. Open **Run and Debug** and choose:

- `Train stage 1: arm velocity 0.015 (default)`;
- `Train stage 2: add 100 Hz filter`;
- `Train stage 3b: refined curriculum seed 0`;
- `Train: actor transfer (edit paths)` to copy actor weights;
- `Train: full resume (edit paths)` to restore the complete learner;
- `Evaluate selected stage3b_v0: SAC and hybrid`;
- the two `Finalize stage3b_v0` holdout profiles, SAC first;
- `Finalize stage3b_v0: corner and 30 s stress`.
- `Qualify stage3b_v0: half-rps holdout SAC` for the predeclared restricted
  envelope;
- the `Reproduce Stage3c half-rps` profiles to inspect or execute each tracked
  training stage;
- `Continue completed Stage3c half-rps: exact +500k` to resume its complete
  final learner without changing the task.

For the two continuation profiles, edit the source and fresh destination paths
in `.vscode/launch.json` before starting. The launch profiles are simply the
documented command-line flags in a form that does not require typing a terminal
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

The default evaluator now uses the frozen `models/stage3b_v0/model.zip` and its
full-scale configuration:

```bash
python src/evaluate.py
```

The combined SAC-only and hybrid results are written to:

```text
src/runs/evaluation_stage3b_v0
```

Generated training and evaluation directories under `src/runs/` are ignored
by Git because models, replay buffers, plots, and traces are large generated
artifacts. The small frozen controller under `models/stage3b_v0/` is the
intentional canonical exception and contains no replay buffer. Keep full runs
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

The selected policy is a curriculum starting point, not yet a hardware-ready
controller. Relative to the direct-current stage-1 reference it improves arm
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
outward. Stage 3b therefore restarts from the same selected Stage-2 source with
`50--100 Hz`, `0--1` delay samples, and `+/-1.5 rotations/s`. It does not resume
the rougher Stage-3a final policy.

The selected Stage-3b seed-0 controller passed the complete selection
evaluation and is being finalized through an untouched holdout and deterministic
stress suite. Adaptation seeds 1 and 2 remain useful research confirmation, but
are deferred because they do not improve the already selected controller.
Those branches would share the pretraining history and therefore test
curriculum-adaptation repeatability, not the full training pipeline. A
hardware-facing statistical claim still requires independent end-to-end seeds
or equivalent evidence in addition to holdout simulation.

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
