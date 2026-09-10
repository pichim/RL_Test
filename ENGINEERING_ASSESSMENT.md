# Engineering Assessment: Furuta SAC

Historical assessment. For the current state as of 2026-09-10, read
[HANDOFF.md](HANDOFF.md). The replacement current1.5/delta3 and subsequent
current1.5/delta4 runs have completed; statements below about pending training
or absent checkpoints describe the earlier interrupted attempts only.
The original current1/delta3 actor remains the recommended development reference.

Assessment date: 2026-09-09. This is an evidence review, not a release approval.
Existing runs, protocols, model files, dashboards and numerical environments
were not modified. No additional training was launched for this assessment.

## Subsequent Operational Update

Later on 2026-09-09, Windows memory-exhaustion event 2004 and the TensorBoard
MemoryError provided evidence of system-wide commit exhaustion; the origin of
the memory growth remains unresolved. Retry2 reached 176156 recorded steps
and saved a valid 100k model/replay snapshot, superseding the earlier
no-checkpoint observation below. See [MEMORY_INCIDENT.md](MEMORY_INCIDENT.md).

After reboot and checkpoint validation, the user chose to delete all three
interrupted current1.5 attempts and their supervision logs rather than resume.
That cleanup is complete. Historical artifact references below describe the
original assessment; the deleted outputs are no longer evidence available on
disk, and a new run may reuse the original directory name. Completed reference
runs and numerical environments remain intact. No new training was launched.

The replacement workflow is user-operated, with separate training, TensorBoard
and evaluation tasks documented in [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md).

## Executive Decision

Retain the centered delta-current-weight-3 actor as the reference. Keep standard
SAC and the quadratic reward while the current-magnitude-1.5 experiment is
unresolved. Do not increase algorithm complexity to explain one training seed.

The immediate problem is operational: the current experiment is NOT running.
Its status says running/training, but PID22412 is absent and a full Python
process listing contains dashboards and editor services, not a replacement
trainer. The last logged progress was 49,652 decisions (about 2.5% of 2M).
The error log is empty and neither completed.json nor failed.json exists.
The checkpoint directory is empty: no saved model/replay pair is available to
resume. The cause of exit is unknown. The Windows application-crash entries
returned by the audit predate the launch and are not evidence for this exit.
A stale status file is not proof of liveness.

## Ranked Findings

### High: Interrupted experiment and incomplete process supervision

Evidence at assessment time: the original attempt's status and stdout
(subsequently deleted by request), and [runner](src/current_penalty.py).

The runner catches Python exceptions, but that cannot reliably report process
termination, OS kill, or native crashes. Training status is phase-based rather
than a heartbeat. The candidate has no completed evaluation; its early training
success rate is not a performance finding. Preserve this attempt, establish
whether restart is authorized, and use a fresh attempt directory rather than
overwrite or mislabel it as a continuation. Capture the child exit code with an
external supervisor on the next launch; even that cannot survive a host shutdown
without a durable scheduled service. Never infer completion from missing errors.

### High: Nominal success is not hardware qualification

Evidence: [nominal configuration](src/nominal.py),
[plant model](src/furuta_model.py), and
[protocol definition](src/current_penalty.py).

The current protocol uses direct current, exact simulated state and nominal
mechanics. It does not establish tolerance to encoder quantization, velocity
estimation, latency/jitter, current-loop bandwidth, friction mismatch or external
disturbances. Simulator termination after a boundary violation is a measurement,
not a physical safety interlock. Before hardware use, specify measured actuator
limits, real-time deadlines, observation processing, startup action, independent
current/travel protection, and an emergency shutdown. Preserve existing limits.
Do not claim hardware readiness from this study or transfer historical robust
release results to these different actors.

### Medium: A passing pole screen need not describe the reached equilibrium

Evidence: [automatic review](src/current_penalty.py#L191) and
[equilibrium analysis](src/analyze_damping.py#L80).

The gate accepts existence of a stable centered equilibrium with sufficient
slow-mode damping and decay. It does not link every evaluated terminal state to
that root, enforce a minimum fast-pair damping, or score the nonlinear probe
traces. The sign-change root scan is explicitly non-exhaustive; roots exactly on
grid nodes or tangent roots can be missed. A ReLU local Jacobian is not a region
of attraction certificate. Keep the frozen gate as an exploratory screen;
supplement later review with reached-root association, both signs of angle AND
velocity perturbations, nearby Jacobians, saturation checks and actual ringdown.
Do not reinterpret a pass as guaranteed LQR-like dynamics.

### Medium: One training seed cannot establish a reward-weight advantage

Evidence: [paired weight comparison](src/runs/nominal_sac_actor64_delta4_seed0_v0/FINDINGS.md),
[training settings](src/runs/nominal_sac_actor64_seed0_v0/training/training.json),
and the source review below.

Hundreds of reset seeds probe a frozen policy, not variation of the learning
procedure. Reused boundary neighborhoods are diagnostics, not independent random
trials. A perfect 100/100 random test result still permits about a 2.95% failure
probability at a one-sided 95% binomial upper bound, assuming independent draws
from that same distribution. It says nothing about untested hardware conditions.
Report paired episode differences and uncertainty, then repeat training seeds.
Do not select a checkpoint or a new reward using a holdout and still call that
holdout untouched.

### Medium: The training objective differs from persistent balancing

Evidence: [environment step](src/furuta_env.py#L699).

Training ends after a one-second successful hold and awards +100; standalone
evaluation runs the full horizon. That is a defensible episodic task, but does
not directly train long-duration recovery after success. The success-hold counter
is absent from the observation, so the terminal/reward process is not strictly
Markov in the seven observations even for the nominal plant. This is a design
limitation, not evidence that it caused the observed poles. With gamma=.99 at
200Hz, a one-second-ahead contribution is discounted by .99^200, about .134.
Do not change discount, termination and reward simultaneously. First inspect
post-capture coverage and failures; a continuing-balance ablation is a separate
future task change, not part of the current one-weight experiment.

A direct audit probe reproduced identical current/next observations and actions
with terminal flags False versus True and a reward difference of 100 by changing
only the pre-existing success-hold counter. This confirms the partial-observation
issue; it does not quantify its impact on trained-controller performance.

### Medium: Reproduction needs a clean numerical environment

Evidence: [local study findings](src/runs/lqr_weight_study_v1/FINDINGS.md).

The shared SciPy stack previously crashed on native linear algebra. The isolated
study environment works, but inherits other packages from the shared environment;
it is not a fully locked, standalone reproduction. Leave live DLL-using processes
alone. After this experiment, validate a clean environment with pinned numerical
packages, record exact versions and repeat representative inference/plant/solver
checks. Do not upgrade packages during an active run. Old strict experiment
preflights may intentionally reject changed source/config dictionaries; use their
snapshots rather than silently bypassing checks.

## Evidence and Uncertainty

| Status | Observation | Meaning / required check |
|---|---|---|
| Verified saved results | Weight3: 100/100 diagnostic, 81/81 neighborhood, 100/100 fresh30s; no unsafe events in that comparison | Strong reference on these cases, not a global guarantee |
| Verified saved results | Weight4: 97/100 fresh5s, 100/100 fresh30s | Three slow captures; no safety improvement claim from smoothness alone |
| Verified saved results | Weight5: 73/81 neighborhood, eight unsafe | Do not promote on lower current variation |
| Verified prior linearization | Weight3 slow damping .815; held200Hz LQR slow damping .824 | Similar slow damping; fast modes and memory differ |
| Verified local surrogate | All38 LQR designs stable; current coefficient2 gives best tested four-pole distance | Guides an experiment, not SAC pole placement; fifth pole excluded from distance but not stability |
| Unresolved experiment | Current1.5/delta3 run stopped before first100k validation | No result or candidate improvement claim available |
| Hypothesis | More expensive current may reduce aggressive learned corrections | Test with fixed training budget and paired evaluations |
| Unknown | Training-seed variance and hardware robustness | Require separate replication and deployment qualification |

## Physical and Implementation Audit

- Coordinates: upright error is wrapped theta2-pi; both velocities are in rad/s.
  Observations re-encode sin/cos and divide velocities by their30rad/s limits.
  Normalized action maps to current as i=.5a; the previous normalized command
  supplies memory for the change penalty.
- Cost: endpoint state cost plus current and current-change terms. Coefficients
  1 and3 on normalized action correspond to4 and12 on current in amperes,
  before the common .1 scale. Current1.5 corresponds to6, not1.5, in ampere units.
- Timing: five1ms RK4 steps per normal SAC decision. Safety/success may end the
  final decision early. Timeout is returned separately as truncated.
- The plant is coupled and underactuated. Angle feedback modifies effective
  restoring action; velocity feedback modifies effective damping, but reward
  weights do not independently control individual poles.
- Local LQR study includes post-action state cost cross terms, previous-action
  state and discounting. Its ZOH/RK4 Jacobian crosscheck error was3.54e-10.
- Existing LQR evaluates feedback at RK4 stages. For SAC comparison use the
  held200Hz target pairs -4.7026+/-3.2284j and -24.9334+/-17.3616j.
  A fixed arm reference yields isolated equilibria; arbitrary resting position
  as an equilibrium continuum introduces a neutral position mode.
- Mean/maximum finish time is conditional on successful cases. Always display
  failure counts beside it. Current windows pair only complete trajectories;
  excluding failed windows does not remove their reliability failures.
- Current-change RMS, total variation and absolute high-frequency RMS answer
  different questions. Lowpass filtering may add harmful phase lag; do not
  infer actuator feasibility from low spectral fraction alone.

## One Recommended Next Experiment

Complete the already specified current1.5 versus current1 comparison before any
new weight search. Because the launched attempt stopped, recovery is conditional
on authorization and available checkpoint evidence; preserve its artifacts.

| Item | Specification |
|---|---|
| Hypothesis | Current coefficient1.5 reduces current variation without losing weight3 reliability or comparable damping |
| Changed variable | current_weight1 -> 1.5 only |
| Fixed controls | delta weight3; seed0; fresh2M decisions;64x64 actor/critics; gamma.99; centered task; same resets/limits and validation selection |
| Evaluation | Existing284 paired cases per model, including100 fresh30s cases; standard100 diagnostics; local poles and small-signal traces |
| Existing acceptance screens | No case failure/unsafe/late loss; complete100 fresh paired first5s windows; mean deltaRMS/TV/HF RMS no worse; mean successful finish <=110% baseline; centered stable root, slow decay>=3.484/s, slow oscillatory damping>=.7 |
| Supplemental interpretation | Examine all roots, reached equilibrium, fast modes, tail current, travel margins, long-window current and probe trajectories; no automatic promotion |
| Cost | Approximately3-4h training plus evaluation, based on earlier jobs; not an ETA guarantee |
| Stopping conditions | Abort/report numerical faults or missing artifacts; do not change weights mid-run, resume without required state, or hide any failed evaluation |

If the candidate passes, the next proposed training budget is two additional
matched seeds for BOTH settings, not current coefficient2: four runs, roughly
12-16 training hours plus evaluations, requiring separate approval. Three seeds
per setting remain a pilot, not a precise reliability estimate. Freeze a new
confirmation suite before comparing these policies. If it fails, diagnose the
failure before choosing a different reward or training distribution.

## Research Appraisal

Targeted source review accessed2026-09-09; not an exhaustive literature search
or a claim that these are the newest available methods.

| Source | Quality and relevance | Currency and limits | Decision |
|---|---|---|---|
| [SB3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | Maintainer guidance on evaluation, multiple runs, normalized actions and termination | Living documentation; inspect installed implementation for exact behavior | Keep standard SAC; distinguish validation from testing |
| [Agarwal et al., statistical evaluation](https://arxiv.org/abs/2108.13264) | NeurIPS2021 primary research on uncertainty with few RL runs | Revised2022; principles remain relevant, benchmark evidence is not a Furuta guarantee | Report uncertainty and training-seed variation |
| [Tedrake, LQR](https://underactuated.mit.edu/lqr.html) | Authoritative teaching derivation of nonlinear local approximation and discrete/discounted LQR | Living notes; assumes a model and local/unconstrained conditions | Use physical analysis and matched sampling, not pole-distance-only optimization |
| [Mysore et al., CAPS](https://ai.bu.edu/caps/) | ICRA2021 primary method with simulated and hardware smoothness evidence | Transfer to this plant untested; no pole or safety guarantee | Advanced fallback after measuring a persistent smoothness problem |

The CAPS page's informal claim that the Markov property assumes independence of
state-action pairs is not adopted: the Markov property is conditional sufficiency
of the state for predicting transitions, not independence of successive samples.
Do not import the paper's task-specific percentage gains into this project.

## Simplicity and Reward Decision

Do not add a more complex reward now. A quadratic cost already has a sensible
local interpretation, and the surrogate optimum differs substantially from the
learned SAC poles. More terms would not identify the source of that gap.
If failures later isolate a swing-up/settling conflict, consider one smooth
state-dependent penalty or one reset-distribution change as its own experiment.
CAPS, gain matching and residual LQR are advanced extensions, not the teaching
baseline. Keep the student path: understand mechanics, train standard SAC,
evaluate honestly, and explain why a change did or did not help.

## Verification Scope

Earlier experiment preparation passed90 repository tests in the isolated stack,
eight final focused tests in the training environment, an actual28-update SAC
smoke run, and the pole-analysis subprocess smoke test. These checks do not
explain or rule out the subsequent process exit. This assessment changes only
documentation. Additional verification performed for this audit:

- 48 existing core tests passed in 5.755s using the isolated environment,
  including temporary short-learning tests; no new experimental training run.
- 100-state nominal energy-power balance check passed: maximum absolute residual
  9.65e-11 W for dE/dt = km*i*omega1 - b1*omega1^2 - b2*omega2^2.
  Minimum sampled mass-matrix eigenvalue was 7.91e-5, strictly positive. This
  verifies internal mechanical consistency, not identification against hardware.
- All saved source snapshot hashes match the protocol; saved training configuration
  equals the protocol and retains current1.5/delta3, seed0, gamma.99, 2M steps.
- Success-counter probe confirmed the hidden terminal/reward state described above.
- One-sided 95% binomial upper bound for zero failures in 100 independent trials
  computed as 1 - .05^(1/100) = .029513, under its stated sampling assumptions.

No new behavior fix, reward change, model selection, process restart or package
installation was made. Restart needs confirmation because the trainer's exit
may have been deliberate and there is no resumable checkpoint.
