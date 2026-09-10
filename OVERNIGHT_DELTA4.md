# Current1.5 / Delta4 Overnight Run

## Completed Outcome (2026-09-10)

Training and compact evaluation completed. The new actor achieved final balance
in only 3/20 cases, versus 20/20 for current1.5/delta3; it had zero unsafe episodes
but 3 late losses. Its centered slow pair is `+0.4896 +/- j3.0268`, locally unstable.
The improved fast pair does not compensate for lost sustained balance. This is
not a slower-settling rejection. Do not promote this candidate.
See [HANDOFF.md](HANDOFF.md) for the current recommendation and evidence summary.
The commands below document reproduction; the default run already exists and
must not be overwritten or automatically restarted.

This is a fresh SAC experiment, not a resumed run or a controller wrapper.
Only action-change weight changes from 3 to 4 relative to the completed
current1.5/delta3 run. Seed0, 2M decisions, gamma0.99, actor/critics64x64,
nominal plant, safety limits and training success criteria remain unchanged.
The successful local correction is NOT applied during training or evaluation.
This reward experiment does not guarantee the same local poles as that correction.

## VS Code

Choose **Terminal > Run Task**:

- **Current1.5 delta4: overnight** runs preflight, training, then evaluation
  sequentially, stopping if a task fails.
- **Current1.5: TensorBoard** shows five historical/new runs on port6013.

For separate manual operation use **Current1.5 delta4: preflight**, then
**Current1.5 delta4: train**, then **Current1.5 delta4: evaluate** after training
has finished. Do not run these separately while the overnight sequence is active.
Nothing starts on workspace opening. Task output is visible in integrated terminals.

Training uses the original Conda rl-env interpreter. Evaluation uses the existing
isolated lqr_study_env interpreter and the source snapshot saved by training.
These task interpreter paths must be adjusted on another machine. Environment
setup is documented in [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md).

## PowerShell Commands

From the repository root, run these separately (not alongside their task equivalents):

```powershell
& 'C:/Users/pmic/.conda/envs/rl-env/python.exe' src/current15_delta4.py
& 'C:/Users/pmic/.conda/envs/rl-env/python.exe' -X faulthandler -u src/current15_delta4.py --execute
& 'src/runs/lqr_study_env/Scripts/python.exe' -X faulthandler -u src/runs/nominal_sac_current15_delta4_seed0_v0/source/current15_delta4.py --evaluate-only --execute --job-dir src/runs/nominal_sac_current15_delta4_seed0_v0
```

The first command only validates inputs. The second only trains. The third only
evaluates and requires successful training. Omit `--execute` for evaluation checks.
The overnight task automates this ordering; it never starts evaluation concurrently
with training. Training rejects any existing output directory. Evaluation refuses
to run twice over the same output, even following interruption.

Default output: `src/runs/nominal_sac_current15_delta4_seed0_v0`.
For a future separate run use `--job-dir src/runs/<new-name>` for training and
evaluation, updating the source-snapshot and TensorBoard paths accordingly.

## Compact Evaluation

Final evaluation is limited to **20 paired cases**, seeds580000-580019, each30s:
40 episodes total comparing the completed current1.5/delta3 actor with the new
current1.5/delta4 actor. These seeds are declared before inspecting this run.
All episode plots/traces are saved. No284-case evaluation is invoked or resumed.

Training still validates on50 fixed nominal cases every100k steps to select
its best checkpoint, exactly as before. This is distinct from final evaluation.
Changing those training checks would change checkpoint selection and the experiment.

Report priorities are unsafe episodes, sustained final balance, late losses,
current-change RMS, total variation, absolute>=25Hz current RMS, and local poles.
First and last5s current windows are paired; incomplete windows are excluded
for BOTH models while failures remain counted. Five-second success and finish
times are diagnostic only. There is **no slower-settling or slow-decay rejection**.
Final balance still has a finite30s horizon. No automatic model promotion occurs.

Local pole analysis reports every discovered equilibrium, including unstable
ones, using the existing finite-difference-checked5ms closed loop. Root search
is non-exhaustive.20 resets and one training seed are exploratory, not a hardware
or robustness qualification. Reward values differ across weights and should
not be interpreted as directly comparable controller quality scores.

## Results and Dashboard

Open http://127.0.0.1:6013/#scalars and select `delta4_current1.5` for the new run.
Other labels are `delta3_current1.5`, `delta3_current1`, `delta4_current1`,
and `delta5_current1`. New curves appear after the first training events.
Only one TensorBoard server is needed. The existing dashboard task was updated
to include the fifth log directory; an already-running server needs restarting
to pick up a changed logdir specification. Stop TensorBoard with Ctrl+C in its
own terminal, not the training terminal. If6013 is occupied by another service,
choose another port explicitly; do not terminate unrelated processes.

Outputs relative to the job directory:

| Output | Purpose |
|---|---|
| `protocol.json`, `source/` | Frozen settings, seeds, input hashes and source |
| `candidate/training/tensorboard/` | Training/validation curves |
| `candidate/training/best/best_model.zip` | Validation-selected actor |
| `candidate/training/resume/` | Rolling model/replay recovery snapshots |
| `reference/` | Frozen current1.5/delta3 and new delta4 models |
| `training_completed.json` | Training succeeded; evaluation can start |
| `evaluation/` | All40 episode plots/traces, paired CSVs, summary and poles |
| `REPORT.md`, `completed.json` | Finished evaluation report and completion marker |
| `status.json`, `failed.json` | Progress/caught exception details |

`completed.json` means the pipeline finished, not that the candidate is better.
Native exits, Ctrl+C or machine shutdown can leave stale status. Do not infer
completion from an empty error log or restart over existing artifacts.

## Overnight Precautions

Keep VS Code open, the machine on AC power, and Windows awake. Prevent automatic
sleep/reboot yourself; the task does not alter Windows settings or survive reboot.
The earlier memory-exhaustion cause remains unresolved. Use one trainer and one
dashboard; watch Task Manager's **Committed** memory, not only process working set.
There is no automatic memory monitor or unattended safety guarantee.

The stopped full swing-up comparison is preserved as partial evidence:64 of284
paired cases completed before interruption. It is not used as full qualification
and will not be restarted by these tasks.

Focused workflow tests:

```powershell
& 'src/runs/lqr_study_env/Scripts/python.exe' -m unittest discover -s src -p test_current15_delta4.py -v
```
