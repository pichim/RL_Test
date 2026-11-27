# Furuta swing-up and balance

Train one symmetric neural controller from scratch with SAC for the periodic
Furuta pendulum. The controller commands up to ±1 A. LQR supplies local
training examples. The benchmarks evaluate the learned actor and include
LQR as a local comparison. The equations, observations, reward, and
learning procedure are in [README_MATH.md](README_MATH.md).

## Train and evaluate

From the repository root, activate `rl-env` and run the trainer. In VS Code,
select the `rl-env` interpreter and run [src/train.py](src/train.py).

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rl-env
python -u src/train.py
```

Edit [src/config.py](src/config.py) to change the training or evaluation
settings. The current defaults use a 16×16 actor, 64×64 critics, and
300,000 decisions. Training and checkpoint selection run at 200 Hz. The same
selected checkpoint then runs through the nominal, upright, and
post-capture benchmarks at both 200 Hz and 500 Hz. Both rates use the
same benchmark starts and a 1 ms plant integration step. The current
command is held for five integration steps at 200 Hz and two at 500 Hz.

The trainer prints a new `runs/local_…/` directory with these outputs.

- `config.json` and `training.json` record the task and training settings.
- `best/best_model.zip` is the selected checkpoint. `actor.pt2` is its exported actor.
- `evaluation/` and `evaluation_500hz/` contain the effective task settings, benchmark results, and selected episode plots.
- `capture_starts.csv` is the shared post-capture fixture. `timing.json` records training and total evaluation time.

Compare success, safety, and completion time alongside RMS current slew
in A/s. RMS current change in A per update depends on the update rate.

To create a fresh Miniforge environment on Linux, run the following commands.

```bash
conda create -n rl-env python=3.12.16 pip
conda activate rl-env
python -m pip install 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

## Compare training seeds

```bash
python -u src/train_seeds.py
```

Edit `SEEDS` in [src/train_seeds.py](src/train_seeds.py) to choose the seeds.
The default runs seeds 1, 2, and 3 sequentially. Every seed uses the same
settings from [src/config.py](src/config.py), including the training budget
and the validation and evaluation starts. Each seed gets its own run
directory and both rate evaluations.

## Keep training running over SSH

1. Connect to the desktop with VS Code Remote SSH and open a terminal in the
   repository root on the desktop.
2. Create the session with `tmux new -s furuta`. If `furuta` already exists,
   use `tmux attach -t furuta` to return to it. Check the pane before starting
   a second run.
3. Inside tmux, activate `rl-env` and run `python -u src/train.py`.
   The `-u` flag shows Python output as it is written.
4. To detach, press the tmux prefix, release it, then press **d**. The
   default prefix is **Ctrl+B**. The reference desktop uses **Ctrl+S**.
   Run `tmux show-options -g prefix` to check the configured default on
   another machine. After detaching, you can close VS Code or disconnect
   SSH. Training continues while the desktop stays awake and the tmux
   session is alive.
5. Reconnect to the same desktop over SSH. Run `tmux ls` to list sessions,
   then `tmux attach -t furuta` to see the training terminal again. After a
   run finishes and the shell prompt returns, you can start the next run in
   the same session.

See the [tmux guide](https://github.com/tmux/tmux/wiki/Getting-Started#attaching-and-detaching)
for the default key bindings.

## Watch training in TensorBoard

SAC writes TensorBoard logs under `runs/local_…/tensorboard/`. In a
separate remote terminal, from the repository root with `rl-env` active, start TensorBoard.

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

This scans all runs. To view just one run, replace `runs` in `--logdir` with
that run's TensorBoard directory, for example
`runs/local_YYYYMMDD_HHMMSS_ffffff/tensorboard`. Use the path printed by
the trainer. If port 6006 is already in use, stop the previous server from
its TensorBoard terminal or use a different port.

In VS Code's **Ports** view, choose **Forward a Port** and enter **6006**.
Open the forwarded local address shown there, usually
`http://localhost:6006`. After reconnecting over SSH, restore the forward
and restart TensorBoard if needed. It reads the saved logs and can run
independently of the training terminal. [VS Code port forwarding](https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnel)

### Read the charts

Start with **Scalars → `eval/success_rate`**. Choose **Step** for the x-axis
and set **Smoothing** to **0** to see the individual validation points. A
step is one 5 ms control decision. By default, validation uses the same
50 starts every 25,000 decisions.
A value of `0.80` means 40 of 50 starts held balance for a full second.
The trainer selects the checkpoint primarily by this score.

| Chart                    | What it tells you                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `eval/mean_ep_length`    | An episode can end early from either success or a safety violation. The full five seconds are 1,000 decisions. Read alongside the success rate and the `unsafe` and `timeout` counts in `validation_metrics.json`. |
| `eval/mean_reward`       | Average total reward during validation. It can rise while fewer starts balance, so check the success rate first.                                                                                                   |
| `rollout/*`              | Recent training episodes, including random warm-up actions and later exploration. Compare with deterministic `eval/*` to see how the controller performs without exploration.                                      |
| `teacher/local_loss`     | Error against sampled LQR commands and slopes around upright. A low value shows good agreement on those examples. The swing-up and recovery tests measure the controller's behavior.                               |
| `train/*` and `time/fps` | Optimizer diagnostics and training throughput. Use evaluation to judge controller quality.                                                                                                                         |

The dashboard is TensorBoard's standard Scalars view. The `teacher/*`
charts are project-specific. For capture and LQR-region counts, open
`validation_metrics.json`. For actual commanded current over time, open
the episode plots or CSVs in `evaluation/` and `evaluation_500hz/` after training.

## Evaluate and export

```bash
python src/evaluate.py
```

This defaults to [the frozen reference](models/reference/results.json).
Change `EVALUATION.model` in `src/config.py` to score another checkpoint.
The evaluator reads the task config saved with the model. For a standalone
evaluation, `Evaluation.capture_starts` selects the post-capture start list.
Point it at a run's saved `capture_starts.csv` to reproduce that panel.
The standalone evaluator writes both rate folders under a new
`runs/evaluation_…/` directory. Each episode plot shows angles, angular
velocities, and current. Its CSV keeps the unwrapped arm angle.

Training exports `actor.pt2` alongside the selected SB3 checkpoint. Load it
with `torch.export.load(path).module()`. The input is a batch of seven
float32 features in training order. Multiply the normalized output by the
saved maximum current in amperes.

```bash
python -m unittest discover -s tests
```

See [PRESERVATION.md](PRESERVATION.md) for the frozen reference,
archival workflow, and actor-width comparison.
