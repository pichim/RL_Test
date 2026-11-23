# Frozen guided nominal controller, 64×64

Selected from the 2026-11-22 desktop RTX 3080 run at 500,000 decisions, seed 0.
The deterministic actor is 7 → 64 → 64 → 1 with 4,737 parameters. It uses
the same weights for the original and reflected observations. Training used
centered LQR supervision, energy shaping with weight 0.25, a ±1 A current limit,
a 5 ms held-command period and 1 ms physics integration. The critics were
64×64. Validation used the unshaped reward.

The selected checkpoint passed 50/50 validation and 100/100 final nominal
evaluation on seeds 80000–80099, with zero unsafe episodes. Training took
35 min 12.4 s, and evaluation took 1 min 28.4 s. The largest sampled local
closed-loop pole modulus was 0.97525. See [the run report](FINDINGS.md) and
[nominal evaluation summary](nominal_summary.csv).

## Restore the selected controller

- `actor.pt2` is the standalone actor. Load it with
  `torch.export.load(path).module()`. It takes an N×7 float32 tensor in the
  original observation order. The output is normalized current, multiplied by
  the 1 A limit to obtain amperes.
- `model.zip` is the selected SB3 checkpoint for the original evaluator.
  `config.json` supplies its physical settings. `source.zip` contains the
  exact Python source files copied at run startup, including the custom SAC
  policy needed to load this checkpoint.
- `manifest.json` records SHA-256 hashes for the artifacts and source files.
  `training.json` and `experiment.json` preserve the recorded command,
  dependency versions, and training settings. The run used Python 3.12.16,
  PyTorch 2.14.0+cu132, Stable-Baselines3 2.9.0, Gymnasium 1.3.0, and NumPy 2.5.5.

From the repository root, activate the Linux `rl-env` environment as shown in
[the setup guide](../../README.md#linux-cuda-setup-and-nominal-benchmark).
Extract the frozen code and rerun the original evaluation in a fresh directory:

```bash
release_dir="$PWD/models/nominal_guided_1a_actor64_v0"
snapshot_dir="$(mktemp -d)"
python -m zipfile -e "$release_dir/source.zip" "$snapshot_dir"
results_dir="src/runs/reeval_guided_1a_actor64_$(date +%Y%m%d_%H%M%S)"
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python "$snapshot_dir/evaluate.py" --model "$release_dir/model.zip" \
  --config "$release_dir/config.json" --controller sac --mode nominal \
  --episodes 100 --base-seed 80000 --results-dir "$results_dir"
```

To retrain from scratch with the original settings, use the extracted
`nominal.py --guided --max-current 1 --timesteps 500000 --seed 0 --execute`
and a fresh `--job-dir`. Numerical variation across environments can produce a
different policy. Loading the frozen actor restores the original controller.

The full training history, plots, and separate final learner/replay pair remain
in the original Git-ignored run shown in `manifest.json`. Back up that run
directory separately if learner-state restoration or all raw plots matter.
The guided launcher currently starts fresh, and the selected evaluation
checkpoint has no portable replay buffer in this release.
