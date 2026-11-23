# Periodic Furuta 16×16 scratch release

Frozen on 2026-11-23 after the seed-0 matched comparison. The
[local 225k actor](local/actor.pt2) is the current periodic nominal reference.
The [recovery 200k actor](recovery/actor.pt2) is the matched comparison.

Both runs used 250,000 decisions, a symmetric seven-input 16×16 actor, twin
64×64 critics, a 5 ms control period and ±1 A. They started from random
weights with empty replay and used the same 500 local LQR warm-up updates.
The recovery run added supervised LQR approach trajectories to the actor
updates during SAC. LQR recovered 180 of 192 sampled starts, producing 21,637
teaching transitions. The deployed policies each contain one deterministic
actor.

| Measure | Local teaching | Recovery teaching |
|---|---:|---:|
| Selected checkpoint | 225k | 200k |
| First perfect validation | 225k | 200k |
| Validation at 250k | 12/50 | 50/50 |
| Training wall time on RTX 3080 | 24 min 51 s | 24 min 58 s |
| Nominal balance on seeds 80000–80099 | 100/100 | 100/100 |
| Fresh nominal balance on seeds 91000–91049 | 50/50 | 50/50 |
| Upright benchmark | 81/81 | 81/81 |
| Fixed capture benchmark | 30/30 | 30/30 |
| Mean nominal time to finish the balance hold | 2.986 s | 3.163 s |
| Mean nominal RMS current change per command | 0.0293 A | 0.0350 A |
| Largest sampled upright pole modulus | 0.9783 | 0.9831 |

Both selected actors had zero unsafe episodes on these simulation panels.
The recovery run reached 50/50 validation sooner and kept that score at 200k,
225k and 250k. These training comparisons cover one seed. The nominal plant
has direct current and ideal state inputs. Lag, command delay, randomized
mechanics, and sensor noise are future tests for this release.

## Evidence and reproduction

Both checkpoints use the [same task configuration](config.json). Each variant
contains the selected `model.zip`, the standalone `actor.pt2`, training
settings, validation history, nominal summary, upright diagnostics, and
capture and upright benchmark results. The [fresh nominal results](fresh_nominal.json)
include individual episodes. The [fixed capture starts](capture_starts.csv)
are evaluation data from the preceding pilot. The recovery teacher draws
separate simulated starts.

Verify the saved artifacts and the training and evaluation source from the
repository root.

```bash
(cd models/nominal_scratch_actor16_v1 && sha256sum -c SHA256SUMS)
sha256sum -c models/nominal_scratch_actor16_v1/source.sha256
```

With `rl-env` active, reproduce the fixed nominal evaluation using a fresh
output directory. The evaluator resolves the saved configuration from the
release directory.

```bash
release=models/nominal_scratch_actor16_v1
out="src/runs/frozen_scratch_local_$(date +%Y%m%d_%H%M%S)"
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python src/evaluate.py --model "$release/local/model.zip" --results-dir "$out" \
  --controller sac --mode nominal --episodes 100 --base-seed 80000
```

Load the standalone export with `torch.export.load(path).module()`. Its input
is an N×7 float32 batch in the training observation order. Its normalized
output is multiplied by the 1 A current limit. See the
[training commands](../../README.md#matched-250k-scratch-runs-with-lqr-recovery)
for the scratch comparison. The full original runs and rollout plots are under
`src/runs/nominal_sac_scratch_{local,recovery}_seed0_20261123_182519/`.
