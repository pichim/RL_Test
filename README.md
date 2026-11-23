# Furuta swing-up and balance

Train a single symmetric neural controller from scratch for the nominal
periodic Furuta pendulum. The controller runs at 200 Hz and commands up to
±1 A. Local LQR supplies training examples. Evaluation and deployment use
the learned actor.

## Run

Use the RTX 3080 desktop's `rl-env` environment. From the repository root:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate rl-env
python -u src/train.py
```

Edit [src/config.py](src/config.py) for the seed, budget, validation frequency,
task parameters, or evaluation settings. In VS Code, select the `rl-env`
interpreter and run [src/train.py](src/train.py). The default is the seed-0
250,000-decision training recipe. The trainer validates every 25,000 decisions,
keeps the selected checkpoint, and runs evaluation automatically. It creates a
new timestamped directory under `runs/`.

For a fresh Miniforge environment on the desktop:

```bash
conda create -n rl-env python=3.12.16 pip
conda activate rl-env
python -m pip install 'torch==2.14.0+cu132' --index-url https://download.pytorch.org/whl/cu132
python -m pip install -r requirements.txt
python -m pip check
```

## Evaluate and export

```bash
python src/evaluate.py
```

This defaults to [the frozen reference](models/reference/results.json).
Change `EVALUATION.model` in `src/config.py` to score another checkpoint.
The evaluator reads the task config saved with the model. It writes nominal
results, the upright and post-capture benchmarks, and selected episode plots
under `runs/`. Each plot shows angles, angular velocities, and current.
Its CSV keeps the unwrapped arm angle.

Training exports `actor.pt2` alongside the selected SB3 checkpoint. Load it
with `torch.export.load(path).module()`. The input is a batch of seven
float32 features in training order. Multiply the normalized output by the
saved maximum current in amperes.

```bash
python -m unittest discover -s tests
```

See [PRESERVATION.md](PRESERVATION.md) for the historical snapshot and the
reference comparison.
