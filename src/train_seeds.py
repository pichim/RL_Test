"""Train and evaluate the same configuration with several training seeds."""

from dataclasses import replace

from config import TRAINING
from train import main as train_and_evaluate

SEEDS = (1, 2, 3)


def main():
    for seed in SEEDS:
        print(f"\n=== Seed {seed} | {TRAINING.decisions:,} decisions ===", flush=True)
        train_and_evaluate(replace(TRAINING, seed=seed))


if __name__ == "__main__":
    main()
