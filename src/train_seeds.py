"""Train and evaluate several seeds in parallel on separate Linux CPU cores."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
import os
from pathlib import Path

from config import TRAINING

SEEDS = (0, 1, 2)
DECISIONS = 550_000


def available_cores():
    """Return one allowed logical CPU per physical core."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        siblings = (topology / "thread_siblings_list").read_text().strip()
        cores.setdefault(siblings, cpu)
    return tuple(cores.values())


def run_seed(seed, cpu):
    os.sched_setaffinity(0, {cpu})
    from train import main as train_and_evaluate

    print(f"\n=== Seed {seed} | CPU {cpu} | {DECISIONS:,} decisions ===", flush=True)
    train_and_evaluate(replace(TRAINING, seed=seed, decisions=DECISIONS))


def main():
    cpus = available_cores()
    if len(cpus) < len(SEEDS):
        raise RuntimeError(f"Need {len(SEEDS)} available physical CPU cores, found {len(cpus)}")

    with ProcessPoolExecutor(max_workers=len(SEEDS), mp_context=get_context("spawn")) as pool:
        # Consume results so errors in a worker are reported by the launcher.
        for _ in pool.map(run_seed, SEEDS, cpus[:len(SEEDS)]):
            pass


if __name__ == "__main__":
    main()
