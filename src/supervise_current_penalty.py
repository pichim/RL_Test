"""Run the current-penalty pipeline with independent exit-code reporting."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JOB = ROOT / "src/runs/nominal_sac_current15_delta3_seed0_retry1_v0"


def write_event(directory, name, payload):
    with (directory / name).open("x", encoding="utf-8") as handle:
        json.dump({"utc": datetime.now(timezone.utc).isoformat(), **payload}, handle, indent=2)
        handle.write("\n")


def supervise(command, directory, cwd):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with (directory / "stdout.log").open("x", encoding="utf-8") as stdout, \
                (directory / "stderr.log").open("x", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr)
            write_event(directory, "started.json", {"supervisor_pid": os.getpid(),
                        "child_pid": child.pid, "command": command, "cwd": str(cwd)})
            exit_code = child.wait()
        write_event(directory, "exited.json", {"child_pid": child.pid, "exit_code": exit_code,
                    "state": "exited", "note": "Check pipeline completed.json; exit zero alone is not qualification."})
        return exit_code
    except Exception:
        write_event(directory, "supervisor_failed.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    args = parser.parse_args()
    job = args.job_dir.resolve()
    if job.exists():
        raise FileExistsError(f"Refusing existing experiment: {job}")
    monitor = job.with_name(job.name + "_supervision")
    command = [sys.executable, "-X", "faulthandler", "-u", str(ROOT / "src/current_penalty.py"),
               "--execute", "--job-dir", str(job)]
    sys.exit(supervise(command, monitor, ROOT))
