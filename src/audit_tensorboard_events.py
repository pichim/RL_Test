"""Read-only, bounded TFRecord integrity check for TensorBoard event files."""

import argparse
import json
from pathlib import Path
import struct

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c


def audit(path):
    path = Path(path)
    size = path.stat().st_size
    records, largest, last_step = 0, 0, None
    with path.open("rb") as handle:
        while handle.tell() < size:
            offset = handle.tell()
            header = handle.read(12)
            if len(header) != 12:
                raise ValueError(f"{path}: partial header at {offset}")
            length, checksum = struct.unpack("<QI", header)
            if masked_crc32c(header[:8]) != checksum:
                raise ValueError(f"{path}: header checksum at {offset}")
            if length > 16 * 1024 * 1024 or length + 4 > size - handle.tell():
                raise ValueError(f"{path}: invalid/partial record length {length} at {offset}")
            payload = handle.read(length)
            checksum = struct.unpack("<I", handle.read(4))[0]
            if masked_crc32c(payload) != checksum:
                raise ValueError(f"{path}: payload checksum at {offset}")
            event = Event.FromString(payload)
            last_step = int(event.step)
            records += 1
            largest = max(largest, length)
    return {"path": str(path), "bytes": size, "records": records,
            "largest_record_bytes": largest, "last_step": last_step, "valid": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    paths = sorted({path for directory in args.directories
                    for path in directory.rglob("events.out.tfevents.*") if path.is_file()})
    if not paths:
        raise FileNotFoundError("No event files found")
    print(json.dumps([audit(path) for path in paths], indent=2))
