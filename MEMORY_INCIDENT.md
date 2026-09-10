# TensorBoard and Training Memory Incident

Investigation: 2026-09-09, after the user restarted VS Code.

## Subsequent Reboot and Cleanup

Later on 2026-09-09, the user rebooted the computer. Commit usage was then
10435833856 of 38258860032 bytes, with 21641 MiB available RAM. Only editor
Python services remained. The retry2 100k model/replay hashes, loading,
configuration compatibility, finite policy parameters and inference all passed.

The user subsequently chose a fresh, manually operated experiment instead of
recovery and explicitly requested deletion of all three interrupted attempts
and their supervision logs. Those outputs, including the validated checkpoint,
were deleted. Completed baseline/delta4/delta5 runs and the LQR study/environment
were retained. The observations below are historical evidence, not claims that
the deleted files remain available. New runs may reuse the original job name.

Use [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) for the replacement manual
training, single-dashboard and evaluation commands. Nothing was launched during
this setup. The reboot and cleanup do not establish the cause of memory growth.

## Confirmed Evidence

- TensorBoard's supplied traceback contains MemoryError while reading the next
  TFRecord header in its reloader thread. That exception alone does not prove
  why the entire terminal or training supervisor subsequently exited.
- Windows System event 2004, Resource-Exhaustion-Detector, at
  2026-09-09T12:19:15Z (14:19:15 local) explicitly reports low virtual memory.
- SystemCommitCharge: 136436367360 bytes; SystemCommitLimit: 136506236928 bytes.
  This is approximately 127.07 / 127.13 GiB, over 99.9% committed.
- PhysicalMemoryUsage: 33094668288 of 33427021824 bytes, approximately 99%.
- The event lists trainer PID27960 at 851808256 bytes (about 0.79 GiB),
  Dropbox at 1170817024 bytes, and Firefox at 778178560 bytes. These process
  figures do not explain the total commit charge. The event's aggregate process
  commit is only 17026928640 bytes. Do not attribute the discrepancy to a
  particular application, driver, cache or leak without further evidence.
- At the later check, commit was 16807018496 of 80942141440 bytes, with
  22003 MiB available physical memory. These are post-restart observations,
  not measurements of the failure's onset. Page-file allocation was 45313 MiB.

## Event File Integrity

Read-only audit of all four files used by the comparison dashboard passed.
Every record had a valid length, header CRC, payload CRC, and Event protobuf.
The largest record was 72 bytes. The checker bounds allocations at 16 MiB and
rejects lengths beyond the remaining file before reading payloads.

| Run | File bytes | Records | Last recorded step |
|---|---:|---:|---:|
| Delta3/current1 | 278114 | 4899 | 2000000 |
| Delta4/current1 | 277092 | 4881 | 2000000 |
| Delta5/current1 | 329725 | 5808 | 2000000 |
| Delta3/current1.5 retry2 | 33860 | 598 | 176156 |

Checker: [src/audit_tensorboard_events.py](src/audit_tensorboard_events.py).
These files are not currently corrupt; a historical transient read issue cannot
be excluded solely by a later audit. System-wide exhaustion is the directly
documented failure condition, not a claim about its originating cause.

## Training Recovery

No trainer or supervisor was present during this investigation. Earlier statements
that retry2 stopped near 40740 steps and had no checkpoint are superseded:
the event file reached 176156 steps, and the resume pointer identifies step100000.

The snapshot directory contained model.zip, replay_buffer.pkl and state.json.
Its resume pointer identified snapshot_000000100000. Integrity and loading were
subsequently validated after reboot, before user-requested deletion.
Recovery from100000 would repeat at least76156 recorded decisions. It is not
a bit-for-bit continuation because complete environment/RNG state is not retained.

## Original Recovery Recommendations

These preceded the user's decision to delete the attempts and start fresh.

1. Do not start another training attempt until a monitored recovery is agreed.
2. Consolidate the five redundant old TensorBoards into one comparison dashboard
   after approval. They account for roughly2.4 GiB of committed memory in total;
   removing them saves resources but does not explain or fix a127 GiB incident.
3. Before recovery, log system commit/limit, available RAM, kernel pools and
   per-process private memory periodically outside the VS Code terminal lifecycle.
   Preserve a warning snapshot before commit reaches a conservative threshold
   such as85%; do not automatically kill unrelated user applications.
4. If system commit grows far beyond accounted process memory again, collect
   Windows memory diagnostics with the system administrator. Do not simply
   enlarge the page file or blame Dropbox because this repository is synced.
5. Validate the100k model/replay snapshot and resume in a new output directory,
   documenting the interruption and keeping all original logs unchanged.

No training or TensorBoard was launched, no old processes were stopped, no
event files were changed, and no package or page-file settings were modified
in this investigation. Only the audit helper and this note were added.
