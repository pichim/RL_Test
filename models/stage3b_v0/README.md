# Stage-3b v0 frozen controller

This directory freezes the selected seed-0 Stage-3b controller at absolute SAC
timestep 2,400,000. It continued the selected Stage-2 learner for 900,000
decisions, reaching curriculum scale 0.9. Standalone evaluation used the full
configured uncertainty ranges and achieved 600/600 SAC-only plus hybrid
successes with no unsafe episode on selection seeds 10000--10299.

`model.zip` is the deployment/evaluation model. It intentionally has no replay
buffer and must not be presented as an exactly resumable learner. `config.json`
is the full-scale evaluation configuration. `manifest.json` records provenance,
hashes, and the acceptance evidence. `acceptance.json` freezes the holdout seed
range and pass limits before that unseen evaluation is run. The immutable
holdout outcome is summarized in `holdout_report.json`.

The selection seeds were used while choosing this checkpoint. Finalization
therefore used untouched holdout base seed 50000. SAC achieved 100/100 nominal,
99/100 randomized broad, and 100/100 randomized downward-rest success. Broad
seed 50116 crossed the negative arm limit from a rare near-limit, high outward
velocity reset. The same state fails before hybrid handover. The frozen model
therefore remains a useful simulation candidate but did not pass finalization.

A second independently seeded protocol restricted randomized initial arm speed
to `+/-0.5 rotations/s` while retaining the other uncertainties. It passed
`100/100` randomized and `100/100` downward-rest episodes, but nominal seed
`60066` started near upright at arm angle `+0.985 rad` with outward arm speed
`+1.906 rad/s` and crossed the positive arm limit after `0.276 s`. Hybrid
reproduced the failure before handover. The predeclared protocol and outcome are
stored in `acceptance_half_rps_v1.json` and
`holdout_half_rps_v1_report.json`.
