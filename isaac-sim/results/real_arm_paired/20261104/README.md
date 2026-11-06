# Paired real-arm runs, ACT vs IK (issue #52), 2026-11-04

Physical xArm 7 (192.168.1.241), marker B, US Letter sheet in the usual placement, one fresh sheet per run.
Drawings are the 12 gold drawings listed in #52 (`isaac-sim/data/act_test/<Class>/<id>.json`), ACT and IK
alternating. This session covers runs 01-02; runs 03-24 follow in a later session.

- **ACT:** `run_xarm7_act_hardware.py --execute --z-offset-mm 0.0 --floor-below-touch-mm 0.6 --stall-release-s 0.5
  --max-seconds 900 --start-clearance-mm 25 --start-joints-deg 345.3 9.9 25.8 59.1 -5.4 50.3 14.4`, checkpoint
  `act_real_kinematic_v2`. Same floor and start settings as the marker B portraits of 2026-11-02.
- **IK:** `run_xarm7_hardware.py --execute`, the controller's `set_position` per waypoint, stopping at each one.
- **Touch height:** marker B, TCP z 51.687 at (484.8, 35.6) with a 2.4 deg tilt, moved to the drawing centre with
  the corner fit below (-0.23 mm): 51.46 mm. Used by both controllers. No `--paper-plane`; calibration files unchanged.
- **Offline checks first:** all 12 IK plans (`--plan-only`) and all 12 ACT drawings (`--offline`) pass.

Files per run: `<NN>_<class>_<id>_<act|ik>_summary.json`, `_events.jsonl`, `_robot_report.jsonl` (ACT only), and
`_photo.jpg` (wrist camera from the park pose; the 600 dpi scans for `ink_fidelity.py` come later).
`plane_touches_20261104.json` is the three-corner plus centre touch from part 3.

## Runs

| NN | Drawing | Controller (checkpoint) | Finished | Path RMSE (mm) | p95 | Max | Ink chamfer | Coverage @ 1 mm | Time (s) | Stall releases | Nudges | Date | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | Bird `c326f229ac98c00a` | ACT (`act_real_kinematic_v2`) | yes | 0.365 | 0.766 | 1.158 | scan pending | scan pending | 70.5 | 0 | 0 | 2026-11-04 | clean line; lowest commanded point 0.28 mm below the touch height |
| 02 | Bird `c326f229ac98c00a` | IK (`set_position`) | yes | 0.000 (settled) | 0.000 | 0.001 | scan pending | scan pending | 287.1 | - | 0 | 2026-11-04 | 821/821 waypoints, 0 retries; much heavier line than ACT, ink pooled on the head and wing |

**Reading the path RMSE:** ACT's value is the real tip (10 Hz SDK reads) against the planned path while moving.
IK stops and settles at every waypoint and is only read at rest, so its arm-reported error is about zero and is
not comparable. The sheet scans (`ink_fidelity.py`) are the like-for-like comparison between the two.

Runs 03 to 22 (9 pairs) are in `../20261105/README.md`.

## Still to run (as of 2026-11-04)

| NN | Drawing | Controller |
|---|---|---|
| 03, 04 | Bird `c3b05340ef29131f` | ACT, IK |
| 05, 06 | Cat `1163bceaf6134cf5` | ACT, IK |
| 07, 08 | Cat `70bc30f8a5f918eb` | ACT, IK |
| 09, 10 | Dog `4cf9d2757b8a6282` | ACT, IK |
| 11, 12 | Dog `cd5042b9731161f9` | ACT, IK |
| 13, 14 | Fish `ba3e1657db195fe4` | ACT, IK |
| 15, 16 | Fish `dd305a2886cbf23c` | ACT, IK |
| 17, 18 | Frog `6bbd7fa65eede66e` | ACT, IK |
| 19, 20 | Frog `8ae3adc42375beba` | ACT, IK |
| 21, 22 | Horse `a84882576126b982` | ACT, IK |
| 23, 24 | Horse `e659138f70574103` | ACT, IK |

Re-measure marker B's touch height at the start of the next session.
