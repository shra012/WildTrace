# Face portraits on the physical xArm 7, 2026-11-02

Two face sketches from WildTrace Studio drawn on the lab arm with ACT (`act_real_kinematic_v2`) through
`scripts/run_xarm7_act_hardware.py`. The trajectories are the Studio jobs' `trajectory.json` (MediaPipe face
landmarks and hair, `wildtrace/face_lineart.py`); the uploaded photos are not committed.

| Drawing | Studio job | Strokes / points |
|---|---|---|
| `e67870b5` (man, short hair) | `job_e67870b5` | 48 / 411 |
| `d3a63644` (woman, long hair) | `job_d3a63644` | 59 / 483 |

**Settings for every real run:** paper `paper_center_mm` xy [476.55, 86.613] (US Letter sheet), plane at the tool's touch
height plus `--z-offset-mm`, `--stall-release-s 0.5`, `--start-clearance-mm 25`, wrist starting on its +14.4 deg
equivalent (`--start-joints-deg 345.3 9.9 25.8 59.1 -5.4 50.3 14.4`; the -345.6 deg start came within 3 deg of
joint 7's limit offline). Touch heights (TCP z with the tip on the paper): pencil 81.2 mm (the pencil sticks out
~30 mm more than the marker), marker A 51.614 mm, marker B 51.149 mm. Floor guard: pencil 0.2 mm, markers 0.6 mm
below the touch height. Each real run was checked offline first.

## Runs

Measured = the arm's reported tip against the path (`robot_report.jsonl`, 10 Hz); commanded = from the commanded
joints. mm throughout.

| # | Drawing | Policy | Tool | Run | Finished | Measured RMSE | p95 | Max | Commanded RMSE | Time s | Stall releases | Safety stop | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `e67870b5` | ACT `act_real_kinematic_v2` | pencil | dry, first 6 strokes, +8 mm | no | - | - | - | - | - | - | joint path to the start pose dips to z=127.1353530883789 mm at t=0.93 | refused before moving: the start-pose clearance check needs 40 mm and the pencil sits ~38 mm above the paper there; led to --start-clearance-mm |
| 2 | `e67870b5` | ACT `act_real_kinematic_v2` | pencil | dry, first 6 strokes, +8 mm | yes | 0.537 | 0.997 | 1.207 | 0.735 | 63.1 | 0 | - | pen 8 mm above the paper; checks the start move and tracking |
| 3 | `e67870b5` | ACT `act_real_kinematic_v2` | pencil | real, plane +0.4 mm | yes | 0.438 | 0.891 | 1.262 | 0.583 | 487.3 | 61 | - | drew all strokes but only a few faint marks: the plane sat 0.4 mm above the touch height to protect the nib |
| 4 | `e67870b5` | ACT `act_real_kinematic_v2` | marker A | real, plane 0.0 mm | no | - | - | - | - | - | - | control tick took 116 ms | safety stop at stroke 23 of 48: control tick took 116 ms (then 0.10 s limit); marker lifted 60 mm |
| 5 | `e67870b5` | ACT `act_real_kinematic_v2` | marker A | real, resume from stroke 23 | no | - | - | - | - | - | - | set_servo_angle_j returned 9 | stopped by hand: marker A ran out of ink |
| 6 | `e67870b5` | ACT `act_real_kinematic_v2` | marker B | real, plane 0.0 mm | yes | 0.486 | 0.912 | 1.306 | 0.633 | 507.8 | 50 | - | test on a used sheet |
| 7 | `e67870b5` | ACT `act_real_kinematic_v2` | marker B | real, plane 0.0 mm | yes | 0.514 | 0.966 | 1.305 | 0.651 | 498.9 | 36 | - | clean sheet |
| 8 | `d3a63644` | ACT `act_real_kinematic_v2` | marker B | real, plane 0.0 mm | yes | 0.49 | 0.905 | 1.273 | 0.642 | 582.7 | 52 | - | clean sheet; long hair below the shoulders faint (pressure falls off across the sheet) |

**Summary (ACT, finished real portraits with ink, runs 6-8):** 3/3 finished, measured path RMSE 0.497 mm mean
(0.486-0.514), p95 0.93 mm, 36-52 stall releases per portrait (tiny eye and lash strokes), 8.3-9.7 min each.

**Learned:**
- A pencil needs contact pressure: at +0.4 mm it barely marked. Markers draw well with the plane at the touch height
  (lowest commanded point 0.25-0.31 mm below it).
- Lines are bold on one side of the drawing and thin on the other in every drawing so far: the sheet or table is
  likely not level. A three-point touch to fit the paper plane would fix it.
- The face is drawn sideways on the sheet (the drawing frame's up is along the sheet's short edge).

## Files

`NN_face_<id>_act_<tool>_{summary.json,robot_report.jsonl,events.jsonl,topdown.png}` per run (runs 1, 4 and 5 stopped,
so they have no summary; 1 never moved), and `face_<id>_trajectory.json`.
