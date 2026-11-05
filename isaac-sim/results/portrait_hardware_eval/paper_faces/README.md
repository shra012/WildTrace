# Paper faces on the physical arm, pencil (issue #52), 2026-11-04

The 5 paper faces in `isaac-sim/inputs/trajectories/Face/`, drawn with ACT (`act_real_kinematic_v2`) and a
**pencil** (not marker B as #52 asks; a marker set can follow). One fresh US Letter sheet per face. `<id>.jpg` is
the final sheet for each face (wrist camera from the park pose); `20261104/` holds every run with its
`summary.json`, `events.jsonl`, `robot_report.jsonl` and photo, failures included.

Common settings: `run_xarm7_act_hardware.py --execute --stall-release-s 0.5 --start-clearance-mm 25
--start-joints-deg 345.3 9.9 25.8 59.1 -5.4 50.3 14.4`. A pencil needs real contact pressure, so depth is the
main variable:

- Run 01 used one touch height (no plane). From run 02 on, `--paper-plane` (opt-in, #52 follow-up) makes pen-down
  heights follow a surface fitted to the three-corner plus centre touch (`../../real_arm_paired/20261104/plane_touches_20261104.json`).
- `surface_bowl` adds a shallow bowl fitted to a pencil pressure ladder. `surface_bowl_sharpened` is the same surface 1.0 mm lower, found by a
  camera-checked contact search after the pencil was sharpened. `_m03`, `_m08`, `_m115` are that surface shifted
  down 0.3, 0.8 and 1.15 mm (the runner caps `--z-offset-mm` at -1).
- "Depth" is the total pen-down offset against `surface_bowl_sharpened` where it applies.

| NN | Face | Finished | Path RMSE (mm) | p95 | Max | Commanded RMSE | Surface | z-offset (mm) | Lowest below surface (mm) | Time (s) | Stall releases | Ink on the sheet |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | `58265341` | yes | 0.464 | 0.977 | 1.250 | 0.602 | none (touch 74.40) | +0.2 | - | 218.0 | 0 | whole face, light |
| 02 | `cdb9aedb` | yes | 0.419 | 0.890 | 1.495 | 0.549 | `plane_touches` | +0.2 | 0.124 | 231.2 | 0 | blank: the centre sits 0.23 mm below the corner plane |
| 03 | `cdb9aedb` | no | - | - | - | - | `surface_bowl` | +0.15 | - | - | - | stopped by hand: pressure too low, no marks |
| 04 | `cdb9aedb` | yes | 0.419 | 0.878 | 1.221 | 0.548 | `surface_bowl` | -0.2 | 0.489 | 231.8 | 0 | whole face |
| 05 | `e00984bb` | no | - | - | - | - | `surface_bowl_sharpened` | 0.0 | - | - | - | stopped at target 1404/3581: no marks |
| 06 | `e00984bb` | yes | 0.443 | 0.898 | 1.140 | 0.611 | `surface_bowl_sharpened` | -0.5 | 0.759 | 229.0 | 0 | lower face only |
| 07 | `e00984bb` | yes | 0.459 | 0.903 | 1.148 | 0.613 | `surface_bowl_sharpened` | -0.9 | 1.160 | 229.8 | 0 | same sheet; face complete, hair faint |
| 08 | `e00984bb` | yes | 0.443 | 0.880 | 1.245 | 0.599 | `_m03` (depth -1.3) | -1.0 | 1.283 | 75.8 | 0 | same sheet; 5 hair and right-side strokes only; top arc still partial |
| 09 | `f4d9516c` | yes | 0.518 | 1.026 | 1.246 | 0.676 | `_m08` (depth -1.8) | -1.0 | 1.375 | 182.7 | 0 | all 15 strokes |
| 10 | `fb1b8bca` | yes | 0.475 | 0.935 | 1.210 | 0.640 | `_m115` (depth -2.15) | -1.0 | 1.406 | 223.7 | 2 | all 18 strokes |

Path RMSE is the real tip (10 Hz SDK reads) against the planned path; commanded RMSE is the policy's commanded tip.
No manual nudges in any run.

**Findings**
- With a pencil, the fitted plane is not enough: toward +x the paper needed 0.4 to 0.9 mm more depth than the
  surface predicts (contact search just past the drawing), so partial strokes cluster on that side.
- Pressing deeper did not hurt tracking: path RMSE stayed at 0.44 to 0.52 mm from -0.5 to -2.15 mm depth.
- Lines in the photos are pencil-light; scans will score them better than the 0.6 mm/px photos.
