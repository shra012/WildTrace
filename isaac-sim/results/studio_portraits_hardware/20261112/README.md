# Portraits and cats on the physical arm with a new pen, 2026-11-12

Physical xArm 7, ACT (`act_real_kinematic_v2`), `run_xarm7_act_hardware.py --execute` with `--paper-plane`
(the opt-in surface option, local runner change), `--stall-release-s 0.5 --floor-below-touch-mm 0.6
--start-clearance-mm 25`. Kept for training and analysis: every run folder has the 60 Hz per-tick log
(`act_ticks.csv.gz`: commanded joints, planned target, ideal-FK tip, tracking error, real-tip lag), the SDK
robot report (`robot_report.jsonl.gz`, ~10 Hz real TCP and joints), `events.jsonl`, `summary.json` (finished
runs), the trajectory and the pen surface used. Photos: `park_photo.png` (wrist camera, about 0.6 mm/px) and,
from example4 on, `closeup.png` / `closeup_full.png` (wrist camera centred 260 mm above the drawing, about
3.2 px/mm), `plan_overlay.png` and `stroke_shifts.csv` (each stroke's best shift against the plan, from the
close-up). Overhead videos are kept locally only (`isaac-sim/outputs/studio_pretty/video/`).

## Pen and calibration (`pen_calibration/`)

- New pen, 24 mm longer than the 11-05 marker. Paper height from the wrist depth camera (588.0 mm at TCP 429.65,
  plane rms 1.3 mm). Pressure set from stepped test lines at the user's X (drawing centre moved to (499, 100));
  the user picked each level by eye.
- Without holder: line 1 (TCP 76.88 at x 505) too light, line 4 (76.13 at x 517) drew the cat.
- With the pen holder: line 3 (76.38 at x 513), then 76.50 at x 509, then test line B (0.125 mm lighter,
  `holder_20261112_lineB.json`, centre z 76.504) as the current setting.
- All surfaces use the slope measured on 2026-11-05 (`marker_20261105_measured_m04`).

## Runs

See `runs.csv` (one row per run: time, real-tip RMSE / p95 / max while moving, commanded RMSE, stall releases,
lowest point below the surface, safe stop reason, surface, note).

| Run | Drawing | Pressure | Result |
|---|---|---|---|
| cat1163_pen_line1 | Cat `1163bcea` | pen in gripper, line 1 | light, partial ink |
| cat1163_pen_line4_pass2 | Cat `1163bcea` | line 4 | second pass, outlines about 3 mm apart |
| cat1163_pen_line4 | Cat `1163bcea` | line 4 | fresh sheet, complete single outline |
| cat1163_holder_line3, _b | Cat `1163bcea` | holder, line 3 | user: upper part missing |
| cat1163_holder_7650 | Cat `1163bcea` | holder, 76.50 | safe stop, one control tick took 529 ms |
| face_fb1b8bca_holder_line0 | Face `fb1b8bca` | holder, line 0 | finished |
| face_fb1b8bca_holder_7650 | Face `fb1b8bca` | holder, 76.50 | complete in one pass |
| example2_part1, _part2 | Studio example2 (183 strokes) | holder, 76.50 | stopped at stroke 106 (1124 ms tick), resumed with `--skip-strokes 106` |
| example4_original | Studio example4 (352 strokes) | holder, 76.50 | finished in 43 min; strokes shifted up to about 6 mm |

**example4 shift.** The arm's own real-tip error stayed at 0.44 mm, the overhead video shows the sheet did not
move, but in the close-up strokes drawn early (hair) and late (collar) sit up to about 6 mm off the plan, mostly
along +y, while the face strokes are within 1 mm (`stroke_shifts.csv`). So the pen tip moved relative to the
gripper during the run. The user then fixed the pen tip and holder and the drawing was repeated.

**Slow control ticks.** Two long runs stopped on the 0.5 s tick guard (529 ms, 1124 ms) while the laptop was
on AC with Edge, Cursor and VS Code busy. The runner lifted the pen each time.
