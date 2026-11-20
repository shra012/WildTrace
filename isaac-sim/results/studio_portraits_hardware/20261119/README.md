# Studio portrait on the physical arm with the new pen mount and stand, 2026-11-19

Same setup as `../20261112/` (ACT `act_real_kinematic_v2`, `run_xarm7_act_hardware.py --execute --paper-plane`,
`--stall-release-s 0.5 --floor-below-touch-mm 0.6 --start-clearance-mm 25`, drawing centre (499, 100)), with a
new pen mount and stand fitted to the gripper. Per run: 60 Hz tick log and SDK robot report (gzipped), events,
summary, trajectory, surface, wrist-camera close-up (centred, 260 mm above the drawing), plan overlay and per-stroke
shifts. The overhead video stays local (`isaac-sim/outputs/studio_pretty/video/`).

## Pen pressure (`pen_calibration/stand_20261119_line2.json`)

Test lines in the margin (y 168-180), offsets against the 2026-11-12 line B surface:

| x (mm) | Offset | TCP z (mm) | Ink |
|---|---|---|---|
| 530 | +2.0 | 78.81 | none |
| 538 | +1.5 | 78.41 | light |
| 534 | +1.0 | 77.86 | drew (chosen) |
| 542 | +0.5 | 77.46 | darker |

So the tip first touches between +2.0 and +1.5 mm; the chosen level is line B + 1.0 mm, centre z 77.504.

## Runs (`runs.csv`)

| Run | Drawing | Time | Real-tip RMSE / p95 / max (mm) | Stall releases | Result |
|---|---|---|---|---|---|
| example1_simplified | Studio example1, simplified (232 -> 138 strokes) | 1157 s | 0.372 / 0.703 / 1.38 | 61 | finished in one pass; face coherent; per-stroke shifts 0.2-0.8 mm (median per group); shoulders and collar light |
| example6_simplified | Studio example6 (glasses), simplified (127 -> 55 strokes) | 511 s | 0.371 / 0.807 / 1.22 | 4 | finished in one pass; glasses, face, collar and button complete; lines light, ink found for 27 of 55 strokes, those within 0.6 mm |
| example7_simplified | Studio example7 (couple, centre-line trace), 610 -> 427 strokes | 3321 s | 0.425 / 0.644 / 1.01 | 601 | finished in one pass (55 min); both figures complete, about 84 x 120 mm; per-stroke shifts mostly under 1 mm, 1.6-3 mm in strokes 200-299 |

The per-stroke shifts are down from up to about 6 mm on 2026-11-12 (pen moving in the old holder) to under 1 mm,
so the new mount holds the tip. The simplified trajectory is `example1_simplified/trajectory.json` (strokes joined
within 1 mm, strokes under 3 mm dropped outside the eye band, Douglas-Peucker 0.15 mm, nearest-neighbour order).

**example7 preparation.** The Studio file is normalised separately by width and height (`width_height_normalized`),
which would draw the couple about 1.4x too wide; x was rescaled by 1050/1498 to restore the aspect. Simplified as the
others, except that the man's face (box x 0.194-0.374, y 0.040-0.267 of the drawing) was kept as drawn (no joining or
dropping, Douglas-Peucker 0.08 mm) at the user's request, which is why it needed 601 stall releases.
