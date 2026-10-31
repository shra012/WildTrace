# Camera heights above the paper, lab arm, 2026-10-30

Both RealSense D435i cameras measured two independent ways (#46):

- **Depth** (`scripts/measure_camera_heights.py`): 30-60 depth frames aligned to colour (848x480), sheet found
  by colour, deprojected with the colour intrinsics, plane fitted (RANSAC + least squares). Height = perpendicular
  distance from the optical centre to the paper plane; the wood around the sheet is a cross-check.
- **Sheet corners** (`scripts/camera_pose_from_sheet.py`): the sheet's four edges are fitted as lines and
  intersected, then PnP (IPPE, planar) with the sheet size and the colour intrinsics (1280x720) gives the camera's
  full pose in the sheet frame. Pure geometry, no depth.

**The paper is US Letter (279.4 x 215.9 mm), not A4.** Corners fit Letter to 0.08 px (wrist) and 1.0 px
(overhead), A4 to 11.6 px and 4.7 px (`*_a4_rejected.json`).

## Results

| Camera | Arm TCP (mm, pen straight down unless noted) | Depth height | Corner (PnP) height | Tilt (PnP / depth) | PnP reprojection |
|---|---|---|---|---|---|
| Wrist | [426.9, 58.0, 380.0] | 540.1 mm | 534.5 mm | 1.0 / 0.2 deg | 0.08 px |
| Wrist | [426.9, 58.0, 430.0] | - | 585.3 mm | 0.9 deg | 0.08 px |
| Wrist | [426.9, 58.0, 480.0] | 642.2 mm | 632.3 mm | 1.3 / 0.3 deg | 0.14 px |
| Wrist | [489.1, 90.2, 275.1], pitch -2.6 deg | 433.2 mm | - | 2.1 deg | - |
| Wrist | [489.1, 90.2, 225.1], pitch -2.6 deg | 382.3 mm | - | 2.6 deg | - |
| Wrist | [373.0, 49.4, 50.4], pen tip touching, roll 3.6 deg | 201.6 mm | - | 2.6 deg | - |
| Overhead | arm swung aside | 837.4 mm (tabletop) | 833.3 mm | 12.7 / 10.8 deg | 1.0 px |

### Wrist camera (serial 109622070191)

- Over the same 100 mm of arm travel, depth changed 102.1 mm (2.1% long) and the corner fit 97.8 mm (2.2% short).
  Their average changed 99.95 mm, so the average is used.
- **With the pen pointing straight down, the optical centre is 207.7 mm above the pen tip**
  (same value at TCP z 380 and 480): camera height above paper = (TCP z - 50.44) + 207.7 mm, where 50.44 mm is
  the TCP z at which the current pen touches the paper. The camera looks within ~1 deg of straight down.
- The pen-touching capture (201.6 mm) had the wrist rolled 3.6 deg; the camera sits to the side of the tool axis,
  so it drops several mm in that pose. Use the straight-down value.

### Overhead camera (serial 134322070512)

- **~835 mm above the sheet** (corner fit 833.3 mm, tabletop depth 837.4 mm), **~12.7 deg from vertical**,
  854 mm along its optical axis.
- In the sheet frame (origin at the sheet centre, X along the long edge, Y along the short edge, Z up) it sits at
  [-47, 208, 833] mm and its optical axis meets the table at [137, 176] mm, just past the sheet's short-edge side.
  A rectangle is symmetric under a 180 deg turn, so the X/Y signs follow the image (see `pnp_overhead.json`).
- Earlier depth-only fits on the sheet itself extrapolated to 848-849 mm (a ~1 deg plane tilt from depth noise at
  85 cm); the corner fit settles it.

## Intrinsics (colour stream, zero distortion)

| Camera | Resolution | fx | fy | ppx | ppy |
|---|---|---|---|---|---|
| wrist | 848x480 | 606.84 | 606.87 | 419.81 | 232.21 |
| overhead | 848x480 | 603.74 | 602.58 | 428.01 | 238.62 |

The 1280x720 intrinsics used by the corner fit are in each `pnp_*.json`.

## Files

- `wrist_*.json`, `overhead_arm_clear.json`: depth captures (arm pose, plane fit, table cross-check, intrinsics).
- `pnp_*.json`: corner-fit poses (camera position and rotation in the sheet frame, corners, reprojection error).
- `*_paper.png`: depth plane fit, green on the plane and red off it. `pnp_*_corners.png`: detected corners.
- Depth arrays stay local in `isaac-sim/outputs/camera_heights/20261030/`.
