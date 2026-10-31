# Camera heights above the paper, lab arm, 2026-10-30

Both RealSense D435i cameras measured with `scripts/measure_camera_heights.py` (#46). Each camera
averages 30-60 depth frames aligned to colour (848x480), finds the sheet by colour, deprojects it
with the colour intrinsics and fits a plane (RANSAC + least squares). Height = perpendicular distance
from the camera's optical centre to the paper plane. The same fit on the wood around the sheet is a
cross-check, since the sheet lies on the table.

## Results

| Capture | Camera | Arm TCP z | Height above paper | Along optical axis | Tilt from vertical | Fit RMS | Paper vs table |
|---|---|---|---|---|---|---|---|
| `wrist_pen_touching` | wrist | 50.44 mm (pen tip on paper) | **201.6 mm** | 201.8 mm | 2.6 deg | 0.55 mm | 0.1 mm, 0.1 deg |
| `wrist_tcp225` | wrist | 225.15 mm | 382.3 mm | 382.6 mm | 2.6 deg | 0.74 mm | 0.4 mm, 0.1 deg |
| `wrist_tcp275` | wrist | 275.15 mm | 433.2 mm | 433.5 mm | 2.1 deg | 0.72 mm | 0.3 mm, 0.5 deg |
| `overhead_arm_clear` | overhead | arm at its initial pose, out of view | **~837 mm** (see below) | 850-865 mm | 10.8 deg | 2.6-3.1 mm | see below |

**Wrist camera** (serial 109622070191). Its height follows the arm exactly:
camera height = 1.032 x (TCP z - 50.44) + 201.7 mm, residuals under 0.4 mm over the three poses.
- The optical centre sits **201.6 mm above the pen tip** (measured with the tip touching the paper),
  tilted about 2.5 deg from the tool axis.
- The D435i depth reads 3.2% long against the arm's own motion (50.9 mm for a 50.0 mm move).
  Taking the arm as ground truth, the offset is about 196 mm. D435i spec: +-2% at this range.
- The pen sticks out 172 mm on the nominal TCP; this pen touched at TCP z 50.44 mm.

**Overhead camera** (serial 134322070512), arm moved out of view:
- The tabletop around the sheet gives 838 mm and the whole table 837 mm. The sheet alone (white arm pixels
  removed, a third of the colour mask) extrapolates to 848-849 mm with the same 10.8 deg tilt.
- The sheet is ~40 cm off the camera axis; at 85 cm the D435i depth noise (~3 mm) tips a plane fitted to
  the sheet by ~1 deg, which moves the extrapolated height ~10 mm. The table spans more of the view, so
  **~837 mm +-10 mm** is the better estimate. If this camera also reads ~3% long, about 810 mm.
- A pose from the sheet's four corners (PnP with the intrinsics) would pin this down; it needs the paper size.

## Intrinsics (colour stream, 848x480, depth aligned to it)

| Camera | fx | fy | ppx | ppy | distortion |
|---|---|---|---|---|---|
| wrist | 606.84 | 606.87 | 419.81 | 232.21 | none (inverse Brown-Conrady, zero) |
| overhead | 603.74 | 602.58 | 428.01 | 238.62 | none |

## Files

- `*.json`: arm pose at capture (TCP, joints, TCP offset), height, axis distance, tilt, the paper normal in
  the camera frame, fit residuals, the table cross-check and the intrinsics.
- `*_paper.png`: colour frame with points on the fitted plane in green and off it in red.
- Depth arrays (`*_depth.npy`) stay local in `isaac-sim/outputs/camera_heights/20261030/`.
