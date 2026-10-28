# ACT on the physical xArm 7: 30-drawing evaluation, 2026-10-27

`models/act_real_kinematic_v2` drew 5 drawings per class (Bird, Cat, Dog, Fish, Frog, Horse) on the lab
arm with `scripts/run_act_eval_batch.py`, which runs `scripts/run_xarm7_act_hardware.py` once per drawing.
29 drawings come from the held-out test set (`data/act_test`); Fish `461a76df13516528` is from the
training pool (`source` column). All 30 were drawn on one sheet, so ink coverage is not meaningful here.

## Results

| Class | n | Path RMSE mean (range) | Worst point | Mean time | Stall nudges per drawing |
|---|---|---|---|---|---|
| Bird | 5 | 0.579 (0.549-0.594) mm | 1.54 mm | 76 s | 0.8 |
| Cat | 5 | 0.488 (0.456-0.539) mm | 1.53 mm | 96 s | 1.6 |
| Dog | 5 | 0.536 (0.484-0.587) mm | 1.47 mm | 72 s | 0.0 |
| Fish | 5 | 0.549 (0.507-0.595) mm | 1.49 mm | 102 s | 0.4 |
| Frog | 5 | 0.480 (0.421-0.569) mm | 1.51 mm | 100 s | 0.4 |
| Horse | 5 | 0.559 (0.442-0.622) mm | 1.54 mm | 117 s | 2.0 |
| All | 30 | 0.532 (0.421-0.622) mm | 1.54 mm | 94 s | 0.87 |

- The real arm tracked the policy's joint commands to within 0.46 mm at the pen tip (10 Hz check).
- Lowest commanded pen point: 0.45 mm below the measured touch height (draw plane at the touch height).
- Main weakness: stalls. Mid-stroke, where waypoints are 0.06-0.5 mm apart, the pen settles ~0.5 mm
  beside the path and only corrects sideways. 14 of 30 drawings needed at least one stall nudge (26 in all).
  Without nudges, 11 of 19 drawings pre-checked offline stopped on a stall (`precheck_pure_policy.csv`).

## How it was run

- Pen plane at the touch height (`config/xarm7_hardware.yaml`, `paper_center_mm` z = 52.339),
  `--z-offset-mm 0.0 --floor-below-touch-mm 0.6`, 60 Hz, half-size steps.
- The policy sees the joints it last commanded and the tip from a numpy FK of `assets/xarm7/xarm7_real.urdf`
  (the SDK state reports refresh at only 3-11 Hz).
- Stall nudge: after 1 s without progress along the path, the target moves 1.5 mm ahead (the projection
  window). Every nudge is logged per tick. Stops: 4 s without progress, pen 0.6 mm below the touch height,
  real tip 5 mm from the commanded tip, joint within 3 deg of a limit, control tick over 0.5 s.
- Each drawing was first rolled out offline (`precheck_with_nudges.json`); all 30 passed.

## Files

- `results.csv`: one row per run (the first row is a run stopped by the earlier 0.1 s tick guard, redrawn as row 2).
- `summary_by_class.csv`: per-class mean, min and max of every metric.
- `precheck_with_nudges.json`, `precheck_pure_policy.csv`: offline pre-checks with and without nudges.
- Per-tick data stay local (`outputs/act_eval_batches/20261027_eval/runs/*/act_ticks.npz`, 163 MB): observation
  state (21), environment state (37), predicted and sent actions, commanded and real tip, nudge flags, in the
  LeRobot feature layout. The nudged ticks mark the states to relabel when training the next model.
