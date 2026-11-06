# Paired real-arm runs, ACT vs IK (issue #52), 2026-11-05

Continues `../20261104/` (runs 01-02). Physical xArm 7 (192.168.1.241), ACT checkpoint `act_real_kinematic_v2`,
IK = `run_xarm7_hardware.py --execute` (the controller's `set_position` per waypoint, stopping at each one).
Drawings from `isaac-sim/data/act_test/<Class>/<id>.json`, the 12 listed in #52.

**Coverage.** 9 of the 12 pairs are drawn: both birds, both cats, both dogs, and one each of fish, frog and horse
(the user cut the batch to one drawing per class to finish in an hour). Still to run: Fish `dd305a2886cbf23c`
(15, 16), Frog `8ae3adc42375beba` (19, 20), Horse `e659138f70574103` (23, 24).

## What changed during the day

- **Runs 03-05: marker B**, flat touch height 50.68 mm (no plane), the setup the issue asks for.
- **Run 05: marker B's tip broke.** The replacement marker (blue, round fresh tip) needed a new height. A touch taken
  109 mm off centre plus yesterday's corner fit put the centre too high (05b, no ink) and the corner-fit tilt was
  too steep (05c to 05f). The old tip had worn flat and slanted, which hid small height errors; the fresh tip only
  inks under pressure.
- **From run 06 both controllers follow the same tilted surface** (`--paper-plane`). The IK runner got the same
  opt-in option as ACT for this (local change to `run_xarm7_hardware.py`, not committed yet): every waypoint height
  and the fence follow the surface. Without the flag the runner is unchanged.
- **Run 08 onward:** surface lowered 0.2 mm for both (user). **Run 10 onward:** `measured_m04`, a plane through two
  camera-checked marker contact points (x 378 and x 538), 0.4 mm past contact. It replaced the 2026-11-04 corner-fit
  slope, which was about 1.5 times too steep for this sheet stack (0.0200 vs 0.0129 mm per mm along x).
- **Runs 13-22 share one sheet** (user): ACT and IK fish, frog and horse drawn on top of each other. The arm metrics
  are per run; ink can't be scored per run on that sheet.
- Every surface JSON and the three contact searches are in `surfaces/`.

## Runs

Path RMSE: ACT is the real tip (SDK reports, about 10 Hz) against the planned path while moving. IK is read only
after it has stopped at each waypoint, so its value is about zero by construction and not comparable; the ink is
the like-for-like comparison. "Paired" marks the run used for the ACT/IK pair; extra passes and failed attempts are
listed with their reason. No manual nudges in any run.

| NN | Drawing | Controller | Finished | Path RMSE (mm) | p95 | Max | Time (s) | Stall releases | Paired | Surface and ink |
|---|---|---|---|---|---|---|---|---|---|---|
| 03 | Bird `c3b05340ef29131f` | ACT | yes | 0.428 | 0.837 | 1.016 | 100.8 | 3 | yes | marker B, flat 50.68; clean line |
| 04 | Bird `c3b05340ef29131f` | IK | yes | 0.000 (settled) | 0.000 | 0.001 | 507.4 | - | yes | marker B, flat 50.68; wing filled solid, eye dot lost |
| 05 | Cat `1163bceaf6134cf5` | ACT | yes | 0.478 | 1.013 | 1.286 | 99.1 | 5 | no | marker B tip broke during the run |
| 05b | Cat `1163bceaf6134cf5` | ACT | no | - | - | - | - | - | no | new marker, flat 54.46: no ink, stopped by the user |
| 05c | Cat `1163bceaf6134cf5` | ACT | no | - | - | - | - | - | no | `tilt_m05`: no ink on the low-x side, stopped by the user |
| 05d | Cat `1163bceaf6134cf5` | ACT | yes | 0.484 | 0.987 | 1.354 | 99.3 | 5 | no | `tilt_m08`: patchy ink |
| 05e | Cat `1163bceaf6134cf5` | ACT | yes | 0.480 | 0.988 | 1.407 | 100.0 | 5 | no | `tilt_m11`, second pass over 05d |
| 05f | Cat `1163bceaf6134cf5` | ACT | yes | 0.473 | 0.932 | 1.294 | 99.8 | 5 | yes | `tilt_m13`, clean sheet; one gap on the high-y edge |
| 06 | Cat `1163bceaf6134cf5` | IK | yes | 0.000 (settled) | 0.000 | 0.000 | 488.3 | - | yes | `tilt_m13`; complete |
| 07 | Cat `70bc30f8a5f918eb` | ACT | yes | 0.456 | 0.909 | 1.215 | 98.7 | 2 | yes | `tilt_m13`; high-y side mostly missing |
| 08 | Cat `70bc30f8a5f918eb` | IK | yes | 0.000 (settled) | 0.000 | 0.001 | 556.3 | - | yes | `tilt_m15` (0.2 mm lower than 07); complete, eyes and nose blobbed |
| 09 | Dog `4cf9d2757b8a6282` | ACT | no | - | - | - | - | - | no | stopped by the user: the run 08 sheet was still in place |
| 09b | Dog `4cf9d2757b8a6282` | ACT | yes | 0.490 | 0.911 | 1.292 | 64.8 | 0 | yes | `tilt_m15`; head, belly and rump only |
| 09c | Dog `4cf9d2757b8a6282` | ACT | yes | 0.474 | 0.821 | 1.071 | 65.2 | 0 | no | `tilt_m18`, second pass over 09b |
| 09d | Dog `4cf9d2757b8a6282` | ACT | yes | 0.462 | 0.854 | 1.049 | 65.3 | 0 | no | `measured_m04`, third pass; back still missing |
| 09e | Dog `4cf9d2757b8a6282` | ACT | yes | 0.426 | 0.702 | 0.767 | 18.2 | 0 | no | `measured_m08`, back segment only (`09e_dog4cf9_back_segment_trajectory.json`); sheet complete |
| 10 | Dog `4cf9d2757b8a6282` | IK | yes | 0.000 (settled) | 0.000 | 0.001 | 238.8 | - | yes | `measured_m04`; complete |
| 11 | Dog `cd5042b9731161f9` | ACT | yes | 0.518 | 0.933 | 1.127 | 108.2 | 0 | yes | `measured_m04`; bottom line broken |
| 12 | Dog `cd5042b9731161f9` | IK | yes | 0.000 (settled) | 0.000 | 0.001 | 498.1 | - | yes | `measured_m04`; lower body missing (counted as drawn, user) |
| 13 | Fish `ba3e1657db195fe4` | ACT | yes | 0.482 | 0.922 | 1.151 | 74.1 | 0 | yes | `measured_m04`; clean sheet; only a few short pieces inked |
| 14 | Fish `ba3e1657db195fe4` | IK | yes | 0.000 (settled) | 0.000 | 0.001 | 381.4 | - | yes | `measured_m04`; over the run 13 sheet; complete |
| 17 | Frog `6bbd7fa65eede66e` | ACT | yes | 0.393 | 0.678 | 1.167 | 120.8 | 2 | yes | `measured_m04`; shared sheet (13, 14); partial ink |
| 18 | Frog `6bbd7fa65eede66e` | IK | yes | 0.000 (settled) | 0.000 | 0.001 | 681.8 | - | yes | `measured_m04`; shared sheet; complete |
| 21 | Horse `a84882576126b982` | ACT | yes | 0.514 | 0.967 | 1.210 | 84.6 | 10 | yes | `measured_m04`; shared sheet; mostly inked |
| 22 | Horse `a84882576126b982` | IK | yes | 0.000 (settled) | 0.000 | 0.001 | 376.6 | - | yes | `measured_m04`; shared sheet; complete |

## Summary over the paired runs (01-22, 9 pairs)

| | ACT | IK |
|---|---|---|
| Finished | 9/9 | 9/9 |
| Path RMSE, real tip while moving | 0.458 mm mean (0.365 to 0.518) | not comparable (read at rest) |
| Time per drawing | 91.4 s mean | 446.2 s mean (4.9x slower) |
| Stall releases / nudges | 22 automatic / 0 | 0 retries / 0 |
| Ink from the wrist photos | 2 complete (01, 03, both marker B), 7 with gaps | 8 complete (04 with blobbed small features), 1 with a gap (12) |

**Findings**
- On ink, IK beats ACT on the real arm with a pressure-sensitive marker: it settles on the planned height at every
  waypoint, while ACT's commanded pen height drifts by about +-0.15 mm with position on the sheet (higher on the
  high-y side), and a fresh round marker tip loses contact at +0.1 mm. With marker B (worn flat) ACT inked fully.
- ACT draws 4.9x faster and keeps thin lines; IK pools ink where waypoints are dense (eyes, noses, wing).
- The paper surface matters at the 0.1 mm level for both: a single touch plus a corner-fit tilt was not good enough
  for a fresh marker; two camera-checked contact points were.

## Files

Per run: `<NN>_<class>_<id>_<act|ik>_summary.json`, `_events.jsonl`, `_robot_report.jsonl` (ACT), `_photo.jpg`
(wrist camera, park pose, about 0.6 mm per pixel). `paired_runs_all.json` has every row including 01-02.
600 dpi scans and `ink_fidelity.py` scores are still to do (the photos are too coarse for chamfer at 0.5 mm).
