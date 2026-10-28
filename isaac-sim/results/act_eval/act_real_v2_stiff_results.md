# ACT closed-loop evaluation (gold test set, 1 g)

| Controller | Finished | RMSE mm | p95 mm | max mm | Heading RMSE deg | Corner RMSE mm | Jerk RMS rad/s3 | Draw time s |
|---|---|---|---|---|---|---|---|---|
| ik_real_stiff | 44/48 | 0.283 | 0.406 | 0.775 | 6.354 | 0.373 | 752.060 | 81.506 |
| act_real_v2_stiff | 48/48 | 0.195 | 0.339 | 0.812 | 8.656 | 0.328 | 389.757 | 79.286 |

## Paired against `ik_real_stiff` (drawings both finished)

| Controller | Pairs | Mean RMSE delta mm | Lower RMSE |
|---|---|---|---|
| act_real_v2_stiff | 44 | -0.089 | 44/44 |

## Per category (finished / RMSE mm)

| Category | ik_real_stiff | act_real_v2_stiff |
|---|---|---|
| Bird | 8/9 · 0.273 | 9/9 · 0.192 |
| Cat | 12/12 · 0.278 | 12/12 · 0.197 |
| Dog | 12/13 · 0.286 | 13/13 · 0.194 |
| Fish | 4/4 · 0.300 | 4/4 · 0.198 |
| Frog | 4/5 · 0.294 | 5/5 · 0.187 |
| Horse | 4/5 · 0.282 | 5/5 · 0.203 |

## Failures

**ik_real_stiff**
- c3a528aafbfd7b55: Waypoint timeout in DRAW_STROKE_0 at index 83
- 9c942e2bb316cdb6: Waypoint timeout in DRAW_STROKE_0 at index 109
- aef94e7551606d10: Waypoint timeout in LIFT_PEN_STROKE_3 at index 2
- 165c17090deb8a48: Waypoint timeout in LIFT_PEN_STROKE_2 at index 2

## Real arm (physical xArm 7, sim to real)

Batch run on the physical lab arm with the ACT policy, 2026-10-27: 30 drawings, 5 per class. Every
drawing finished, with no safety stops after the first restart. The run logs are not in the repo yet.

| Class | Drawings | RMSE mean | RMSE range | Worst point | Mean time | Nudges per drawing |
|---|---|---|---|---|---|---|
| Bird | 5 | 0.579 mm | 0.549 to 0.594 mm | 1.54 mm | 76 s | 0.8 |
| Cat | 5 | 0.488 mm | 0.456 to 0.539 mm | 1.53 mm | 96 s | 1.6 |
| Dog | 5 | 0.536 mm | 0.484 to 0.587 mm | 1.47 mm | 72 s | 0.0 |
| Fish | 5 | 0.549 mm | 0.507 to 0.595 mm | 1.49 mm | 102 s | 0.4 |
| Frog | 5 | 0.480 mm | 0.421 to 0.569 mm | 1.51 mm | 100 s | 0.4 |
| Horse | 5 | 0.559 mm | 0.442 to 0.622 mm | 1.54 mm | 117 s | 2.0 |
| **All** | **30** | **0.532 mm** | 0.421 to 0.622 mm | 1.54 mm | 94 s | 0.87 |

**Twin against real arm (ACT, mean path RMSE)**

| Class | Twin (gold set above) | Real arm | Real / twin |
|---|---|---|---|
| Bird | 0.192 mm | 0.579 mm | 3.0x |
| Cat | 0.197 mm | 0.488 mm | 2.5x |
| Dog | 0.194 mm | 0.536 mm | 2.8x |
| Fish | 0.198 mm | 0.549 mm | 2.8x |
| Frog | 0.187 mm | 0.480 mm | 2.6x |
| Horse | 0.203 mm | 0.559 mm | 2.8x |
| **All** | **0.195 mm** | **0.532 mm** | **2.7x** |

The real-arm drawings are not the same 48 as the twin's gold set, so the per-class comparison is indicative, not
paired.

**Findings**
- Accuracy: path RMSE is about 0.53 mm overall, 2.7 times the twin's. Frogs and cats came out best, birds worst.
- Stalls are the policy's main weakness on the real arm. 14 of 30 drawings needed at least one nudge, 26 nudges in
  total, and horses needed the most. Without nudging, roughly half the birds and most of the cats stalled in the
  pre-check.
- The stalls happen mid-stroke where the waypoints are packed tightly: the pen sits about 0.5 mm beside the path
  and only corrects sideways, so it never reaches the waypoint and the drawing does not advance.
