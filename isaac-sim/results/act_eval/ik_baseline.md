# ACT closed-loop evaluation (gold test set, 1 g)

| Controller | Finished | RMSE mm | p95 mm | max mm | Heading RMSE deg | Corner RMSE mm | Jerk RMS rad/s3 | Draw time s |
|---|---|---|---|---|---|---|---|---|
| ik | 39/48 | 0.698 | 0.883 | 0.985 | 10.025 | 0.813 | 806.218 | 93.814 |

## Paired against `ik` (drawings both finished)

| Controller | Pairs | Mean RMSE delta mm | Lower RMSE |
|---|---|---|---|

## Per category (finished / RMSE mm)

| Category | ik |
|---|---|
| Bird | 9/9 · 0.696 |
| Cat | 10/12 · 0.684 |
| Dog | 11/13 · 0.702 |
| Fish | 0/4 · - |
| Frog | 5/5 · 0.725 |
| Horse | 4/5 · 0.692 |

## Failures

**ik**
- 1163bceaf6134cf5: Waypoint timeout in DRAW_STROKE_0 at index 1005
- d190d0c5f8107942: Waypoint timeout in DRAW_STROKE_0 at index 976
- 96ab6db6d9116afe: Waypoint timeout in DRAW_STROKE_0 at index 877
- 98980e7442f4a93c: Waypoint timeout in DRAW_STROKE_0 at index 1949
- a7cbe5e2b0f491bb: Waypoint timeout in DRAW_STROKE_0 at index 540
- ba3e1657db195fe4: Waypoint timeout in DRAW_STROKE_0 at index 548
- ce7b60c8b362b074: Waypoint timeout in DRAW_STROKE_0 at index 552
- dd305a2886cbf23c: Waypoint timeout in DRAW_STROKE_0 at index 595
- 7c54b4c75832b3b9: Waypoint timeout in DRAW_STROKE_0 at index 1034
