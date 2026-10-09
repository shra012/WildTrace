# ACT closed-loop evaluation (gold test set, 1 g)

| Controller | Finished | RMSE mm | p95 mm | max mm | Heading RMSE deg | Corner RMSE mm | Jerk RMS rad/s3 | Draw time s |
|---|---|---|---|---|---|---|---|---|
| ik | 39/48 | 0.698 | 0.883 | 0.985 | 10.025 | 0.813 | 806.218 | 93.814 |
| act_kin | 48/48 | 0.366 | 0.567 | 2.599 | 13.020 | 0.307 | 1018.235 | 66.910 |
| act_kin_60hz | 48/48 | 0.325 | 0.483 | 2.360 | 12.244 | 0.292 | 272.189 | 67.599 |

## Paired against `ik` (drawings both finished)

| Controller | Pairs | Mean RMSE delta mm | Lower RMSE |
|---|---|---|---|
| act_kin | 39 | -0.327 | 38/39 |
| act_kin_60hz | 39 | -0.371 | 38/39 |

## Per category (finished / RMSE mm)

| Category | ik | act_kin | act_kin_60hz |
|---|---|---|---|
| Bird | 9/9 · 0.696 | 9/9 · 0.371 | 9/9 · 0.318 |
| Cat | 10/12 · 0.684 | 12/12 · 0.403 | 12/12 · 0.337 |
| Dog | 11/13 · 0.702 | 13/13 · 0.293 | 13/13 · 0.259 |
| Fish | 0/4 · - | 4/4 · 0.389 | 4/4 · 0.376 |
| Frog | 5/5 · 0.725 | 5/5 · 0.470 | 5/5 · 0.476 |
| Horse | 4/5 · 0.692 | 5/5 · 0.336 | 5/5 · 0.291 |

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
