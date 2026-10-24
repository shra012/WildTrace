# ACT closed-loop evaluation (gold test set, 1 g)

| Controller | Finished | RMSE mm | p95 mm | max mm | Heading RMSE deg | Corner RMSE mm | Jerk RMS rad/s3 | Draw time s |
|---|---|---|---|---|---|---|---|---|
| ik | 39/48 | 0.698 | 0.883 | 0.985 | 10.025 | 0.813 | 806.218 | 93.814 |
| act_kin_v2 | 48/48 | 0.239 | 0.413 | 1.472 | 9.162 | 0.289 | 254.751 | 66.204 |
| act_kin_v4 | 48/48 | 0.256 | 0.443 | 0.885 | 9.704 | 0.396 | 285.462 | 75.814 |

## Paired against `ik` (drawings both finished)

| Controller | Pairs | Mean RMSE delta mm | Lower RMSE |
|---|---|---|---|
| act_kin_v2 | 39 | -0.461 | 39/39 |
| act_kin_v4 | 39 | -0.439 | 39/39 |

## Per category (finished / RMSE mm)

| Category | ik | act_kin_v2 | act_kin_v4 |
|---|---|---|---|
| Bird | 9/9 · 0.696 | 9/9 · 0.236 | 9/9 · 0.255 |
| Cat | 10/12 · 0.684 | 12/12 · 0.252 | 12/12 · 0.257 |
| Dog | 11/13 · 0.702 | 13/13 · 0.227 | 13/13 · 0.262 |
| Fish | 0/4 · - | 4/4 · 0.269 | 4/4 · 0.236 |
| Frog | 5/5 · 0.725 | 5/5 · 0.229 | 5/5 · 0.253 |
| Horse | 4/5 · 0.692 | 5/5 · 0.231 | 5/5 · 0.258 |

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
