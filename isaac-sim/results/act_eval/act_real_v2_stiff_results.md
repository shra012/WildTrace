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
