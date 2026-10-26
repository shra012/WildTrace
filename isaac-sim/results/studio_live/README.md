# Live runs from WildTrace Studio (lab-arm twin, 1 g)

These drawings were started from the Studio web app (`app/`) and run through the wildtrace-sim MCP server
(`draw_trajectory`) in Isaac Sim 6.0.1, using the lab-arm twin scene (`config/xarm7_drawing_real.yaml`) on 2026-10-26.

ACT uses the `models/act_real_kinematic_v2` checkpoint. IK is Lula IK.

| Drawing | Strokes | Controller | Finished | Path RMSE | p95 error | Max error | Sim time |
|---|---|---|---|---|---|---|---|
| Cat `1163bceaf6134cf5` (gold) | 4 | ACT | yes | 0.213 mm | 0.356 mm | 1.136 mm | 55.9 s |
| Cat `1163bceaf6134cf5` (gold) | 4 | IK | yes | 0.310 mm | 0.513 mm | 0.793 mm | 106.2 s |
| Portrait (group photo, face 2 of 3) | 21 | ACT | yes | 0.241 mm | 0.403 mm | 1.112 mm | 143.5 s |
| Portrait (group photo, face 2 of 3) | 21 | IK | yes | 0.408 mm | 0.675 mm | 0.828 mm | 286.3 s |

The portrait is a new kind of subject for both controllers: ACT was trained only on animal outlines. On it, ACT keeps
its lead over IK, with 41% lower RMSE in half the time.

The IK plot shows a visible tremor along every stroke, from stopping and settling at each waypoint. IK has a lower
worst-case error, but ACT's p95 error is still lower than IK's, so its maximum comes from a few short excursions
rather than from the whole path.

**Files in this folder:**
- `*_comparison.png`: the desired path against the executed pen path, top-down, with the paper placed as in the run's scene config.
- `*_run_metrics.json`: the run reports.
- `portrait_trajectory.json`: the face trajectory that was drawn.

The videos (WideFront and PaperTop cameras) are under `isaac-sim/outputs/videos/studio_*`. They are not committed.
