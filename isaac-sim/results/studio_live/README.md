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

## Sketch-style portrait and short hops (2026-10-27)

Pen-up travel was most of a portrait's time: on the run above only 29% of the samples were pen-down. With
`drawing.short_hop_m: 0.03`, a stroke that starts within 30 mm of the last lift is reached at pen-up height instead
of climbing to approach height and back. Pen heights, gates and calibration are unchanged. Same ACT checkpoint, same
scene.

| Drawing | Strokes | Short hops | Finished | Path RMSE | p95 error | Max error | Sim time |
|---|---|---|---|---|---|---|---|
| Portrait (above, mesh style) | 21 | no | yes | 0.241 mm | 0.403 mm | 1.112 mm | 143.5 s |
| Same portrait | 21 | yes | yes | 0.236 mm | 0.404 mm | 1.063 mm | 105.1 s |
| Same face, sketch style with hair strands | 21 | yes | yes | 0.242 mm | 0.397 mm | 1.261 mm | 110.8 s |

Short hops cut the same drawing's time by 27% with no change in tracking. The sketch style draws more ink (2,903
pen-down samples against 2,467) and still finishes 23% sooner than the old run.

**Files in this folder:**
- `*_comparison.png`: the desired path against the executed pen path, top-down, with the paper placed as in the run's scene config.
- `*_run_metrics.json`: the run reports.
- `portrait_trajectory.json`: the face trajectory that was drawn; `portrait_sketch_trajectory.json`: the sketch-style one.
- `portrait_hops_act_*`, `portrait_sketch_act_*`: the short-hop runs.

The videos (WideFront and PaperTop cameras) are under `isaac-sim/outputs/videos/studio_*`. They are not committed.
