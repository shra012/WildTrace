# Drawing speed sweep (#16, lab-arm twin, 1 g)

How fast can a policy draw before accuracy drops? `action_scale` is the share of each predicted joint step that
the MCP `draw_trajectory` runner executes per 60 Hz tick (0.5 is the default and the teacher's pace; larger draws
faster). One drawing, Cat `323bd30d` (`data/act_test/Cat/323bd30d42ee2cf3.json`), no hybrid assist, no video.

- Scene: `config/xarm7_drawing_real.yaml`, config hash `4926204b8a89` (the same robot and placement as `61a50cf4c384`
  in `results/smolvla/`; only `vla_camera.height_above_paper_m` changed, 0.60 to 0.835).
- ACT: `models/act_real_kinematic_v2`, started by the MCP runner.
- SmolVLA: `models/smolvla_stage2`, zero noise, re-planning every action, 3 flow steps, served through a CUDA graph
  (`act_policy_server.py --policy smolvla --replan-every 1 --noise zero --flow-steps 3 --cuda-graph`).
- Date: 2026-10-31, code at `6527265`.

Path RMSE / p95 / max is the desired-to-executed nearest-path error in mm.

| Controller | Checkpoint | action_scale | Finished | Path RMSE / p95 / max (mm) | Sim time s |
|---|---|---|---|---|---|
| ACT | `act_real_kinematic_v2` | 0.5 (default) | yes | **0.206 / 0.338** / 0.990 | 48.7 |
| ACT | `act_real_kinematic_v2` | 0.75 | yes | 0.219 / 0.356 / 0.905 | 37.1 |
| ACT | `act_real_kinematic_v2` | 1.0 | yes | 0.228 / 0.388 / **0.871** | 30.3 |
| ACT | `act_real_kinematic_v2` | 1.5 | yes | 0.318 / 0.552 / 0.965 | 22.0 |
| ACT | `act_real_kinematic_v2` | 2.0 | no: ACT tip error 5.21 mm in DRAW_STROKE_0 exceeds the safety limit | 18.614 / 35.610 / 44.036 (aborted) | 5.4 |
| SmolVLA stage 2, CUDA graph | `smolvla_stage2` | 0.5 (default) | yes | **0.252 / 0.465** / 1.180 | 49.8 |
| SmolVLA stage 2, CUDA graph | `smolvla_stage2` | 1.0 | yes | 0.300 / 0.521 / **0.931** | 29.8 |
| SmolVLA stage 2, CUDA graph | `smolvla_stage2` | 1.5 | yes | 0.514 / 0.850 / 1.578 | 21.3 |

ACT at 0.75 is the fastest scale that keeps path RMSE within 10% of the default (+6%, 24% less sim time); 1.0 is
just past it (+11%, 38% less sim time), 1.5 costs +54%, and 2.0 overshoots the first stroke and trips the safety
stop after 5.4 s. SmolVLA has no faster scale within 10%: 1.0 is +19% and 1.5 doubles the error, though both still
finish.

**Files:** `cat_323bd30d_{act,smolvla_stage2_graph}_scale<value>_{comparison.png,run_metrics.json}`.
