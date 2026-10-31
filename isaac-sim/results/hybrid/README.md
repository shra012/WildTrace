# Hybrid policy control (#46, lab-arm twin, 1 g)

The hybrid (`src/policy_assist.py`, `draw_trajectory(policy_assist=true)`) lets a learned policy draw and hands
control to Lula IK in two cases, both found from the run's own ink bookkeeping, not from the policy:

- **stall:** the drawing target has not advanced for 1.5 s; IK drives the next 5 mm of path and hands back.
- **gap:** a stretch of path behind the pen has no ink (the kinematic labeller's coverage rule); IK goes back along
  the line past the gap and forward again to where the pen left off, then the policy continues.

The **skip test** (`skip_test_spacing_m=0.06`) makes the pen fail to ink on purpose: 3 to 8 mm skips placed along
the inked travel distance, 60 mm apart on average (exponential), seed 0 (`act_features.SkipSchedule`). It checks
that the hybrid finds and redraws gaps. "Uncovered" is the share of the path still without ink at the end.

- Scene: `config/xarm7_drawing_real.yaml`, config hash `4926204b8a89` (the same robot and placement as `61a50cf4c384`
  in `results/smolvla/`; only `vla_camera.height_above_paper_m` changed, 0.60 to 0.835).
- ACT: `models/act_real_kinematic_v2`, default `action_scale` (0.5), started by the MCP runner.
- SmolVLA: `models/smolvla_stage2`, zero noise, re-planning every action, 3 flow steps, CUDA graph.
- Runner: MCP `draw_trajectory`, no video. Date: 2026-10-31, code at `6527265`.
- Drawings: the six gold animals from `data/act_test/`, five face sketches from the validation split of
  `wildtrace_real_kin_gaps_faces_v2` and the Studio portrait sketch (`studio_live/portrait_sketch_trajectory.json`),
  the same set as `results/smolvla/README.md`.
- Plain ACT baseline: the ACT columns of `results/smolvla/README.md` (same runner and twin, not rerun). The animal
  baselines date from 2026-10-28 (config `4dc272edd730`); plain ACT on the cat today gives 0.206 / 0.338 / 0.990 mm
  in 48.7 s (`results/speed/`), so most of the 2 to 7 s sim-time gap on the animals comes from the
  runner, not the hybrid (2.7 of 3.1 s on the cat).

Errors are path RMSE / p95 / max in mm (desired-to-executed nearest-path error).

## (a) Hybrid ACT

Animals:

| Drawing | Finished | Hybrid ACT | Plain ACT | Sim time s (Hybrid / plain ACT) | Stall take-overs | Redraws (given up) | Skips | Uncovered % |
|---|---|---|---|---|---|---|---|---|
| Cat `323bd30d` | yes | 0.211 / 0.348 / 1.019 | 0.209 / 0.355 / 1.019 | 49.1 / 46.0 | 0 | 0 (0) | 0 | 0.00 |
| Bird `1a5733d2` | yes | 0.208 / 0.343 / 0.641 | 0.207 / 0.356 / 0.641 | 43.3 / 40.1 | 0 | 0 (0) | 0 | 0.00 |
| Dog `07379e95` | yes | 0.212 / 0.331 / 0.731 | 0.205 / 0.362 / 0.736 | 41.3 / 38.7 | 0 | 0 (0) | 0 | 0.00 |
| Fish `a7cbe5e2` | yes | 0.203 / 0.316 / 1.186 | 0.204 / 0.312 / 1.186 | 90.2 / 83.5 | 0 | 0 (0) | 0 | 0.06 |
| Frog `3f221bad` | yes | 0.197 / 0.311 / 0.782 | 0.197 / 0.313 / 0.790 | 35.5 / 33.5 | 0 | 0 (0) | 0 | 0.00 |
| Horse `165c1709` | yes | 0.225 / 0.368 / 0.781 | 0.214 / 0.380 / 0.781 | 48.5 / 45.0 | 0 | 0 (0) | 0 | 0.00 |
| **Mean** | 6/6 | 0.210 / 0.336 / 0.857 | 0.206 / 0.346 / 0.859 | 51.3 / 47.8 | 0 total | 0 (0) total | 0 total | 0.01 |

Faces:

| Drawing | Finished | Hybrid ACT | Plain ACT | Sim time s (Hybrid / plain ACT) | Stall take-overs | Redraws (given up) | Skips | Uncovered % |
|---|---|---|---|---|---|---|---|---|
| Face `04369443` | yes | 0.267 / 0.455 / 1.173 | 0.267 / 0.455 / 1.173 | 90.2 / 90.2 | 0 | 0 (0) | 0 | 0.03 |
| Face `1710ea61` | yes | 0.246 / 0.400 / 0.988 | 0.246 / 0.400 / 0.988 | 78.5 / 78.5 | 0 | 0 (0) | 0 | 0.00 |
| Face `26870728` | yes | 0.261 / 0.395 / 1.177 | 0.261 / 0.395 / 1.177 | 92.6 / 92.6 | 0 | 0 (0) | 0 | 0.03 |
| Face `3572f3a5` | yes | 0.247 / 0.411 / 1.237 | 0.247 / 0.411 / 1.237 | 123.0 / 123.0 | 0 | 0 (0) | 0 | 0.05 |
| Face `4f1c49dc` | yes | 0.250 / 0.408 / 1.048 | 0.250 / 0.408 / 1.048 | 104.1 / 104.1 | 0 | 0 (0) | 0 | 0.05 |
| Portrait sketch (Studio) | yes | 0.250 / 0.394 / 1.285 | 0.242 / 0.397 / 1.261 | 122.0 / 110.8 | 0 | 0 (0) | 0 | 0.04 |
| **Mean** | 6/6 | 0.253 / 0.410 / 1.152 | 0.252 / 0.411 / 1.147 | 101.7 / 99.9 | 0 total | 0 (0) total | 0 total | 0.03 |

## (b) Hybrid ACT, skip test (60 mm)

Animals:

| Drawing | Finished | Hybrid ACT | Plain ACT | Sim time s (Hybrid / plain ACT) | Stall take-overs | Redraws (given up) | Skips | Uncovered % |
|---|---|---|---|---|---|---|---|---|
| Cat `323bd30d` | yes | 0.212 / 0.357 / 1.021 | 0.209 / 0.355 / 1.019 | 49.5 / 46.0 | 0 | 4 (0) | 6 | 0.19 |
| Bird `1a5733d2` | yes | 0.211 / 0.346 / 0.641 | 0.207 / 0.356 / 0.641 | 43.8 / 40.1 | 0 | 4 (0) | 6 | 0.25 |
| Dog `07379e95` | yes | 0.215 / 0.332 / 0.731 | 0.205 / 0.362 / 0.736 | 42.0 / 38.7 | 0 | 5 (0) | 7 | 0.35 |
| Fish `a7cbe5e2` | yes | 0.203 / 0.331 / 1.183 | 0.204 / 0.312 / 1.186 | 91.3 / 83.5 | 0 | 7 (0) | 8 | 0.18 |
| Frog `3f221bad` | yes | 0.202 / 0.321 / 0.782 | 0.197 / 0.313 / 0.790 | 36.3 / 33.5 | 0 | 5 (0) | 6 | 0.08 |
| Horse `165c1709` | yes | 0.226 / 0.375 / 0.781 | 0.214 / 0.380 / 0.781 | 49.3 / 45.0 | 0 | 5 (0) | 7 | 0.26 |
| **Mean** | 6/6 | 0.212 / 0.344 / 0.856 | 0.206 / 0.346 / 0.859 | 52.0 / 47.8 | 0 total | 30 (0) total | 40 total | 0.22 |

Faces:

| Drawing | Finished | Hybrid ACT | Plain ACT | Sim time s (Hybrid / plain ACT) | Stall take-overs | Redraws (given up) | Skips | Uncovered % |
|---|---|---|---|---|---|---|---|---|
| Face `04369443` | yes | 0.267 / 0.450 / 1.035 | 0.267 / 0.455 / 1.173 | 91.1 / 90.2 | 0 | 7 (0) | 8 | 0.03 |
| Face `1710ea61` | yes | 0.247 / 0.405 / 0.986 | 0.246 / 0.400 / 0.988 | 79.3 / 78.5 | 0 | 7 (0) | 8 | 0.03 |
| Face `26870728` | yes | 0.262 / 0.399 / 1.184 | 0.261 / 0.395 / 1.177 | 93.6 / 92.6 | 0 | 7 (0) | 8 | 0.06 |
| Face `3572f3a5` | yes | 0.247 / 0.429 / 1.229 | 0.247 / 0.411 / 1.237 | 124.7 / 123.0 | 0 | 11 (0) | 12 | 0.48 |
| Face `4f1c49dc` | yes | 0.254 / 0.411 / 1.052 | 0.250 / 0.408 / 1.048 | 105.4 / 104.1 | 0 | 9 (0) | 12 | 0.49 |
| Portrait sketch (Studio) | yes | 0.261 / 0.431 / 1.293 | 0.242 / 0.397 / 1.261 | 124.2 / 110.8 | 0 | 16 (0) | 17 | 0.10 |
| **Mean** | 6/6 | 0.256 / 0.421 / 1.130 | 0.252 / 0.411 / 1.147 | 103.1 / 99.9 | 0 total | 57 (0) total | 65 total | 0.20 |

## (c) Hybrid SmolVLA stage 2 (CUDA graph), skip test (60 mm)

Same six animals; the baseline column is still plain ACT. Plain SmolVLA stage 2 with the same server settings drew
the cat at 0.252 / 0.465 / 1.180 mm in 49.8 s (`results/speed/`, scale 0.5).

| Drawing | Finished | Hybrid SmolVLA | Plain ACT | Sim time s (Hybrid / plain ACT) | Stall take-overs | Redraws (given up) | Skips | Uncovered % |
|---|---|---|---|---|---|---|---|---|
| Cat `323bd30d` | yes | 0.253 / 0.458 / 1.197 | 0.209 / 0.355 / 1.019 | 50.5 / 46.0 | 0 | 4 (0) | 6 | 0.66 |
| Bird `1a5733d2` | yes | 0.244 / 0.397 / 1.058 | 0.207 / 0.356 / 0.641 | 43.8 / 40.1 | 0 | 5 (0) | 6 | 0.05 |
| Dog `07379e95` | yes | 0.239 / 0.389 / 1.085 | 0.205 / 0.362 / 0.736 | 40.9 / 38.7 | 0 | 6 (0) | 7 | 0.12 |
| Fish `a7cbe5e2` | yes | 0.240 / 0.404 / 1.374 | 0.204 / 0.312 / 1.186 | 89.0 / 83.5 | 0 | 6 (0) | 8 | 0.40 |
| Frog `3f221bad` | yes | 0.234 / 0.380 / 0.890 | 0.197 / 0.313 / 0.790 | 35.0 / 33.5 | 0 | 5 (0) | 6 | 0.16 |
| Horse `165c1709` | yes | 0.253 / 0.425 / 0.748 | 0.214 / 0.380 / 0.781 | 49.8 / 45.0 | 0 | 6 (0) | 7 | 0.00 |
| **Mean** | 6/6 | 0.244 / 0.409 / 1.059 | 0.206 / 0.346 / 0.859 | 51.5 / 47.8 | 0 total | 32 (0) total | 40 total | 0.23 |

## Findings

All 30 runs finished, and IK never had to take over a stall (0 stall take-overs): ACT and SmolVLA stage 2 keep the
target moving on every drawing. Without injected skips the hybrid matches plain ACT (mean RMSE 0.210 vs 0.206 mm on
the animals, 0.253 vs 0.252 mm on the faces; the five face sketches, whose baselines were recorded 2026-10-30/31,
reproduce them exactly, error and sim time). With the skip test, IK made 87 redraws for 105 injected skips under
ACT and 32 for 40 under SmolVLA, none given up, leaving at most 0.49% (ACT) and 0.66% (SmolVLA) of a path uncovered.
For ACT that costs at most 0.011 mm RMSE and 0.4 to 2.2 s of sim time over the same hybrid without skips.

**Files:** `<drawing>_<id8>_act_hybrid[_skip60]_{comparison.png,run_metrics.json}`,
`<drawing>_<id8>_smolvla_stage2_graph_hybrid_skip60_{comparison.png,run_metrics.json}`; faces are `face_<id8>`,
the portrait is `portrait_sketch`.
