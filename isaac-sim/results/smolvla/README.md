# SmolVLA vs ACT (lab-arm twin, 1 g)

SmolVLA (`lerobot/smolvla_base`, 450M parameters) fine-tuned with `scripts/train_smolvla.py` on
`wildtrace_real_kin_v2`, the same LeRobot dataset as ACT's `act_real_kinematic_v2`. 20k steps, batch 32, bf16,
vision encoder and language model frozen (100M parameters train), about 3.5 h on the RTX 3080 Ti laptop GPU,
2026-10-27. Observation and serving: `src/vla_features.py`, `scripts/act_policy_server.py --policy smolvla`.

## Offline: same 1,024 held-out frames

`scripts/compare_policies_offline.py` scores each policy's first predicted joint step against the label.

| Policy | First-action error (1.0 = predicting zero) | Mean error | Median direction cosine |
|---|---|---|---|
| ACT | **0.28** | **0.42 mrad** | **0.99** |
| SmolVLA, 1 sample (as served) | 0.75 | 1.13 mrad | 0.82 |
| SmolVLA, mean of 4 samples | 0.62 | 0.93 mrad | 0.85 |

The label's mean step is 1.50 mrad. SmolVLA's validation error during training: 0.95 at 2k steps, 0.74 to 0.82
from 8k to 18k, 0.65 at 20k (`train_log.ndjson`, one sample on 64 frames per check).

## Closed loop: Cat `323bd30d` on the lab-arm twin

| Controller | Finished | Path RMSE | p95 error | Max error | Sim time |
|---|---|---|---|---|---|
| ACT | yes | **0.209 mm** | **0.355 mm** | **1.019 mm** | 46.0 s |
| SmolVLA (3 actions per sampled chunk) | yes | 0.501 mm | 0.816 mm | 1.206 mm | 42.1 s |

SmolVLA draws the whole cat and stays on the path, but every line wobbles: each chunk is sampled from fresh noise,
and consecutive chunks disagree by a fraction of a millimetre. ACT's lines are smooth. SmolVLA is also slow to
serve: about 0.6 s per chunk on this GPU, so the 42 s drawing took about 13 minutes of wall time.

**Files:** `cat_323bd30d_{smolvla,act}_comparison.png` (desired against executed pen path) and
`cat_323bd30d_{smolvla,act}_run_metrics.json`.

## Stage 2: full state, direct-state MLP, 60k more steps (2026-10-28)

`models/smolvla_stage2`: the full 58-value state (ACT state plus the 30-value path window) fed through a direct-state
MLP, 8 VLM layers, 256 px images, 20-action chunks, all three recordings of each drawing (19.8M frames), 60k steps
from stage 1 (local checkpoint, not in `models/`), batch 64, lr 1e-4 (51M of 323M parameters train), about 5.5 h on the same GPU.

Offline, same 1,024 held-out frames:

| Policy | First-action error | Mean error | Median direction cosine |
|---|---|---|---|
| ACT | **0.28** | **0.42 mrad** | **0.99** |
| SmolVLA stage 2, zero noise (as served) | 0.36 | 0.54 mrad | 0.97 |
| SmolVLA stage 2, mean of 4 samples | 0.40 | 0.61 mrad | 0.96 |

Zero starting noise beats averaging samples. About 220 ms per chunk with 10 flow-matching steps.

Closed loop on the lab-arm twin, one gold drawing per category, both controllers through the MCP `draw_trajectory`
runner in the same session (SmolVLA: zero noise, 3 actions per chunk). Path RMSE / p95 / max error in mm:

| Drawing | Finished | SmolVLA stage 2 | ACT | Sim time s (SmolVLA / ACT) |
|---|---|---|---|---|
| Cat `323bd30d` | both | 0.315 / 0.605 / 1.618 | **0.209 / 0.355 / 1.019** | 46.3 / 46.0 |
| Bird `1a5733d2` | both | 0.343 / 0.630 / 1.234 | **0.207 / 0.356 / 0.641** | 42.0 / 40.1 |
| Dog `07379e95` | both | 0.329 / 0.552 / 1.398 | **0.205 / 0.362 / 0.736** | 39.6 / 38.7 |
| Fish `a7cbe5e2` | both | 0.307 / 0.541 / 1.530 | **0.204 / 0.312 / 1.186** | 86.7 / 83.5 |
| Frog `3f221bad` | both | 0.337 / 0.551 / 1.108 | **0.197 / 0.313 / 0.790** | 33.1 / 33.5 |
| Horse `165c1709` | both | 0.333 / 0.607 / 1.313 | **0.214 / 0.380 / 0.781** | 48.2 / 45.0 |
| **Mean** | 6/6 | 0.327 / 0.581 / 1.367 | **0.206 / 0.346 / 0.859** | |

Stage 2 cuts the cat's path RMSE from 0.50 to 0.32 mm and the zero-noise chunks remove the wobble, but ACT is still
about 1.6 times more accurate on every drawing. SmolVLA's largest errors are at stroke ends (whisker and tail tips
cut short). Each SmolVLA drawing took 4 to 10 minutes of wall time, ACT 1.5 to 3 minutes.
The ACT numbers here come from the MCP runner and differ from the batch runner's (`act_eval/act_real_v2_stiff`,
which steps the policy at 60 Hz with half-size steps): compare within one runner only.

**Files:** `<drawing>_<id>_{smolvla_stage2,act}_{comparison.png,run_metrics.json}`.

## What could close the gap

- Sample each chunk from the same fixed noise, or average several samples, to remove the chunk-to-chunk wobble.
- Train longer: the last validation check (0.65 at 20k) was the best one, though the checks are noisy.
- Replace the rendered path image with a real camera view, which is what a VLA is built for, and is the case where
  ACT's hand-made path window is not available.

## Camera views of the paper (#46, 2026-10-29)

`camera_v1` (local checkpoint, not in `models/`): stage 2's model and full state, with the rendered path image
replaced by two synthetic camera views of the paper (`src/vla_features.py`, `train_smolvla.py --view camera`):

- **overhead:** a fixed top-down camera over the 176 mm drawing area, the same framing as the twin's `vla_camera`
  in `config/xarm7_drawing_real.yaml` (`camera/cat_323bd30d_overhead_isaac_vs_synthetic.png`);
- **tip crop:** the same paper zoomed to 45 mm around the pen tip, with the upcoming path drawn over it.

Both show the target traced so far in light blue under the ink drawn so far in black, rendered from the episode's
earlier frames in training and from the ticks since reset in the policy server (`camera/bird_training_views.png`).
40k steps from `smolvla_stage2`, batch 64, lr 1e-4 with warmup and cosine decay, about 5.5 h on the same GPU.

Offline, same 1,024 held-out frames:

| Policy | First-action error | Mean error | Median direction cosine | ms per chunk |
|---|---|---|---|---|
| ACT | **0.277** | **0.42 mrad** | **0.986** | |
| SmolVLA camera_v1, zero noise, 3 flow steps | 0.306 | 0.46 mrad | 0.981 | 107 |
| SmolVLA camera_v1, zero noise, 10 flow steps | 0.303 | 0.45 mrad | 0.982 | 241 |
| SmolVLA stage 2, zero noise, 3 flow steps | 0.366 | 0.55 mrad | 0.97 | 95 |

Does it use the camera? The same frames with the views changed (3 flow steps):

| Views | First-action error |
|---|---|
| True views | 0.306 |
| Empty page (no history, only the current tip and upcoming path) | 0.306 |
| Ink dropped (every segment drawn so far looks missed) | 0.307 |

The model ignores what is on the page. In the training data the IK teacher always follows the path, whatever was
inked before, so the history never changes the label and there is nothing to learn from it. The offline gain over
stage 2 comes from the 40k extra steps (a fresh learning-rate decay) or the upcoming path in the tip crop; a stage 2
run continued without the camera would separate the two.

Closed loop on the lab-arm twin, same six drawings and runner settings as stage 2 (zero noise, 3 actions per chunk,
10 flow steps). Path RMSE / p95 / max error in mm:

| Drawing | Finished | SmolVLA camera_v1 | SmolVLA stage 2 | ACT | Sim time s (camera_v1) |
|---|---|---|---|---|---|
| Cat `323bd30d` | yes | 0.315 / 0.584 / 1.502 | 0.315 / 0.605 / 1.618 | **0.209 / 0.355 / 1.019** | 46.7 |
| Bird `1a5733d2` | yes | 0.319 / 0.570 / 1.415 | 0.343 / 0.630 / 1.234 | **0.207 / 0.356 / 0.641** | 42.4 |
| Dog `07379e95` | yes | 0.312 / 0.595 / 1.513 | 0.329 / 0.552 / 1.398 | **0.205 / 0.362 / 0.736** | 39.9 |
| Fish `a7cbe5e2` | yes | 0.306 / 0.557 / 1.614 | 0.307 / 0.541 / 1.530 | **0.204 / 0.312 / 1.186** | 87.0 |
| Frog `3f221bad` | yes | 0.347 / 0.574 / 1.182 | 0.337 / 0.551 / 1.108 | **0.197 / 0.313 / 0.790** | 34.6 |
| Horse `165c1709` | yes | 0.348 / 0.647 / 1.276 | 0.333 / 0.607 / 1.313 | **0.214 / 0.380 / 0.781** | 48.4 |
| **Mean** | 6/6 | 0.324 / 0.588 / 1.417 | 0.327 / 0.581 / 1.367 | **0.206 / 0.346 / 0.859** | |

The offline gain does not carry over: on the twin camera_v1 draws like stage 2, and ACT is still about 1.6 times more
accurate. The scene config hash changed from `4dc272edd730` to `61a50cf4c384` only because the `vla_camera` section
was added; the robot and placement are unchanged.

For the camera to matter, the training data has to contain mistakes and their corrections: states where the pen
missed or drifted, labelled by a teacher that goes back to fill the gap. That is the next step.

**Files:** `<drawing>_<id>_smolvla_camera_v1_{comparison.png,run_metrics.json}`, `camera/`.

## Re-planning after every action (stage 2, 2026-10-29)

Same `smolvla_stage2` checkpoint and runner settings as above (zero noise, 10 flow steps), but each chunk's first
action only, re-planning every control tick (`act_policy_server.py --replan-every 1`) instead of executing 3 actions
per chunk. No retraining. Path RMSE / p95 / max error in mm:

| Drawing | Finished | Every action | Every 3 actions | ACT | Sim time s (every action) |
|---|---|---|---|---|---|
| Cat `323bd30d` | yes | 0.240 / 0.449 / 1.154 | 0.315 / 0.605 / 1.618 | **0.209 / 0.355 / 1.019** | 45.9 |
| Bird `1a5733d2` | yes | 0.223 / 0.433 / 0.877 | 0.343 / 0.630 / 1.234 | **0.207 / 0.356 / 0.641** | 40.7 |
| Dog `07379e95` | yes | 0.225 / 0.375 / 0.959 | 0.329 / 0.552 / 1.398 | **0.205 / 0.362 / 0.736** | 37.8 |
| Fish `a7cbe5e2` | yes | 0.222 / 0.394 / 1.332 | 0.307 / 0.541 / 1.530 | **0.204 / 0.312 / 1.186** | 84.5 |
| Frog `3f221bad` | yes | 0.219 / 0.363 / 0.843 | 0.337 / 0.551 / 1.108 | **0.197 / 0.313 / 0.790** | 32.8 |
| Horse `165c1709` | yes | 0.251 / 0.448 / 0.839 | 0.333 / 0.607 / 1.313 | **0.214 / 0.380 / 0.781** | 45.4 |
| **Mean** | 6/6 | 0.230 / 0.410 / 1.001 | 0.327 / 0.581 / 1.367 | **0.206 / 0.346 / 0.859** | |

Most of stage 2's gap to ACT was execution, not prediction: playing 3 actions blind per chunk. Re-planning every
tick cuts the mean path RMSE by 30% and leaves SmolVLA about 12% behind ACT (it was 59%). The cost is serving: every
tick needs a full chunk (about 220 ms with 10 flow steps on this GPU), so each drawing took 15 to 40 minutes of wall
time on the twin, and the real arm would need asynchronous chunks to keep 30 Hz.

The face portrait sketch (`studio_live/portrait_sketch_trajectory.json`, 21 strokes, not like anything in training)
with the same settings: 0.278 / 0.467 / 1.304 mm in 121.3 s, finished (ACT 0.242 / 0.397 / 1.261 mm). That is 21%
above SmolVLA's animal mean, so face sketches were added to the next training set.

**Files:** `<drawing>_<id>_smolvla_stage2_replan1_{comparison.png,run_metrics.json}`,
`portrait_sketch_smolvla_stage2_replan1_{comparison.png,run_metrics.json}`.

## Face portraits (stage 2 vs ACT, 2026-10-30/31)

`smolvla_stage2` (zero noise, re-planning every action, 10 flow steps) against ACT `act_real_kinematic_v2`,
both through the MCP `draw_trajectory` runner on the lab-arm twin (`config/xarm7_drawing_real.yaml`), the same
runner and settings as the animal tables above. Five face sketches from the validation split of
`wildtrace_real_kin_gaps_faces_v2` that are not in the DAgger recordings, plus the Studio portrait sketch
(`studio_live/portrait_sketch_trajectory.json`). Neither policy was trained on faces, so all six are new
to both. Path RMSE / p95 / max error in mm:

| Face | Finished | SmolVLA stage 2 | ACT | Sim time s (SmolVLA / ACT) |
|---|---|---|---|---|
| `04369443` | both | 0.301 / 0.526 / 1.309 | **0.267 / 0.455 / 1.173** | 94.8 / 90.2 |
| `1710ea61` | both | 0.277 / 0.459 / 1.079 | **0.246 / 0.400 / 0.988** | 79.9 / 78.5 |
| `26870728` | both | 0.297 / 0.454 / 1.387 | **0.261 / 0.395 / 1.177** | 93.8 / 92.6 |
| `3572f3a5` | both | 0.278 / 0.463 / 1.425 | **0.247 / 0.411 / 1.237** | 127.7 / 123.0 |
| `4f1c49dc` | both | 0.296 / 0.470 / 1.203 | **0.250 / 0.408 / 1.048** | 107.2 / 104.1 |
| Portrait sketch (Studio) | both | 0.278 / 0.467 / 1.304 | **0.242 / 0.397 / 1.261** | 121.3 / 110.8 |
| **Mean** | 6/6 both | 0.288 / 0.473 / 1.284 | 0.252 / 0.411 / 1.147 | |

On faces SmolVLA stage 2 averages 0.288 mm path RMSE and ACT 0.252 mm (+14% for SmolVLA; on the six animals it was +12%). Against their own animal means (0.230 and 0.206 mm), SmolVLA is +25% on faces and ACT +22%.

Scene config hash `61a50cf4c384` for the five faces; the portrait sketch rows ran at `4dc272edd730`, the same
robot and placement before the `vla_camera` section was added.

**Files:** `face_<id>_{smolvla_stage2,act}_{comparison.png,run_metrics.json}`; the portrait sketch rows are
`portrait_sketch_smolvla_stage2_replan1_*` here and `studio_live/portrait_sketch_act_*`.

## Camera SmolVLA camera_v2 to camera_v6 and the hybrid controller (#46, 2026-10-31)

Goal: a camera SmolVLA that sees the page and redraws its own gaps, which ACT's numeric path window cannot. Each
try fine-tunes the one before with `train_smolvla.py --view camera`. Closed loop is cat `323bd30d` on the lab-arm
twin through the MCP `draw_trajectory` runner; path RMSE / p95 / max error in mm. ACT on the same runner:
0.211 / 0.348 / 1.019 in 49.1 s.

| Try | Change | Offline first-action error | Twin, cat |
|---|---|---|---|
| camera_v2 | Gap-labelled data: ink skips, the IK teacher goes back over the gap and forward again; face sketches added | 0.309; redraw frames 0.754, page blanked 0.751 | Still ignores the page; cat and bird stall |
| camera_v3 | The target follows the pen back over a gap; the VLM trains; redraw frames x4 | 0.280; redraw 0.657, page blanked 0.760 | Reads the camera, but stalls at the cheek notch (index 94 to 221) |
| camera_v4 | + DAgger round 1 | 0.266 | Stalls at 96; with the page blanked it gets to 311, so the view causes the stall |
| camera_v5 | + targeted DAgger round 2 on the failed drawings, retrace frames (ink already at the target) x3 | 0.260; retrace 0.238 | Stalls at 231: waypoint timeout in `DRAW_STROKE_0`, failed |
| camera_v6 | + the teacher's redraw mode (back / forward) in the state (`--mode-state`) | 0.277; redraw 0.651 | Redraws by itself: backs up 3 to 8 mm but never gets within 0.6 mm of the line, 8/8 redraws given up |

**Why the redraws fail.** Validation frames of `wildtrace_real_kin_gaps_faces_v2` by teacher mode (384 frames per
mode; in the data 90.9% are normal, 4.6% back to the gap, 4.5% forward again). First-action error, with the median
direction cosine and the share pointing the opposite way where measured:

| Policy | Normal | Back to the gap | Forward again |
|---|---|---|---|
| Nearest normal-frame label | 0.639 | 1.80 (cos -0.12, 60% opposite) | 1.28 |
| ACT | 0.318 | 1.618 (38% opposite) | 1.02 |
| SmolVLA stage 2 | 0.399 | 1.63 (44% opposite) | 1.087 |
| SmolVLA camera_v5 | **0.284** | **0.627** (cos 0.876, 2% opposite) | **0.644** (cos 0.908) |

A redraw label points away from the normal drawing direction, and the policies that cannot see the ink (ACT, stage
2) get it about as wrong as the nearest normal label. camera_v5 reads the direction off the page (2% opposite), but
its step size is off: error 0.63 on redraw frames against 0.28 on normal ones. Right idea, imprecise motion, and a
stall wherever the path runs over ink it already drew.

**Serving speed.** `act_policy_server.py --cuda-graph` (`src/smolvla_cuda_graph.py`) captures sampling once and
replays it: about 30 ms per chunk instead of about 225 ms with 3 flow steps, same first-action error. Stage 2 on the
cat with it (3 flow steps, re-planning every action): 0.252 / 0.465 / 1.180 in 49.8 s, finished.

**Hybrid control.** The policy draws and a supervisor (`src/policy_assist.py`) watches the run's own bookkeeping.
After a 1.5 s stall IK drives the next 5 mm and hands back. When at least 1.5 mm of path behind the pen has no ink
within 1 mm (the labeller's coverage rule), IK goes back past the gap and forward again while the run's progress
stays put. The skip test (`skip_test_spacing_m` 0.06) injects 3 to 8 mm ink skips every 60 mm of inked travel on
average (seed 0).

| Cat `323bd30d` | Finished | Path RMSE / p95 / max | Sim time | Supervisor |
|---|---|---|---|---|
| ACT | yes | 0.211 / 0.348 / 1.019 | 49.1 s | - |
| **ACT + hybrid, skip test** | yes | **0.212 / 0.357 / 1.021** | **49.5 s** | 6 skips, 4 IK redraws, 0 given up, 0.19% uninked |
| SmolVLA stage 2, CUDA graph | yes | 0.252 / 0.465 / 1.180 | 49.8 s | - |
| SmolVLA camera_v5 | no, stalled at 231 | - | 43.2 s | - |
| SmolVLA camera_v5 + hybrid | yes | 0.290 / 0.590 / 1.696 | 91.3 s | 5 stall take-overs, 1 redraw, 1.0% uninked |
| SmolVLA camera_v6 redrawing itself, skip test | yes | 0.302 / 0.598 / 1.693 | 139.8 s | 9 stall take-overs, 8/8 redraws given up, 1.7% uninked |

Two of ACT's six skips needed no redraw: one fell during the first redraw's pass over line it had just re-inked,
and one 3.0 mm skip in a tight curve of stroke 3 left only 0.5 mm without ink within 1 mm, under the 1.5 mm gap
threshold.

**Decision:** ACT + the hybrid supervisor is the drawing controller. It repairs skipped ink at no accuracy cost
and about 1% extra sim time. Camera SmolVLA stays experimental: with the supervisor it is about 40% less accurate
than ACT and nearly 2 times slower. The camera only pays off for mistakes the run's bookkeeping cannot see (a paper
offset, skips on the real arm). A full twin evaluation of the hybrid controller is in [`../hybrid/`](../hybrid/).

**What did not work:** gap-labelled data alone (camera_v2 ignored the page); more DAgger and retrace weighting
(camera_v4, camera_v5 still stall at ink they already drew, and the stall moved but did not go away); telling the
policy the redraw mode (camera_v6 turns back but never lands on the line). None of camera_v2 to camera_v5 finishes
the cat on its own, and camera_v6 was only run with the supervisor.

Checkpoints: `models/smolvla_camera_v3` (Git LFS); camera_v2, camera_v4, camera_v5 and camera_v6 are local only.
Scene config hash `61a50cf4c384` for the stage 2 and camera_v5 runs, `4926204b8a89` for the hybrid runs (only the
`vla_camera` height changed, 0.60 to 0.835 m; robot and placement unchanged).

**Files:** `cat_323bd30d_{act_hybrid_skip60,smolvla_camera_v5_hybrid,smolvla_camera_v6_policy_redraw_skip60,smolvla_stage2_graph,smolvla_camera_v5}_{comparison.png,run_metrics.json}`.

### camera_v7_route: redraw as path following (2026-10-31)

While the teacher goes back to a gap, camera_v7 gets the stroke it already drew, reversed, as its target and path
window (`vla_features.redraw_route_env`, `train_smolvla.py --redraw-route`), built from the page history and the
mode per tick, so training and serving compute the same thing. The server switches it on from
`wildtrace_views.json` in the checkpoint. Fine-tuned from camera_v6 for 9k steps (`--mode-state`, same data and
weights).

Offline (val, 384 frames per mode; first-action error / median cosine): going back 0.531 / 0.939 (camera_v6 0.611 /
0.887), forward again 0.634 / 0.918 (0.635 / 0.909), normal 0.277 (0.282), retrace 0.231 (0.246). The val curve
was flat over the last 3k steps.

| Cat `323bd30d` | Finished | Path RMSE / p95 / max | Sim time | Supervisor |
|---|---|---|---|---|
| SmolVLA camera_v7 | yes | 0.282 / 0.560 / 1.645 | 58.3 s | - |
| SmolVLA camera_v7 redrawing itself, skip test | no: tip error 22.27 mm in `DRAW_STROKE_0` exceeds the safety limit, at 51% | - | 83.1 s | 5 skips, 9 redraws, 8 given up, 0 stall take-overs |

camera_v7 is the first camera policy to finish the cat on its own: no stall where camera_v5 stopped at 231, and
with the hybrid supervisor on (no skips) the run is identical, since the supervisor never steps in. It is still
34% less accurate than ACT and 19% slower. The redraw route does not carry over to closed loop: the policy spends
most ticks (3,665 of 4,800) backing up without landing within 0.6 mm of the line, as camera_v6 did, and the ninth
redraw ran 22 mm off it. Longer training on the same data is not worth it (flat val curve, closed-loop failure
rather than offline error); the next step would be on-policy data from camera_v7's own redraws (DAgger).

Checkpoint local only. Config hash `4926204b8a89`. **Files:**
`cat_323bd30d_{smolvla_camera_v7_route,smolvla_camera_v7_route_policy_redraw_skip60}_{comparison.png,run_metrics.json}`.
