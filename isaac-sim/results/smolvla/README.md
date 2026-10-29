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
from `smolvla_stage1`, batch 64, lr 1e-4 (51M of 323M parameters train), about 5.5 h on the same GPU.

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
  in `config/xarm7_drawing_real.yaml`;
- **tip crop:** the same paper zoomed to 45 mm around the pen tip, with the upcoming path drawn over it.

Both show the target traced so far in light blue under the ink drawn so far in black, rendered from the episode's
earlier frames in training and from the ticks since reset in the policy server.
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

**Files:** `<drawing>_<id>_smolvla_camera_v1_{comparison.png,run_metrics.json}`.
