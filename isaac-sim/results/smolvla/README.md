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

## What could close the gap

- Sample each chunk from the same fixed noise, or average several samples, to remove the chunk-to-chunk wobble.
- Train longer: the last validation check (0.65 at 20k) was the best one, though the checks are noisy.
- Replace the rendered path image with a real camera view, which is what a VLA is built for, and is the case where
  ACT's hand-made path window is not available.
