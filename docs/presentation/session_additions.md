# Presentation additions

These notes use records already in the checkout. SDXL drawing quality is not filled in here. Those values are null in `docs/benchmarks/sdxl_preflight.json`.

## 1. End-to-end showcase

Two committed artifacts have to stay separate. The dog below is a pipeline drawing with its exported strokes. The tracking numbers are the configured Isaac Sim puppy run, not a replay of that dog.

### Dog `96ab6db6d9116afe`

| Stage | Record |
| --- | --- |
| Label | `benchmarks/labelset_v1.json` item `dog_96ab6db6`, pipeline status `accepted`, attempt 01 |
| Drawing in this checkout | `isaac-sim/inputs/diagrams/Dog/96ab6db6d9116afe.png` |
| Diagram path named by the label set | `outputs/silver/line_diagrams/Dog/96ab6db6d9116afe_attempt01.png` is not in this checkout. The committed copy is the isaac-sim diagram above. Silver photos, crops, and masks are also absent (0 of 20 evalset crops). |
| Strokes | `isaac-sim/inputs/trajectories/Dog/96ab6db6d9116afe.json`: 4 strokes, 210 points, canvas 689×616, `normalized_canvas`, `angular_subsample` |
| SVG | `isaac-sim/inputs/svg/Dog/96ab6db6d9116afe.svg` |

Four strokes is inside the current export cap of 6 (`configs/export.yaml`).

### Historical tracking (puppy two-stroke)

The simulator default is `isaac-sim/data/puppy_two_stroke.json` (`drawing_id: puppy_two_stroke`), a purpose-built two-stroke target. The image is `isaac-sim/outputs/two_stroke_comparison.png`. `isaac-sim/outputs/run_metrics.json` for that run:

| Metric | Value |
| --- | --- |
| Status | finished |
| Strokes | 2 |
| Pen-down position RMSE | 0.738 mm |
| Pen-down mean / p95 / max | 0.686 mm / 1.138 mm / 2.125 mm |
| Desired-to-executed path RMSE | 0.177 mm |
| Duration | 539.7 s |

`isaac-sim/outputs/RESULTS_COMPARISON.md` is a different before/after (2.00 mm RMSE to 1.35 mm). Do not merge those figures with `run_metrics.json`.

## 2. Failure analysis

Round `r2` (`docs/benchmarks/r2.md`) is the evidence. It was measured before the contour-threshold fix, so it is the historical failure record, not a retest of the current helper.

| Failure | What r2 saw | How it is detected | What the retry does |
| --- | --- | --- | --- |
| Blank page | OmniGen2 left 9 of 20 final drawings blank. The local 7B judge described blanks as a simple cartoon and passed them. | OpenCV `low_foreground_ratio` when dark pixels fall under `min_foreground_ratio` (0.01). The 235B referee rejected blanks; qwen3-vl-8b agreed with that referee (κ = 0.83). | `next_generator_params` raises `strength` by 0.05, which gives the generator more freedom to add lines. OmniGen2 and SDXL also change the seed on the next attempt filename, so a blank seed is not repeated. SDXL lowers ControlNet scale when strength rises. |
| Changed pose | FLUX passed the strong referee 20/20 but drew a generic pose. Median silhouette IoU was about 0.2. A lying cat became a sitting cat. Facing-direction match was 65% for FLUX and 20% for OmniGen2. | The OpenCV prescreen does not check pose. Pose shows up in the VLM reason and in silhouette IoU / direction metrics in `wildtrace/eval_metrics.py`. | A semantic failure lowers `strength` by 0.03 so the next draw stays closer to the contour. The judge may also return a `prompt` suggestion. SDXL raises ControlNet scale when strength falls. |
| Fragmented drawing | OmniGen2's faithful sketches (betta IoU 0.89, frog IoU 0.91) were rejected for too many components. Thin broken lines split apart. | `too_many_components`, `too_many_small_contours`, and `too_many_lower_fragments`. | Too many components lowers strength by 0.05. Small contours or lower-body fragments lower it by 0.03. The loop stops at 3 attempts and stores `rejected` with the flags. |

The CPU perturbation check below is the same prescreen reacting to damaged lines, without a VLM.

## 3. Validator robustness (CPU)

`scripts/validator_robustness.py` rescores the 10 committed drawings in `isaac-sim/inputs/diagrams` with the production OpenCV prescreen. No VLM and no diffusion model. Full rows are in `docs/presentation/validator_robustness.json`.

| Variant | Pass rate | Mean score | Flags that appeared |
| --- | --- | --- | --- |
| Original | 10/10 | 0.571 | none |
| Gaussian blur, radius 2 | 0/10 | 0.000 | too many components 10/10, lower-body fragments 9/10, low foreground 7/10 |
| Resize to half and back | 3/10 | 0.191 | too many components 7/10, lower-body fragments 6/10, small contours 3/10 |
| 2% salt-and-pepper | 0/10 | 0.000 | too many components, small contours, and lower-body fragments on all 10 |

Blur and speckles break the line into components, which is the same gate that rejected fragmented OmniGen2 drawings. The prescreen is sensitive to those defects. It does not measure pose or species.

## 4. Traceability example

Subject `1e178834a919a6ab` in `benchmarks/evalset_v2.json` (Cat, seed 14):

| Field | Value |
| --- | --- |
| Angle | `front_right_3q` |
| Crop / mask / isolated | `outputs/silver/crops/Cat/1e178834a919a6ab_crop.png`, `outputs/silver/masks/Cat/1e178834a919a6ab_mask.png`, `outputs/silver/isolated_subjects/Cat/1e178834a919a6ab_isolated.png` |
| Lineage | bronze sample `1e178834a919a6ab`, bronze input hash `f6ca5656d160f1a2`, silver config hash `6ac55c46dea589c5`, silver input hash `0f9ff33cd831dcac`, stage `enrich_and_crop_subjects` |
| Label | BioCLIP `cat`, confidence 0.545. That confidence is not a labeled accuracy. |
| Quality | `accepted`, segmentation score 0.907 |

A comparison run then attaches:

- Settings from `configs/models.yaml` plus the selected benchmark file (`WILDTRACE_BENCHMARK_CONFIG`, default `configs/benchmark.yaml`).
- Provenance `git_sha` and `config_hash`. The hash covers the selected benchmark file, `models.yaml`, and `export.yaml`.
- Each attempt file `{category}/{sample_id}_attemptNN.png`, OpenCV flags, validator reason, and combined score from `run_langgraph_validation_loop`.
- Trajectory export only if the drawing yields 1 to `export.max_strokes` (6) strokes.

For `sdxl-r1`, point `WILDTRACE_BENCHMARK_CONFIG` at `configs/benchmark_sdxl.yaml`. Outputs land in `outputs/benchmarks/sdxl-r1`. That hash will differ from an `r2` hash because the benchmark file bytes differ.

## 5. Model decision table

| Model | Role | Status | Finding | Decision |
| --- | --- | --- | --- | --- |
| FLUX.1-schnell GGUF img2img | Outline generator | Implemented. Evaluated in r2. | Strong referee 20/20, exportable 20/20, weak pose match. | Stays the production default. |
| OmniGen2 | Outline generator | Implemented. Evaluated in r2. | Keeps pose when it draws; 9/20 finals were blank. | Not the default. Rerun in `sdxl-r1` under the same hosted judge. |
| SD1.5 ColoringBook + lineart ControlNet | Outline generator | Implemented. Not in the r2 quality table. | Small GPU model. Shares the contour helper, including the threshold fix. | Registered. Not the comparison under test. |
| SDXL + scribble ControlNet | Outline generator | Implemented now. Inference not run. | No quality or latency number. Integration tests mock the pipeline. | Alternative in `sdxl-r1`. Contribution: a larger text-to-image prior with scribble control, on the same validation and trajectory path, so pose can be constrained without leaving the existing loop. |
| informative-drawings ONNX | Outline generator | Implemented. | Photo-to-sketch. It does not use the contour or the text prompt. | Speed floor in the rectifier list, not in `sdxl-r1`. |
| OpenCV silhouette | Outline generator | Implemented. | No learned model. | Reference floor. |
| qwen2.5vl:7b local | Judge | Evaluated in r2. | Passed blanks. κ = 0 against the 235B referee. | Not the judge for `sdxl-r1`. |
| qwen3-vl-8b hosted | Judge | Evaluated in r2. | Agreed with the 235B referee (κ = 0.83) and rejected blanks. | Shared judge for FLUX, OmniGen2, and SDXL. |
| qwen3-vl-235b hosted | Referee | Evaluated in r2. | Strong referee. | Final-drawing referee for `sdxl-r1`. |
| Molmo | Stroke planning | Owned by another teammate. | Not part of this outline comparison. | Left out of this work. |
| Florence-2 | Prompt enrichment | Considered, not implemented. | — | Not selected. |
| InternVL3.5-8B | Alternate judge | Proposed, not implemented. | — | Not selected. |

## Notes the workbook still needs, stated from the repo

The workbook file was not rewritten.

**Gravity.** `isaac-sim/config/xarm7_drawing.yaml` sets `gravity_m_s2: 9.81`. The README says the scene runs at Earth gravity and that an older zero-gravity checkpoint must not be resumed, because gravity changes the transition dynamics. Zero-g reach numbers in `isaac-sim/WildTrace_RL_Isaac_Sim_Overview.md` are the previous setup. Current drawing completion is the 1 g run.

**Eight strokes versus six.** `configs/export.yaml` sets `max_strokes: 6`. `trajectory_ok` requires 1 to 6 strokes. The stored horse trajectories `isaac-sim/inputs/trajectories/Horse/7c54b4c75832b3b9.json` and `e659138f70574103.json` have `stroke_count: 8`. The workbook's 8-stroke horse timing is that historical trajectory. A new export of the same drawing would fail the current gate. The dog showcase above (4 strokes) would pass it.

**Labeled accuracy.** BioCLIP confidence on a subject is not accuracy. `benchmarks/labelset_v1.json` is a diagram label set, and this checkout does not contain the crops needed to score enrichment or viewpoint classification against held-out labels.

**Behavior cloning.** Gravity-aware PPO and a behavior-cloning test live under `isaac-sim/`. This checkout does not contain a held-out rollout table for the cloning policy. Do not treat training logs as that evaluation.

**What is proposed, implemented, or evaluated.** Use the table above. SDXL is implemented and not evaluated. r2 is evaluated on the pre-fix contour helper.

**Records to keep.** r2 writeups are `docs/benchmarks/r2.md` and `docs/benchmarks/r2_findings.md`. Remote backup branches fetched with this session include `backup/before-gravity-author-fix-20261004` and `backup/development-before-molmo-author-fix-20261004`.

## Pipeline interface

`run_langgraph_validation_loop` in `wildtrace/diagram.py` is the generation API.

Inputs: sample dict (`sample_id`, `category`, `subcategory`, `angle_bucket`, optional `view_features` and `crop_bbox`), crop path, optional isolated-image path, optional mask path, diagram directory, rectifier, validator, OpenCV settings, initial params, and `max_attempts`.

Output record: `diagram_path`, `diagram_attempt`, `diagram_validation_status` (`accepted` or `rejected`), `diagram_validation_score` (mean of the OpenCV score and the judge score), `opencv_flags`, `outline_validator_reason`, and `diagram_generator_backend` (model ids, seed, prompt, and scales for SDXL).

Failures:

- Unknown `backend` raises `ValueError` before a run.
- Missing `torch` or `diffusers` raises `RuntimeError` from `validate_ready`.
- A checkpoint download or load failure raises `RuntimeError` from `_load_pipeline`.
- OpenCV flags or a judge rejection are not exceptions. The loop retries until `max_attempts`, then returns `rejected`.
- An unparseable judge response is a failed validation with score 0.
- In the benchmark worker, an exception on one subject is stored as `error` and the next subject still runs.

Benchmark commands are `scripts/benchmark_setups.py pipeline|evaluate|referee|report`. `pipeline --resume` skips sample ids already present for that setup.

## Record schema

The system of record is files, not a database service.

| Record | Where it lives | Identity |
| --- | --- | --- |
| Eval subject | `benchmarks/evalset_v2.json` | `sample_id` |
| Lineage | `lineage` on that subject | bronze sample id, bronze input hash, silver config hash, silver input hash |
| Diagram attempt | `{diagram_root}/{category}/{sample_id}_attemptNN.png` plus the validation record | sample id + attempt |
| Trajectory | `isaac-sim/inputs/trajectories/{category}/{sample_id}.json` | `stroke_count`, `point_count`, normalized points |
| Benchmark row | `outputs/benchmarks/{run_id}/*.ndjson` | setup, sample id, `git_sha`, `config_hash` |

A hosted store, if one is added later, can mirror those four tables. Nothing in this session deploys one.

## Cloud shape, not deployed

Generation stays on one GPU worker because FLUX and SDXL are large and the benchmark already runs a single setup in a subprocess. Crops, masks, and attempt PNGs belong in object storage addressed by `sample_id` and attempt. The hosted judge is OpenRouter (`qwen/qwen3-vl-8b-instruct`, referee `qwen/qwen3-vl-235b-a22b-instruct`); the API key stays on the worker. Isaac Sim stays on the simulation machine and reads exported trajectory JSON. It is not on the request path. This checkout cannot run that job yet: the preflight shows the 20 assets missing, the inference packages missing, and no OpenRouter key, even though a CUDA device is visible.
