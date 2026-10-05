# Plan: Local model benchmark for the WildTrace pipeline (issue #14), v2, local only

## Context
WildTrace turns photos into line diagrams and then gold pen trajectories. Two model slots decide quality:
- **Outline rectifier**: FLUX.1-schnell GGUF img2img on a silhouette contour.
- **Outline validator**: a VLM judge, Ollama qwen2.5vl:7b.

Nothing has ever compared alternatives. The pipeline also records no timings, merges the VLM score 50/50 with OpenCV, and has no human labels.

v2 changes from v1: **local models only, sized for this laptop, at most 4 models per slot (current one included).** OpenRouter is dropped for now and can be added later.

Hardware:
- RTX 3080 Ti Laptop, **16 GB VRAM**
- WSL at **23 GB RAM**. FLUX's CPU offload uses about 20 GB of host RAM, and FLUX plus Ollama running together has OOM'd before.
- 469 GB free disk
- Ollama 0.19 installed; only `qwen2.5vl:7b` pulled

Decisions already made: humans label about 60 diagrams; 30 evaluation subjects (5 per category); both slots.

## Model lineup (4 per slot, 4-bit quantized unless noted)

**Validators** (all through Ollama, so serving, prompt and parsing are identical):
| # | Model | Size | Why |
|---|---|---|---|
**Round 1 (run first): 3 validators, all fitting fully in VRAM**
| # | Model | Size | Why |
|---|---|---|---|
| 1 | `qwen2.5vl:7b` | 6 GB, already pulled | **Current baseline** |
| 2 | `qwen3-vl:8b` | 6.1 GB | Next-generation dedicated vision-language model |
| 3 | `qwen3.5:9b` | 6.6 GB | Newest multimodal model that fits fully (the `latest` tag). Replaces gemma per user request |

**Round 2 (after reviewing round 1): larger models, partly offloaded to system RAM**
| # | Model | Size | Why |
|---|---|---|---|
| 4 | `qwen3-vl:30b-a3b-instruct` | 19.6 GB (~14 GB GPU + ~6 GB RAM) | Mixture of experts, 3B active params. Does the bigger MoE beat the dense 8B? ~10–20 s per call |

All round-1 models are Qwen. That isolates generation and architecture effects, but it doesn't test other model families. If round 1 shows the judge is the bottleneck, round 2 can add a non-Qwen model (e.g. `gemma4:12b`, 7.6 GB).

**Rectifiers** (diffusers/ONNX; run one at a time and unload between them):
| # | Model | Size | Why |
|---|---|---|---|
| 1 | FLUX.1-schnell GGUF Q4 img2img | cached | **Current baseline** |
| 2 | FLUX.1-Kontext-dev GGUF Q4 | ~7 GB transformer (new download) | Instruction-based image edit ("turn into a clean line drawing"); same family, so the comparison is fair |
| 3 | SD1.5 `ColoringBookSD` + ControlNet lineart | SD1.5 and ColoringBookSD cached, +1.4 GB ControlNet | Small and fast (~2 s per image), built for coloring-book output |
| 4 | `informative-drawings` line-art ONNX | 17 MB, cached | Direct photo → line-art network, no diffusion, runs in milliseconds; a speed/quality contrast |
| ref | `silhouette_outline_rectifier` (existing OpenCV code) | none | No-model floor; free, doesn't count toward the 4 |

Candidates were pulled from the previous experiments already in the HF cache. Every model tag is verified in the `smoke` step before the full run; if one doesn't exist or doesn't fit, it gets swapped for the next fallback: `qwen3.5:4b` (3.4 GB) for validators, SDXL + ControlNet for rectifiers. The registry sizes above were confirmed on 2026-09-27. For the 30B MoE, `smoke` must also confirm that the GPU/CPU split leaves RAM headroom (Ollama `num_gpu` layers set automatically, with an override in config).

## 1. Backends (`wildtrace/diagram.py`)
- **Validators**: no new class needed. `OllamaSemanticValidator` already takes `ollama_model_name`, and the benchmark just loops over names.
  - Add a `force_semantic` setting so the benchmark judges prescreen failures too. Today they're skipped at `:406`.
  - Add a `keep_alive` override so warm latency can be measured. It's currently hard-coded to `"0s"` at `:383`.
  - Store the raw VLM score, latency and token counts in `metadata`.
- **Shared helpers** pulled out of FLUX's `run` (`:1069-1137`):
  - `_silhouette_contour_image(subject_image, subject_mask, params, settings)`: Stage 1
  - `_binarize_to_subject(img, size)`: post-processing
  - `build_line_drawing_prompt(sample)`: from `_build_flux_prompt`, `:952`. Keep the method as a thin wrapper, since tests use it.
- **New rectifier classes**, registered in `build_outline_rectifier` (`:571`). Each gets the **same silhouette contour and prompt** as FLUX, and each lazy-loads with a `_load_pipeline()` that tests can monkeypatch, following FLUX:
  - `FluxKontextRectifier` (`flux_kontext_rectifier`): GGUF-quantized transformer, `FluxKontextPipeline`, with an instruction built from the shared prompt.
  - `SDControlNetLineartRectifier` (`sd_controlnet_lineart_rectifier`): `StableDiffusionControlNetPipeline` with the ColoringBookSD weights; the contour is the control image.
  - `InformativeDrawingsRectifier` (`informative_drawings_rectifier`): onnxruntime on the **subject crop**, because it needs a photo rather than a contour. Its input will differ from the others, and the report will say so.
- Every rectifier's `metadata` adds `{model, latency_s, peak_vram_gb}`.
- `configs/models.yaml` gets settings blocks for the new backends. **Defaults don't change** (FLUX and `auto`).
- New dependency if missing: `onnxruntime-gpu`, added through `uv add`.

## 2. Benchmark harness
- **`configs/benchmark.yaml`**: seed, `subjects_per_category: 5`, the model lists above, `referee: auto` (the validator with the best label F1), fallbacks.
- **`scripts/benchmark_models.py`**. Same layout as `sweep_diagrams.py`, but it loads config through `load_runtime_config`. Runs are **strictly sequential, one model in memory at a time**: stop the Ollama model and free CUDA between models. Subcommands:
  1. `build-evalset`
     - A seeded 5-per-category sample of `silver_subjects.ndjson`, written to `benchmarks/evalset_v1.json`.
     - A 60-diagram label set from `validated_diagrams.ndjson`: about 30 accepted and 30 rejected, spread across categories, with at most 10 prescreen failures.
  2. `smoke`: pull and load each model, run 1 sample, record whether it loaded, latency and peak VRAM. Swap in fallbacks where needed.
  3. `validators`
     - All 4 validators judge the same 60 labeled PNGs with the same prompt.
     - Metrics: accuracy, precision, recall, F1 against the human labels; failed-to-parse rate; score spread; warm p50/p95 latency; peak VRAM.
  4. `rectifiers`
     - 4 rectifiers plus the OpenCV reference, with **one attempt** per subject (no retry loop).
     - Metrics:
       - OpenCV pass rate and score (`validate_with_opencv`, `:668`)
       - referee pass rate and score
       - trajectory usable via `_build_trajectory_payload` (`gold_stage.py:66`): 1–6 strokes and point counts
       - s/img, peak VRAM, peak RAM
     - Writes a contact sheet per model.
  5. `report`: `summary.json` and `summary.md` with ranked tables.
- **Outputs**:
  - Raw results go to `outputs/benchmarks/<run_id>/`, which is gitignored.
  - The summary is copied to `docs/benchmarks/<run_id>.md`, which is committed.
  - Every row carries the config hash and git SHA.

## 3. Human labeling page
- Publish a private artifact that shows the 60 diagrams with their source crops, with Accept/Reject buttons and a note field.
- Labels go to the shared `db` so several teammates can label and Claude reads them with `ArtifactData`.
- Images are uploaded as assets.
- Load `artifact-capabilities` and `artifact-design` first.
- Labels are exported to `benchmarks/labels_v1.ndjson`.

## 4. Small fix
- `scripts/sweep_diagrams.py:47`: use `load_runtime_config` instead of raw `yaml.safe_load`. Today `${VAR:-default}` placeholders break `build_outline_rectifier`.

## Volume, time, cost
- **Diagrams generated**: 30 subjects × 5 rectifiers (4 + OpenCV reference) = **150**.
- **Referee calls**: 150.
- **Money**: **$0**, everything runs locally. New downloads: round 1 about 21 GB (qwen3-vl:8b 6.1, qwen3.5:9b 6.6, Kontext GGUF ~7, ControlNet 1.4); round 2 adds about 19.6 GB (qwen3-vl:30b-a3b).
- **Round-1 validator calls**: 60 labels × 3 = 180, plus referee calls.
- **Estimated wall time**, to be confirmed by `smoke`:

| Step | Estimate |
|---|---|
| FLUX-schnell, with CPU offload | ~30 × 20 s ≈ 10 min |
| Kontext | ~30 × 40 s ≈ 20 min |
| SD1.5 | ~1 min |
| ONNX | seconds |
| Validators, warm | ~390 calls at 3–8 s each (30B MoE: 10–20 s) ≈ 45–70 min |
| **Total** | **~1.5–2 h** |

## Critical files
- `wildtrace/diagram.py`: shared helpers, 3 new rectifier classes, Ollama settings, factory
- `configs/models.yaml`, plus the new `configs/benchmark.yaml`
- New `scripts/benchmark_models.py`
- `scripts/sweep_diagrams.py`
- `tests/test_pipeline.py`
- `pyproject.toml`: `onnxruntime-gpu`, only if it's missing

## Verification
1. Unit tests, with no GPU or network. Patch `_load_pipeline` the way the FLUX tests do (`:595-633`) and `_request` for Ollama (`:676-699`):
   - Each new rectifier produces a binarized PNG at the subject's size.
   - The factory accepts the new names.
   - `force_semantic` and `keep_alive` are respected.
   - Run: `uv run pytest -q`. The baseline today is 21 passing.
2. Run `python scripts/sweep_diagrams.py --limit 2` to check the config fix.
3. Run `python scripts/benchmark_models.py smoke`. All 8 models should load, and peak VRAM must stay under 16 GB.
4. Run `build-evalset`, then labeling, then `validators`, `rectifiers` and `report`. Check the tables and contact sheets.
5. Optional: run the winning rectifier's outputs through `extract_trajectories`, then draw one on the robot with the `wildtrace-sim` `draw` tool.
