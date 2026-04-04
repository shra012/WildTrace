# Bronze Pipeline Plan

This document describes the current bronze stage used by the agentic Airflow-ready pipeline. The older ingest/validate bronze workflow has been removed.

## Goal
Keep Open Images acquisition separate from bronze curation so the pipeline can fetch raw assets once, resume cleanly, and curate only new or changed samples on reruns.

This split keeps bronze ready for orchestration as separate Airflow DAG tasks instead of a single monolithic job.

## Bronze Scope
This document covers the current bronze stages:

1. `scripts/fetch_openimages.py`
2. `scripts/curate_bronze.py`

Silver and gold stages read the latest successful bronze accepted view rather than any legacy ingest ledger.

## Why Split Fetch From Curation
`fetch_openimages` and `curate_bronze` have different responsibilities:

- `fetch_openimages` acquires raw source assets and raw metadata from Open Images or a curated record catalog.
- `curate_bronze` removes duplicates and unusable raws, checks angle feasibility, and enriches accepted samples for downstream stages.

That separation gives us reliable restart behavior, cheaper reruns, and cleaner lineage.
It also maps cleanly onto Airflow, where each script can become an isolated DAG task with explicit upstream and downstream dependencies.

## Bronze Semantics
Bronze is the raw immutable local store. There is no separate landing layer in this phase.

Primary locations:
- `raw_data/bronze/openimages/images/<category>/<sample_id>/vXXXX.ext`
- `raw_data/bronze/openimages/masks/<category>/<sample_id>/vXXXX.ext`
- `raw_data/bronze/openimages/metadata/<category>/<sample_id>/vXXXX.json`
- `raw_data/bronze/manifests/openimages_fetch.ndjson`
- `raw_data/bronze/manifests/openimages_fetch_latest.ndjson`
- `raw_data/bronze/manifests/bronze_curated.ndjson`
- `raw_data/bronze/manifests/bronze_accepted.ndjson`

## Checkpoint Model
Checkpointing is append-only and ledger-based.

- Fetch ledger stores every successful or failed fetch attempt that produced a new versioned bronze artifact.
- Curated manifest stores all curation decisions, including deletions and rejected samples.
- Accepted manifest is the downstream handoff for silver.

## Incremental Rules
- Already-fetched unchanged samples are skipped by default.
- Already-curated unchanged samples are skipped by default by reusing the fetch latest view.
- New bronze assets are appended and curated on the next run.
- If content changes for the same logical sample, a new `asset_version` is created and older lineage is preserved.

Identity rules:
- `sample_id` is the stable logical identity.
- `checksum` is the content identity.
- `asset_version` increments when content changes for that logical sample.

## Job Order
1. `fetch_openimages.py`
2. `curate_bronze.py`
3. downstream silver and gold jobs

## Acceptance Criteria
- rerunning fetch with unchanged inputs does not create duplicate bronze versions
- rerunning curation with unchanged inputs does not create duplicate accepted samples
- adding new source images results in only new bronze and accepted records
- duplicate or unusable raws are removed from the working bronze set
- downstream stages read the latest accepted bronze view without any legacy ingest dependency

---

# Silver Pipeline Plan

## Goal
Turn accepted bronze samples into diagram-ready silver assets and generate coloring-book line art using a single FLUX img2img backend.

## Silver Stages

### `scripts/normalize_to_silver.py`
- reads `raw_data/bronze/manifests/bronze_accepted.ndjson`
- creates resized RGB, grayscale, normalized masks, and isolated-subject images
- emits `outputs/silver/checkpoints/silver_subjects.ndjson`

### `scripts/enrich_and_crop_subjects.py`
- crops the drawable subject from the silver normalized image using the segmentation mask
- scores segmentation quality
- emits cropped subject assets under `outputs/silver/crops/`

### `scripts/generate_line_diagrams.py`
- generates coloring-book line art using the FLUX img2img pipeline via `run_drawn_diagram_pass`
- **Stage 1**: extract silhouette contour from the segmentation mask using OpenCV morphological close + `findContours` + `approxPolyDP`
- **Stage 2**: refine with FLUX.1-schnell (GGUF-quantized, Q4_K_S) img2img at `strength=0.90`
- no separate outline-generator stage; the rectifier handles both stages internally
- emits `outputs/silver/checkpoints/line_diagram_attempts.ndjson`

### `scripts/validate_and_retry_diagrams.py`
- runs the LangGraph validation loop: OpenCV pre-screening → semantic validation (Anthropic or Ollama VLM) → retry
- adjusts FLUX `strength` param between attempts (range 0.60–0.98)
- OpenCV checks: foreground ratio, component count, small-contour ratio
- emits `outputs/silver/checkpoints/validated_diagrams.ndjson`

## Diagram Backend
The single supported backend is `FluxSilhouetteRectifier` (configured via `configs/models.yaml` under `outline_rectifier`).

Key config fields:
- `backend: flux_silhouette_rectifier`
- `strength: 0.90` — FLUX img2img denoising strength
- `gguf_repo / gguf_file` — quantized FLUX.1-schnell weights
- `base_model` — Black Forest Labs FLUX.1-schnell for the VAE and text encoder

`SilhouetteOutlineRectifier` is available as a lightweight CPU fallback for tests and local inspection.

## Silver Artifact Locations
- `outputs/silver/images/<category>/` — resized RGB
- `outputs/silver/masks/<category>/` — normalized masks
- `outputs/silver/crops/<category>/` — cropped subjects
- `outputs/silver/diagrams/<category>/` — generated line diagrams
- `outputs/silver/checkpoints/` — NDJSON manifests for each stage

## Silver Acceptance Criteria
- silhouette contour extraction works without GPU
- FLUX img2img produces clean coloring-book outlines at strength=0.90
- OpenCV pre-screening rejects obvious bad outputs before semantic validation
- LangGraph retry loop adjusts strength and retries up to the configured max attempts
- every diagram record carries lineage back to the bronze source and the config hash used to generate it
