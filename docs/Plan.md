# WildTrace ETL Plan

## Goal
Build a reproducible ETL pipeline that turns raw visual assets into IL-ready records for drawing tasks, while leaving enough schema headroom for later tasks such as pick-place and pouring.

## Scope
In scope:
- Open Images ingest
- metadata and lineage capture
- local-first storage layout
- bronze validation
- silver normalization
- model-pluggable outline generation
- outline refinement and SVG export
- normalized trajectory export
- NDJSON packaging
- dataset QA and reporting

Out of scope:
- MuJoCo or Isaac Sim
- xArm execution
- TCP calibration
- world-coordinate robot trajectories
- imitation-learning training jobs

## Architecture
The pipeline is organized as a local-first ETL system with three data layers:

- `bronze`: immutable raw downloads and ingest manifests
- `silver`: normalized images, masks, isolated subjects, and quality metrics
- `gold`: outline rasters, SVGs, normalized stroke trajectories, and canonical NDJSON records

Every derived artifact records lineage back to the bronze source plus the config hash and backend used to generate it.

## Data Source
Open Images is the first supported source. The initial ingest job is driven by a configurable record catalog so the same pipeline can work with local fixtures, curated NDJSON/CSV manifests, or later automated Open Images catalog generation.

## Storage Strategy
Files are stored locally in repo-managed folders today. All records store local paths in a way that can later expand to cloud object storage URIs without changing the record contract.

## Dataset Contract
NDJSON is the canonical processed dataset format. Each final sample is a single JSON object on its own line with source references, artifact references, QA state, lineage, and a normalized trajectory payload.

## Future Compatibility
Every record contains a `task_type` field. This phase uses `drawing`, but the same schema pattern is reserved for future task families such as `pick_place` and `pouring`.

## Pipeline Stages
1. `scripts/fetch_openimages.py`
   - reads `configs/datasets.yaml`
   - discovers Open Images candidates and fetches raw assets into bronze
   - appends to `raw_data/bronze/manifests/openimages_fetch.ndjson`
2. `scripts/ingest_openimages.py`
   - consumes only local fetched bronze assets
   - appends to `raw_data/bronze/manifests/openimages_ingest.ndjson`
3. `scripts/validate_bronze.py`
   - verifies paths, readability, dimensions, checksums, and duplicates
4. `scripts/normalize_to_silver.py`
   - creates resized RGB, grayscale, normalized masks, isolated subjects, and QA records
5. `scripts/run_outline_inference.py`
   - runs a pluggable outline backend
   - stores outline proposal rasters and inference manifests
6. `scripts/refine_outlines.py`
   - converts outline rasters into simplified strokes and SVG assets
7. `scripts/export_gold_ndjson.py`
   - assembles final sample records
8. `scripts/generate_dataset_report.py`
   - produces dataset-level JSON and Markdown summaries

## Record Types
- `BronzeIngestRecord`
- `BronzeValidationRecord`
- `SilverSampleRecord`
- `OutlineInferenceRecord`
- `GoldOutlineRecord`
- `OutlineTrajectory`
- `DatasetReport`

## Acceptance Criteria
- ingest is repeatable from config
- bronze assets remain immutable
- silver assets are normalized and quality-scored
- outline generation uses a stable backend contract
- final gold records are traceable and renderable
- no simulation or robot-only fields are required in this phase
