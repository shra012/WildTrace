# WildTrace ETL Plan

## Goal
Build a reproducible ETL pipeline that turns raw visual assets into IL-ready records for drawing tasks, while leaving enough schema headroom for later tasks such as pick-place and pouring.

The final orchestration target is to run these stage scripts as separate tasks in an Airflow DAG, so each pipeline boundary should stay explicit, scriptable, and scheduler-friendly.

## Scope
In scope:
- Open Images ingest
- metadata and lineage capture
- local-first storage layout
- bronze curation and duplicate removal
- local model enrichment and tagging
- silver normalization and subject cropping
- agentic line-diagram generation
- OpenCV pre-screening and semantic validation
- final angle-based selection per category and subcategory
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

- `bronze`: raw downloads plus first-pass curation, angle feasibility checks, and enrichment metadata
- `silver`: normalized images, masks, isolated subjects, crops, line-diagram attempts, and validation metrics
- `gold`: selected final diagrams, SVGs, normalized stroke trajectories, and canonical NDJSON records

Every derived artifact records lineage back to the bronze source plus the config hash and backend used to generate it.

The repository keeps each stage as an independent script entrypoint so local runs, notebook-driven inspection, and future Airflow DAG tasks all share the same execution surface.

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
2. `scripts/curate_bronze.py`
   - removes duplicates and unusable raws
   - runs angle feasibility checks plus local enrichment and tagging
   - emits accepted bronze records for downstream stages
3. `scripts/normalize_to_silver.py`
   - creates resized RGB, grayscale, normalized masks, and isolated subjects
4. `scripts/enrich_and_crop_subjects.py`
   - crops the drawable subject and scores segmentation quality
5. `scripts/generate_line_diagrams.py`
   - creates initial line-diagram candidates using the diagram backend
6. `scripts/validate_and_retry_diagrams.py`
   - runs OpenCV pre-screening, semantic validation, and retry logic
7. `scripts/select_final_by_angle.py`
   - keeps one best validated diagram per required angle bucket for each category and subcategory
8. `scripts/extract_trajectories.py`
   - converts final diagrams into SVG assets and normalized stroke trajectories
9. `scripts/export_gold_ndjson.py`
   - assembles final sample records
10. `scripts/generate_dataset_report.py`
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
- bronze curation removes duplicates and unusable raws
- silver assets are normalized, cropped, and diagram-ready
- OpenCV pre-screening reduces obvious bad generations before semantic validation
- line-diagram generation uses a stable backend contract and retry loop
- final gold records are traceable, renderable, and trajectory-ready for drawing
- no simulation or robot-only fields are required in this phase
