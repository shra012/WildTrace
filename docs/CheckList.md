# WildTrace Checklist

## Repo Scaffolding
- [x] Create docs, configs, scripts, tests, raw, processed, artifacts, and package directories
- [x] Add `pyproject.toml` for `uv`-managed Python 3.12 work
- [x] Add repo-local `.python-version`

## Environment
- [x] Install `uv` if missing
- [x] Create `.venv` with Python 3.12
- [x] Install project and dev dependencies into `.venv`

## Open Images Ingest
- [x] Add `scripts/fetch_openimages.py`
- [x] Keep fetch as the only raw acquisition stage
- [x] Define fetch config in `configs/datasets.yaml`
- [x] Write bronze fetch NDJSON manifests

## Bronze Curation
- [x] Add `scripts/curate_bronze.py`
- [x] Remove duplicates and unusable raws from the working bronze set
- [x] Add local enrichment for subcategory and tagging
- [x] Emit accepted bronze records for downstream stages

## Silver Normalization
- [x] Add `scripts/normalize_to_silver.py`
- [x] Create resized RGB and grayscale assets
- [x] Normalize masks when available
- [x] Generate isolated-subject images
- [x] Emit silver QA manifest
- [x] Persist mask-derived viewpoint features

## Subject Refinement
- [x] Add `scripts/enrich_and_crop_subjects.py`
- [x] Crop the drawable subject
- [x] Score segmentation quality before diagram generation

## Angle Filtering
- [x] Add `configs/viewpoints.yaml`
- [x] Classify candidates into the shared 5-view drawing buckets
- [x] Reject unusable angles during bronze curation
- [x] Select one final sample per angle bucket in silver

## Diagram Generation
- [x] Add `scripts/generate_line_diagrams.py`
- [x] Add `scripts/validate_and_retry_diagrams.py`
- [x] Use OpenCV pre-screening before semantic validation
- [x] Retry diagram generation with LangGraph until accepted or exhausted

## Gold Export
- [x] Add `scripts/select_final_by_angle.py`
- [x] Add `scripts/extract_trajectories.py`
- [x] Emit final diagrams, SVGs, and normalized stroke trajectories
- [x] Add `scripts/export_gold_ndjson.py`

## QA and Reporting
- [x] Add `scripts/generate_dataset_report.py`
- [x] Emit machine-readable dataset summary
- [x] Emit Markdown dataset summary

## Tests
- [x] Add end-to-end fixture-driven test
- [x] Run tests inside `.venv`

## Notebook Driver
- [x] Add `pipeline/pipeline.ipynb` as the step-by-step driver
- [x] Keep the notebook focused on invoking existing stage scripts in order

## Future Compatibility
- [x] Keep `task_type` in every record
- [x] Keep lineage/config hash in derived records
- [x] Keep trajectory coordinates normalized rather than robot-specific
- [x] Keep local path fields URI-ready for future cloud sync
