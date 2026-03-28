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
- [x] Split fetch from ingest responsibilities
- [x] Add append-only fetch and ingest ledgers
- [x] Add `scripts/ingest_openimages.py`
- [x] Define fetch and ingest config in `configs/datasets.yaml`
- [x] Write bronze fetch and ingest NDJSON manifests
- [x] Preserve raw assets without mutation

## Bronze Validation
- [x] Add `scripts/validate_bronze.py`
- [x] Verify readability, dimensions, checksums, and path integrity
- [x] Flag duplicates and hard failures

## Silver Normalization
- [x] Add `scripts/normalize_to_silver.py`
- [x] Create resized RGB and grayscale assets
- [x] Normalize masks when available
- [x] Generate isolated-subject images
- [x] Emit silver QA manifest

## Gold Outline Export
- [x] Add `scripts/run_outline_inference.py`
- [x] Add `scripts/refine_outlines.py`
- [x] Emit SVG and normalized stroke trajectories
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
