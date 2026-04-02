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
