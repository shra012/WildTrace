# Bronze Pipeline Plan

## Goal
Split Open Images acquisition from bronze registration so the pipeline can fetch raw assets once, resume cleanly, and ingest only new or changed bronze samples on reruns.

## Bronze Scope
This document only covers the bronze-stage refactor:

1. `scripts/fetch_openimages.py`
2. `scripts/ingest_openimages.py`
3. `scripts/validate_bronze.py`

Silver and gold stages stay functionally the same, but they should read the latest successful bronze ingest view rather than a rewritten one-shot manifest.

## Why Split Fetch From Ingest
`fetch_openimages` and `ingest_openimages` have different responsibilities:

- `fetch_openimages` acquires raw source assets and raw metadata from Open Images or a curated record catalog.
- `ingest_openimages` turns those already-fetched local bronze assets into the canonical bronze manifest for downstream stages.

That separation gives us reliable restart behavior, cheaper reruns, and cleaner lineage.

## Bronze Semantics
Bronze is the raw immutable local store. There is no separate landing layer in this phase.

Primary locations:
- `raw_data/bronze/openimages/images/<category>/<sample_id>/vXXXX.ext`
- `raw_data/bronze/openimages/masks/<category>/<sample_id>/vXXXX.ext`
- `raw_data/bronze/openimages/metadata/<category>/<sample_id>/vXXXX.json`
- `raw_data/bronze/manifests/openimages_fetch.ndjson`
- `raw_data/bronze/manifests/openimages_fetch_latest.ndjson`
- `raw_data/bronze/manifests/openimages_ingest.ndjson`
- `raw_data/bronze/manifests/openimages_ingest_latest.ndjson`

## Checkpoint Model
Checkpointing is append-only and ledger-based.

- Fetch ledger stores every successful or failed fetch attempt that produced a new versioned bronze artifact.
- Ingest ledger stores every successful or failed bronze registration attempt that produced a new versioned canonical record.
- Latest-view manifests are derived materializations for downstream consumers.

## Incremental Rules
- Already-fetched unchanged samples are skipped by default.
- Already-ingested unchanged samples are skipped by default.
- New bronze assets are appended and ingested on the next run.
- If content changes for the same logical sample, a new `asset_version` is created and older lineage is preserved.

Identity rules:
- `sample_id` is the stable logical identity.
- `checksum` is the content identity.
- `asset_version` increments when content changes for that logical sample.

## Job Order
1. `fetch_openimages.py`
2. `ingest_openimages.py`
3. `validate_bronze.py`
4. downstream silver and gold jobs

## Acceptance Criteria
- rerunning fetch with unchanged inputs does not create duplicate bronze versions
- rerunning ingest with unchanged inputs does not append duplicate ingest records
- adding new source images results in only new bronze and ingest records
- changed source content produces a new version instead of overwriting prior lineage
- downstream validation reads the latest successful ingest view without duplicate processing
