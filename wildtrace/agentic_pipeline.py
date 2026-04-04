from __future__ import annotations

from wildtrace.bronze_stage import curate_bronze
from wildtrace.gold_stage import export_gold_ndjson, extract_trajectories, generate_dataset_report
from wildtrace.pipeline import fetch_main, reconcile_fetch_storage_main
from wildtrace.silver_stage import (
    enrich_and_crop_subjects,
    generate_line_diagrams,
    normalize_to_silver,
    select_final_by_angle,
    validate_and_retry_diagrams,
)
from wildtrace.stage_io import run_stage_main


def curate_bronze_main() -> int:
    return run_stage_main("Curate bronze assets into accepted training candidates.", curate_bronze)


def normalize_main() -> int:
    return run_stage_main("Normalize accepted bronze records into silver assets.", normalize_to_silver)


def enrich_and_crop_subjects_main() -> int:
    return run_stage_main("Crop and enrich silver subjects for diagram generation.", enrich_and_crop_subjects)


def generate_line_diagrams_main() -> int:
    return run_stage_main("Generate initial line-diagram candidates from cropped subjects.", generate_line_diagrams)


def validate_and_retry_diagrams_main() -> int:
    return run_stage_main("Validate and retry line-diagram generation with an agentic loop.", validate_and_retry_diagrams)


def select_final_by_angle_main() -> int:
    return run_stage_main("Select one final diagram per angle bucket and subcategory.", select_final_by_angle)


def extract_trajectories_main() -> int:
    return run_stage_main("Extract trajectories from final selected line diagrams.", extract_trajectories)


def export_main() -> int:
    return run_stage_main("Export canonical gold NDJSON records.", export_gold_ndjson)


def report_main() -> int:
    return run_stage_main("Generate dataset summary reports.", generate_dataset_report)


__all__ = [
    "curate_bronze",
    "curate_bronze_main",
    "enrich_and_crop_subjects",
    "enrich_and_crop_subjects_main",
    "export_gold_ndjson",
    "export_main",
    "extract_trajectories",
    "extract_trajectories_main",
    "fetch_main",
    "generate_dataset_report",
    "generate_line_diagrams",
    "generate_line_diagrams_main",
    "normalize_main",
    "normalize_to_silver",
    "reconcile_fetch_storage_main",
    "report_main",
    "select_final_by_angle",
    "select_final_by_angle_main",
    "validate_and_retry_diagrams",
    "validate_and_retry_diagrams_main",
]
