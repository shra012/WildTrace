from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from wildtrace.images import sample_outline_strokes, write_svg
from wildtrace.io_utils import read_ndjson, resolve_repo_path, write_ndjson
from wildtrace.pipeline import utc_now
from wildtrace.stage_io import (
    bronze_curated_manifest_path,
    bronze_accepted_manifest_path,
    diagram_trajectories_manifest_path,
    gold_records_dir,
    selected_diagrams_manifest_path,
    silver_subjects_manifest_path,
    validated_diagrams_manifest_path,
)


def _row_input_hash(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True)


def _trajectory_config_hash(runtime: dict[str, Any]) -> str:
    return json.dumps(runtime["export"], sort_keys=True)


def _paths_exist(repo_root: Path, *values: str | None) -> bool:
    for value in values:
        if value and not resolve_repo_path(repo_root, value).exists():
            return False
    return True


def _gold_category_paths(repo_root: Path, runtime: dict[str, Any], category: str) -> dict[str, Path]:
    gold_storage = runtime["storage"]["gold"]
    return {
        "diagrams": resolve_repo_path(repo_root, gold_storage["diagrams_dir"]) / category,
        "svg": resolve_repo_path(repo_root, gold_storage["svg_dir"]) / category,
        "trajectories": resolve_repo_path(repo_root, gold_storage["trajectories_dir"]) / category,
    }


def _selected_gold_candidates(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in read_ndjson(selected_diagrams_manifest_path(repo_root, runtime))
        if record.get("selected_for_gold")
    ]


def _sample_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in read_ndjson(path)}


def _copy_diagram(diagram_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(diagram_path, destination)


def _build_trajectory_payload(diagram: Image.Image, export: dict[str, Any]) -> dict[str, Any]:
    strokes = sample_outline_strokes(
        diagram,
        max_strokes=int(export["max_strokes"]),
        max_points_per_stroke=int(export["max_points_per_stroke"]),
        min_points_per_stroke=int(export["min_points_per_stroke"]),
    )
    return {
        "strokes": strokes,
        "payload": {
            "coordinate_frame": export["coordinate_frame"],
            "canvas_width": diagram.width,
            "canvas_height": diagram.height,
            "stroke_count": len(strokes),
            "point_count": sum(len(stroke) for stroke in strokes),
            "bounds": {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0},
            "normalization_method": "width_height_normalized",
            "simplification_method": export["simplification_method"],
            "strokes": [
                {
                    "stroke_id": index,
                    "pen_state": "down",
                    "points": [{"x": x, "y": y} for x, y in stroke],
                }
                for index, stroke in enumerate(strokes)
            ],
        },
    }


def _write_trajectory_assets(
    record: dict[str, Any],
    diagram: Image.Image,
    repo_root: Path,
    runtime: dict[str, Any],
    export: dict[str, Any],
) -> dict[str, Any]:
    category_paths = _gold_category_paths(repo_root, runtime, record["category"])
    sample_id = record["sample_id"]
    svg_path = category_paths["svg"] / f"{sample_id}.svg"
    trajectory_path = category_paths["trajectories"] / f"{sample_id}.json"
    trajectory_info = _build_trajectory_payload(diagram, export)

    write_svg(
        svg_path,
        trajectory_info["strokes"],
        width=diagram.width,
        height=diagram.height,
        stroke_width=int(export["svg_stroke_width"]),
        stroke_color=str(export["svg_stroke_color"]),
    )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(json.dumps(trajectory_info["payload"], sort_keys=True, indent=2), encoding="utf-8")

    return {
        "svg_path": str(svg_path.relative_to(repo_root)),
        "trajectory_path": str(trajectory_path.relative_to(repo_root)),
        "stroke_count": trajectory_info["payload"]["stroke_count"],
        "point_count": trajectory_info["payload"]["point_count"],
        "trajectory": trajectory_info["payload"],
    }


def extract_trajectories(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    export = runtime["export"]
    existing_rows = {row["sample_id"]: row for row in read_ndjson(diagram_trajectories_manifest_path(repo_root, runtime))}
    trajectory_hash = _trajectory_config_hash(runtime)
    rows: list[dict[str, Any]] = []
    changed = set(existing_rows) != {row["sample_id"] for row in _selected_gold_candidates(repo_root, runtime)}
    for record in _selected_gold_candidates(repo_root, runtime):
        existing = existing_rows.get(record["sample_id"])
        if existing:
            lineage = existing.get("lineage", {})
            if (
                lineage.get("trajectory_source_hash") == _row_input_hash(record)
                and lineage.get("trajectory_config_hash") == trajectory_hash
                and _paths_exist(repo_root, existing.get("diagram_path"), existing.get("svg_path"), existing.get("trajectory_path"))
            ):
                rows.append(existing)
                continue
        changed = True
        category_paths = _gold_category_paths(repo_root, runtime, record["category"])
        diagram_path = resolve_repo_path(repo_root, record["diagram_path"])
        copied_diagram_path = category_paths["diagrams"] / f"{record['sample_id']}.png"
        _copy_diagram(diagram_path, copied_diagram_path)
        diagram = Image.open(diagram_path).convert("L")
        trajectory_assets = _write_trajectory_assets(record, diagram, repo_root, runtime, export)
        rows.append(
            {
                **record,
                "diagram_path": str(copied_diagram_path.relative_to(repo_root)),
                **trajectory_assets,
                "lineage": {
                    **record.get("lineage", {}),
                    "trajectory_source_hash": _row_input_hash(record),
                    "trajectory_config_hash": trajectory_hash,
                },
            }
        )
    if changed or not diagram_trajectories_manifest_path(repo_root, runtime).exists():
        write_ndjson(diagram_trajectories_manifest_path(repo_root, runtime), rows)
    return rows


def _gold_refs(final_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "diagram_path": final_record["diagram_path"],
        "svg_path": final_record["svg_path"],
        "trajectory_path": final_record["trajectory_path"],
    }


def _qa_scores(silver: dict[str, Any], validated_record: dict[str, Any], final_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "segmentation_score": silver["segmentation_score"],
        "view_confidence": silver["view_confidence"],
        "diagram_validation_score": validated_record["diagram_validation_score"],
        "stroke_count": final_record["stroke_count"],
        "point_count": final_record["point_count"],
    }


def _build_gold_record(
    sample_id: str,
    bronze: dict[str, Any],
    silver: dict[str, Any],
    validated_record: dict[str, Any],
    final_record: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    export = runtime["export"]
    return {
        "sample_id": sample_id,
        "task_type": runtime["datasets"]["task_type"],
        "category": bronze["category"],
        "subcategory": bronze["subcategory"],
        "tags": bronze["tags"],
        "source_dataset": bronze["source_dataset"],
        "source_refs": {
            "source_image_id": bronze["source_image_id"],
            "source_url": bronze["image_uri"],
            "mask_url": bronze["mask_uri"],
        },
        "bronze_refs": {
            "image_path": bronze["image_path"],
            "mask_path": bronze["mask_path"],
            "metadata_path": bronze["metadata_path"],
        },
        "silver_refs": {
            "image_path": silver["image_path"],
            "isolated_path": silver["isolated_path"],
            "crop_path": silver["crop_path"],
        },
        "gold_refs": _gold_refs(final_record),
        "angle_bucket": bronze["angle_bucket"],
        "angle_feasible": bronze["angle_feasible"],
        "label_source": bronze["label_source"],
        "label_confidence": bronze["label_confidence"],
        "diagram_attempt": validated_record["diagram_attempt"],
        "outline_generator_backend": validated_record["outline_generator_backend"],
        "outline_rectifier_backend": validated_record["outline_rectifier_backend"],
        "diagram_validation_status": validated_record["diagram_validation_status"],
        "diagram_validation_score": validated_record["diagram_validation_score"],
        "opencv_flags": validated_record["opencv_flags"],
        "outline_validator_model": validated_record["outline_validator_model"],
        "outline_validator_reason": validated_record["outline_validator_reason"],
        "selected_for_gold": True,
        "selection_rank": validated_record["selection_rank"],
        "qa_scores": _qa_scores(silver, validated_record, final_record),
        "lineage": {
            "ndjson_schema_version": export["ndjson_schema_version"],
            "bronze": bronze["lineage"],
            "silver": silver["lineage"],
            "diagram_validation": validated_record["lineage"],
        },
        "trajectory": final_record["trajectory"],
    }


def export_gold_ndjson(repo_root: Path, runtime: dict[str, Any]) -> list[dict[str, Any]]:
    bronze_records = _sample_lookup(bronze_accepted_manifest_path(repo_root, runtime))
    silver_records = _sample_lookup(silver_subjects_manifest_path(repo_root, runtime))
    validated = _sample_lookup(validated_diagrams_manifest_path(repo_root, runtime))
    extracted = _sample_lookup(diagram_trajectories_manifest_path(repo_root, runtime))
    output_path = gold_records_dir(repo_root, runtime) / "gold_samples.ndjson"
    existing_rows = {row["sample_id"]: row for row in read_ndjson(output_path)}
    export_hash = json.dumps({"export": runtime["export"], "datasets": runtime["datasets"]["task_type"]}, sort_keys=True)
    rows: list[dict[str, Any]] = []
    changed = set(existing_rows) != set(extracted)
    for sample_id, final_record in extracted.items():
        source_hash = json.dumps(
            {
                "bronze": bronze_records[sample_id],
                "silver": silver_records[sample_id],
                "validated": validated[sample_id],
                "trajectory": final_record,
            },
            sort_keys=True,
        )
        existing = existing_rows.get(sample_id)
        if existing and existing.get("lineage", {}).get("export_source_hash") == source_hash and existing.get("lineage", {}).get("export_config_hash") == export_hash:
            rows.append(existing)
            continue
        changed = True
        row = _build_gold_record(sample_id, bronze_records[sample_id], silver_records[sample_id], validated[sample_id], final_record, runtime)
        row["lineage"] = {**row["lineage"], "export_source_hash": source_hash, "export_config_hash": export_hash}
        rows.append(row)
    if changed or not output_path.exists():
        write_ndjson(output_path, rows)
    return rows


def _report_summary(
    gold_records: list[dict[str, Any]],
    curated: list[dict[str, Any]],
    validated: list[dict[str, Any]],
) -> dict[str, Any]:
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    by_subcategory: dict[str, Counter[str]] = defaultdict(Counter)
    by_angle: dict[str, Counter[str]] = defaultdict(Counter)
    for row in gold_records:
        categories[row["category"]]["selected"] += 1
        by_subcategory[row["category"]][row["subcategory"]] += 1
        by_angle[f"{row['category']}::{row['subcategory']}"][row["angle_bucket"]] += 1
    return {
        "generated_at": utc_now(),
        "bronze_curated": len(curated),
        "bronze_accepted": sum(1 for row in curated if row["curation_status"] == "accepted"),
        "validated_diagrams": len(validated),
        "accepted_diagrams": sum(1 for row in validated if row["diagram_validation_status"] == "accepted"),
        "total_records": len(gold_records),
        "categories": {key: dict(value) for key, value in categories.items()},
        "subcategories": {key: dict(value) for key, value in by_subcategory.items()},
        "angles": {key: dict(value) for key, value in by_angle.items()},
    }


def generate_dataset_report(repo_root: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    storage = runtime["storage"]
    gold_path = gold_records_dir(repo_root, runtime) / "gold_samples.ndjson"
    gold_records = read_ndjson(gold_path)
    curated = read_ndjson(bronze_curated_manifest_path(repo_root, runtime))
    validated = read_ndjson(validated_diagrams_manifest_path(repo_root, runtime))
    report_root = resolve_repo_path(repo_root, storage["reports_dir"])
    report_json = report_root / "dataset_report.json"
    source_hash = json.dumps({"gold": gold_records, "curated": curated, "validated": validated}, sort_keys=True)
    if report_json.exists():
        existing = json.loads(report_json.read_text(encoding="utf-8"))
        if existing.get("source_hash") == source_hash:
            return existing
    summary = _report_summary(gold_records, curated, validated)
    report_root.mkdir(parents=True, exist_ok=True)
    summary["source_hash"] = source_hash
    (report_root / "dataset_report.json").write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    markdown_lines = [
        "# Dataset Report",
        f"- Generated: {summary['generated_at']}",
        f"- Bronze curated: {summary['bronze_curated']}",
        f"- Bronze accepted: {summary['bronze_accepted']}",
        f"- Validated diagrams: {summary['validated_diagrams']}",
        f"- Accepted diagrams: {summary['accepted_diagrams']}",
        f"- Final gold samples: {summary['total_records']}",
    ]
    for category, counts in summary["categories"].items():
        markdown_lines.append(f"- {category}: {counts}")
    (report_root / "dataset_report.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return summary
