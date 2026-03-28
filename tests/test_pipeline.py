from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from wildtrace.io_utils import read_ndjson


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_test_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    for rel in [
        "configs",
        "raw_data/bronze/openimages/images",
        "raw_data/bronze/openimages/masks",
        "raw_data/bronze/openimages/metadata",
        "raw_data/bronze/openimages/logs",
        "raw_data/bronze/manifests",
        "processed/silver/images",
        "processed/silver/masks",
        "processed/silver/isolated",
        "processed/silver/qa",
        "processed/gold/outlines",
        "processed/gold/svg",
        "processed/gold/trajectories",
        "processed/gold/ndjson",
        "artifacts/reports",
        "fixtures",
    ]:
        (repo_root / rel).mkdir(parents=True, exist_ok=True)

    image_path = repo_root / "fixtures" / "cat.png"
    mask_path = repo_root / "fixtures" / "cat_mask.png"
    image = Image.new("RGB", (400, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 60, 320, 260), fill="black")
    image.save(image_path)
    mask = Image.new("L", (400, 320), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((60, 60, 320, 260), fill=255)
    mask.save(mask_path)

    records_path = repo_root / "fixtures" / "records.ndjson"
    record = {
        "source_image_id": "cat-001",
        "category": "Cat",
        "split": "train",
        "source_url": image_path.resolve().as_uri(),
        "mask_url": mask_path.resolve().as_uri(),
        "license": "CC-BY-4.0",
    }
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    write_yaml(
        repo_root / "configs" / "datasets.yaml",
        {
            "task_type": "drawing",
            "source_dataset": "openimages",
            "categories": [{"name": "Cat", "limit": 10}],
            "fetch": {
                "source_mode": "records",
                "records_path": "fixtures/records.ndjson",
                "records": [],
                "timeout_seconds": 5,
                "user_agent": "WildTraceTest/0.1",
                "manifest_path": "raw_data/bronze/manifests/openimages_fetch.ndjson",
                "latest_view_path": "raw_data/bronze/manifests/openimages_fetch_latest.ndjson",
                "official": {
                    "splits": ["train"],
                    "require_masks": True,
                    "class_descriptions_url": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
                    "image_base_url_template": "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg",
                    "image_info_urls": {"train": "unused"},
                    "segmentation_annotation_urls": {"train": "unused"},
                    "mask_zip_url_templates": {"train": "unused-{prefix}"},
                },
            },
            "ingest": {
                "manifest_path": "raw_data/bronze/manifests/openimages_ingest.ndjson",
                "latest_view_path": "raw_data/bronze/manifests/openimages_ingest_latest.ndjson",
            },
        },
    )
    write_yaml(
        repo_root / "configs" / "storage.yaml",
        {
            "raw_root": "raw_data",
            "processed_root": "processed",
            "artifacts_root": "artifacts",
            "bronze": {
                "images_dir": "raw_data/bronze/openimages/images",
                "masks_dir": "raw_data/bronze/openimages/masks",
                "metadata_dir": "raw_data/bronze/openimages/metadata",
                "logs_dir": "raw_data/bronze/openimages/logs",
                "manifests_dir": "raw_data/bronze/manifests",
            },
            "silver": {
                "images_dir": "processed/silver/images",
                "masks_dir": "processed/silver/masks",
                "isolated_dir": "processed/silver/isolated",
                "qa_dir": "processed/silver/qa",
            },
            "gold": {
                "outlines_dir": "processed/gold/outlines",
                "svg_dir": "processed/gold/svg",
                "trajectories_dir": "processed/gold/trajectories",
                "ndjson_dir": "processed/gold/ndjson",
            },
            "reports_dir": "artifacts/reports",
        },
    )
    write_yaml(
        repo_root / "configs" / "quality.yaml",
        {
            "minimum_width": 256,
            "minimum_height": 256,
            "max_resize": 1024,
            "low_mask_coverage_threshold": 0.05,
            "extreme_aspect_ratio_threshold": 4.0,
            "blur_variance_threshold": 0.00001,
            "allow_flagged_to_continue": True,
        },
    )
    write_yaml(
        repo_root / "configs" / "models.yaml",
        {
            "active_backend": "local_edges",
            "backends": {
                "local_edges": {
                    "model_backend": "local",
                    "model_name": "local_edges",
                    "model_version_or_checkpoint": "baseline-v1",
                    "threshold_percentile": 70,
                }
            },
        },
    )
    write_yaml(
        repo_root / "configs" / "export.yaml",
        {
            "ndjson_schema_version": 1,
            "max_points_per_stroke": 64,
            "max_strokes": 8,
            "min_points_per_stroke": 8,
            "svg_stroke_color": "#000000",
            "svg_stroke_width": 2,
            "simplification_method": "angular_subsample",
            "coordinate_frame": "normalized_canvas",
        },
    )
    return repo_root


def run_script(project_root: Path, repo_root: Path, script_name: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / script_name),
            "--repo-root",
            str(repo_root),
            "--config-dir",
            str(repo_root / "configs"),
        ],
        cwd=project_root,
        check=True,
    )


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = build_test_repo(tmp_path)
    for script_name in [
        "fetch_openimages.py",
        "ingest_openimages.py",
        "validate_bronze.py",
        "normalize_to_silver.py",
        "run_outline_inference.py",
        "refine_outlines.py",
        "export_gold_ndjson.py",
        "generate_dataset_report.py",
    ]:
        run_script(project_root, repo_root, script_name)

    gold_path = repo_root / "processed/gold/ndjson/gold_samples.ndjson"
    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    record = records[0]
    assert record["task_type"] == "drawing"
    assert record["category"] == "Cat"
    assert record["trajectory"]["stroke_count"] >= 1
    assert Path(repo_root / record["gold_refs"]["svg_path"]).exists()
    assert Path(repo_root / "artifacts/reports/dataset_report.md").exists()


def test_fetch_and_ingest_are_incremental_and_versioned(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = build_test_repo(tmp_path)

    run_script(project_root, repo_root, "fetch_openimages.py")
    run_script(project_root, repo_root, "ingest_openimages.py")

    fetch_ledger = repo_root / "raw_data/bronze/manifests/openimages_fetch.ndjson"
    ingest_ledger = repo_root / "raw_data/bronze/manifests/openimages_ingest.ndjson"
    assert len(read_ndjson(fetch_ledger)) == 1
    assert len(read_ndjson(ingest_ledger)) == 1

    run_script(project_root, repo_root, "fetch_openimages.py")
    run_script(project_root, repo_root, "ingest_openimages.py")
    assert len(read_ndjson(fetch_ledger)) == 1
    assert len(read_ndjson(ingest_ledger)) == 1

    image_path = repo_root / "fixtures" / "cat.png"
    image = Image.new("RGB", (400, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 360, 280), fill="black")
    image.save(image_path)

    run_script(project_root, repo_root, "fetch_openimages.py")
    run_script(project_root, repo_root, "ingest_openimages.py")

    fetch_records = read_ndjson(fetch_ledger)
    ingest_records = read_ndjson(ingest_ledger)
    assert len(fetch_records) == 2
    assert len(ingest_records) == 2
    assert fetch_records[-1]["asset_version"] == 2
    assert ingest_records[-1]["asset_version"] == 2
    assert fetch_records[0]["image_checksum"] != fetch_records[-1]["image_checksum"]
