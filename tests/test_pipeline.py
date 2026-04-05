from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from wildtrace.agentic_pipeline import select_final_by_angle
from wildtrace.config import load_runtime_config
from wildtrace.diagram import (
    FluxSilhouetteRectifier,
    OllamaSemanticValidator,
    OpenCVValidationResult,
    OutlineRectifier,
    SemanticValidationResult,
    SilhouetteOutlineRectifier,
    build_outline_rectifier,
    build_outline_validator,
    run_langgraph_validation_loop,
    validate_with_opencv,
)
from wildtrace.io_utils import read_ndjson
from wildtrace.pipeline import reconcile_fetch_storage_stage


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_test_repo(tmp_path: Path, records: list[dict] | None = None) -> Path:
    repo_root = tmp_path / "repo"
    for rel in [
        "configs",
        "outputs/bronze/images",
        "outputs/bronze/masks",
        "outputs/bronze/metadata",
        "outputs/bronze/logs",
        "outputs/bronze/manifests",
        "outputs/silver/images",
        "outputs/silver/masks",
        "outputs/silver/isolated_subjects",
        "outputs/silver/crops",
        "outputs/silver/line_diagrams",
        "outputs/silver/checkpoints",
        "outputs/gold/diagrams",
        "outputs/gold/outlines",
        "outputs/gold/svg",
        "outputs/gold/trajectories",
        "outputs/gold/records",
        "outputs/reports",
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
    draw_mask.ellipse((100, 30, 156, 90), fill=255)
    mask.save(mask_path)

    records_path = repo_root / "fixtures" / "records.ndjson"
    if records is None:
        records = [
            {
                "source_image_id": "cat-001",
                "category": "Cat",
                "split": "train",
                "source_url": image_path.resolve().as_uri(),
                "mask_url": mask_path.resolve().as_uri(),
                "license": "CC-BY-4.0",
                "title": "Domestic cat front view",
            }
        ]
    records_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

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
                "manifest_path": "outputs/bronze/manifests/openimages_fetch.ndjson",
                "latest_view_path": "outputs/bronze/manifests/openimages_fetch_latest.ndjson",
                "official": {
                    "splits": ["train"],
                    "require_masks": True,
                    "class_descriptions_url": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
                    "image_base_url_template": "unused",
                    "image_info_urls": {"train": "unused"},
                    "segmentation_annotation_urls": {"train": "unused"},
                    "mask_zip_url_templates": {"train": "unused-{prefix}"},
                },
            },
            "ingest": {
                "manifest_path": "outputs/bronze/manifests/openimages_ingest.ndjson",
                "latest_view_path": "outputs/bronze/manifests/openimages_ingest_latest.ndjson",
            },
        },
    )
    write_yaml(
        repo_root / "configs" / "storage.yaml",
        {
            "raw_root": "outputs",
            "processed_root": "outputs",
            "artifacts_root": "outputs",
            "bronze": {
                "images_dir": "outputs/bronze/images",
                "masks_dir": "outputs/bronze/masks",
                "metadata_dir": "outputs/bronze/metadata",
                "logs_dir": "outputs/bronze/logs",
                "manifests_dir": "outputs/bronze/manifests",
            },
            "silver": {
                "images_dir": "outputs/silver/images",
                "masks_dir": "outputs/silver/masks",
                "isolated_dir": "outputs/silver/isolated_subjects",
                "crops_dir": "outputs/silver/crops",
                "diagrams_dir": "outputs/silver/line_diagrams",
                "checkpoints_dir": "outputs/silver/checkpoints",
            },
            "gold": {
                "diagrams_dir": "outputs/gold/diagrams",
                "outlines_dir": "outputs/gold/outlines",
                "svg_dir": "outputs/gold/svg",
                "trajectories_dir": "outputs/gold/trajectories",
                "records_dir": "outputs/gold/records",
            },
            "reports_dir": "outputs/reports",
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
            "minimum_segmentation_score": 0.45,
        },
    )
    write_yaml(
        repo_root / "configs" / "models.yaml",
        {
            "enrichment": {
                "backend": "heuristic_enrichment",
                "model_backend": "local",
                "model_name": "heuristic_enrichment",
                "model_id": "heuristic_enrichment",
                "default_confidence": 0.72,
            },
            "outline_rectifier": {
                "backend": "silhouette_outline_rectifier",
                "model_backend": "local",
                "model_name": "silhouette_outline_rectifier",
                "model_version_or_checkpoint": "local-v1",
                "max_attempts": 5,
                "subject_threshold": 248,
                "silhouette_close_kernel": 9,
                "outline_close_kernel": 5,
                "outline_simplify_ratio": 0.012,
                "outline_stroke_width": 3,
                "min_outline_area": 300,
                "max_outlines": 2,
                "min_component_area": 80,
            },
            "outline_validator": {
                "backend": "mock_semantic_validator",
                "model_backend": "mock",
                "model_name": "mock_semantic_validator",
                "ollama_model_name": "qwen2.5vl:7b",
                "ollama_model_id": "qwen2.5vl:7b",
                "anthropic_model_name": "claude-haiku-4-5",
                "anthropic_api_base": "https://api.anthropic.com",
                "min_score": 0.55,
            },
            "opencv_prescreen": {
                "min_foreground_ratio": 0.03,
                "max_foreground_ratio": 0.32,
                "target_foreground_ratio": 0.16,
                "max_component_count": 18,
                "max_small_contour_ratio": 0.70,
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
    write_yaml(
        repo_root / "configs" / "viewpoints.yaml",
        {
            "allowed_buckets": [
                "left_profile",
                "right_profile",
                "front_left_3q",
                "front_right_3q",
                "front",
            ],
            "rejected_buckets": ["rear", "top_down", "occluded", "unknown"],
            "prefilter": {
                "require_mask": True,
                "min_mask_coverage_ratio": 0.08,
                "max_border_touch_ratio": 0.50,
                "max_component_count": 4,
                "min_largest_component_ratio": 0.75,
                "min_bbox_fill_ratio": 0.10,
            },
            "classifier": {
                "backend": "local_score_v1",
                "direction_deadzone": 0.05,
                "front_symmetry_bias": 0.72,
                "three_quarter_symmetry_target": 0.58,
                "three_quarter_symmetry_tolerance": 0.25,
            },
            "min_confidence": 0.42,
            "min_margin": 0.08,
            "target_per_bucket": {"default": 1},
        },
    )
    return repo_root


def run_script(project_root: Path, repo_root: Path, script_name: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
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
        env=env,
    )


def test_default_flux_config_matches_repo_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runtime = load_runtime_config(project_root, project_root / "configs")
    cfg = runtime["models"]["outline_rectifier"]
    assert cfg["backend"] == "flux_silhouette_rectifier"
    assert "gguf_repo" in cfg
    assert "base_model" in cfg
    assert float(cfg["strength"]) >= 0.70


def test_end_to_end_agentic_pipeline(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = build_test_repo(tmp_path)
    for script_name in [
        "fetch_openimages.py",
        "curate_bronze.py",
        "normalize_to_silver.py",
        "enrich_and_crop_subjects.py",
        "generate_line_diagrams.py",
        "validate_and_retry_diagrams.py",
        "select_final_by_angle.py",
        "extract_trajectories.py",
        "export_gold_ndjson.py",
        "generate_dataset_report.py",
    ]:
        run_script(project_root, repo_root, script_name)

    gold_path = repo_root / "outputs/gold/records/gold_samples.ndjson"
    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "Cat"
    assert record["subcategory"] == "domestic_cat"
    assert record["selected_for_gold"] is True
    assert record["diagram_validation_status"] == "accepted"
    assert record["trajectory"]["stroke_count"] >= 1
    assert Path(repo_root / record["gold_refs"]["diagram_path"]).exists()
    assert Path(repo_root / "outputs/reports/dataset_report.md").exists()


def test_curate_bronze_removes_duplicate_raws(tmp_path: Path) -> None:
    duplicate_records = [
        {
            "source_image_id": "cat-001",
            "category": "Cat",
            "split": "train",
            "source_url": (tmp_path / "repo" / "fixtures" / "cat.png").resolve().as_uri(),
            "mask_url": (tmp_path / "repo" / "fixtures" / "cat_mask.png").resolve().as_uri(),
            "license": "CC-BY-4.0",
            "title": "Domestic cat front view",
        },
        {
            "source_image_id": "cat-002",
            "category": "Cat",
            "split": "train",
            "source_url": (tmp_path / "repo" / "fixtures" / "cat.png").resolve().as_uri(),
            "mask_url": (tmp_path / "repo" / "fixtures" / "cat_mask.png").resolve().as_uri(),
            "license": "CC-BY-4.0",
            "title": "Domestic cat duplicate",
        },
    ]
    repo_root = build_test_repo(tmp_path, records=duplicate_records)
    project_root = Path(__file__).resolve().parents[1]
    run_script(project_root, repo_root, "fetch_openimages.py")
    run_script(project_root, repo_root, "curate_bronze.py")
    curated = read_ndjson(repo_root / "outputs/bronze/manifests/bronze_curated.ndjson")
    accepted = read_ndjson(repo_root / "outputs/bronze/manifests/bronze_accepted.ndjson")
    assert len(curated) == 2
    assert len(accepted) == 1
    rejected = [row for row in curated if row["curation_status"] == "rejected"]
    assert rejected
    assert "duplicate_image_checksum" in rejected[0]["curation_flags"]
    assert rejected[0]["deleted_paths"]


def test_fetch_reuses_current_version_when_latest_version_is_missing(tmp_path: Path) -> None:
    repo_root = build_test_repo(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    run_script(project_root, repo_root, "fetch_openimages.py")

    ledger_path = repo_root / "outputs/bronze/manifests/openimages_fetch.ndjson"
    latest_view_path = repo_root / "outputs/bronze/manifests/openimages_fetch_latest.ndjson"
    rows = read_ndjson(ledger_path)
    assert len(rows) == 1
    first = rows[0]

    image_v1 = repo_root / first["image_path"]
    mask_v1 = repo_root / first["mask_path"]
    metadata_v1 = repo_root / first["metadata_path"]
    image_v2 = image_v1.with_name("v0002.jpg")
    mask_v2 = mask_v1.with_name("v0002.png")
    metadata_v2 = metadata_v1.with_name("v0002.json")

    second = {
        **first,
        "record_id": "duplicate-v2",
        "asset_version": 2,
        "image_path": str(image_v2.relative_to(repo_root)),
        "mask_path": str(mask_v2.relative_to(repo_root)),
        "metadata_path": str(metadata_v2.relative_to(repo_root)),
        "image_uri": f"local://{image_v2.relative_to(repo_root).as_posix()}",
        "mask_uri": f"local://{mask_v2.relative_to(repo_root).as_posix()}",
    }
    ledger_path.write_text(
        "\n".join([json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)]) + "\n",
        encoding="utf-8",
    )

    run_script(project_root, repo_root, "fetch_openimages.py")

    rows_after = read_ndjson(ledger_path)
    assert len(rows_after) == 2
    latest_view = read_ndjson(latest_view_path)
    assert len(latest_view) == 1
    assert latest_view[0]["asset_version"] == 1
    assert latest_view[0]["image_path"] == first["image_path"]
    assert image_v1.exists()
    assert not image_v2.exists()


def test_fetch_cleans_identical_duplicate_versions_in_same_sample_folder(tmp_path: Path) -> None:
    repo_root = build_test_repo(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    run_script(project_root, repo_root, "fetch_openimages.py")

    ledger_path = repo_root / "outputs/bronze/manifests/openimages_fetch.ndjson"
    latest_view_path = repo_root / "outputs/bronze/manifests/openimages_fetch_latest.ndjson"
    first = read_ndjson(ledger_path)[0]

    image_v1 = repo_root / first["image_path"]
    mask_v1 = repo_root / first["mask_path"]
    metadata_v1 = repo_root / first["metadata_path"]
    image_v2 = image_v1.with_name("v0002.jpg")
    mask_v2 = mask_v1.with_name("v0002.png")
    metadata_v2 = metadata_v1.with_name("v0002.json")
    image_v2.write_bytes(image_v1.read_bytes())
    mask_v2.write_bytes(mask_v1.read_bytes())
    metadata_v2.write_text(metadata_v1.read_text(encoding="utf-8"), encoding="utf-8")

    second = {
        **first,
        "record_id": "duplicate-v2",
        "asset_version": 2,
        "image_path": str(image_v2.relative_to(repo_root)),
        "mask_path": str(mask_v2.relative_to(repo_root)),
        "metadata_path": str(metadata_v2.relative_to(repo_root)),
        "image_uri": f"local://{image_v2.relative_to(repo_root).as_posix()}",
        "mask_uri": f"local://{mask_v2.relative_to(repo_root).as_posix()}",
    }
    ledger_path.write_text(
        "\n".join([json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)]) + "\n",
        encoding="utf-8",
    )

    run_script(project_root, repo_root, "fetch_openimages.py")

    rows_after = read_ndjson(ledger_path)
    assert len(rows_after) == 2
    latest_view = read_ndjson(latest_view_path)
    assert len(latest_view) == 1
    assert latest_view[0]["asset_version"] == 2
    assert latest_view[0]["image_path"] == second["image_path"]
    assert not image_v1.exists()
    assert image_v2.exists()


def test_reconcile_fetch_storage_stage_cleans_duplicate_current_versions(tmp_path: Path) -> None:
    repo_root = build_test_repo(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    run_script(project_root, repo_root, "fetch_openimages.py")

    ledger_path = repo_root / "outputs/bronze/manifests/openimages_fetch.ndjson"
    first = read_ndjson(ledger_path)[0]
    image_v1 = repo_root / first["image_path"]
    mask_v1 = repo_root / first["mask_path"]
    metadata_v1 = repo_root / first["metadata_path"]
    image_v2 = image_v1.with_name("v0002.jpg")
    mask_v2 = mask_v1.with_name("v0002.png")
    metadata_v2 = metadata_v1.with_name("v0002.json")
    image_v2.write_bytes(image_v1.read_bytes())
    mask_v2.write_bytes(mask_v1.read_bytes())
    metadata_v2.write_text(metadata_v1.read_text(encoding="utf-8"), encoding="utf-8")

    second = {
        **first,
        "record_id": "duplicate-v2",
        "asset_version": 2,
        "image_path": str(image_v2.relative_to(repo_root)),
        "mask_path": str(mask_v2.relative_to(repo_root)),
        "metadata_path": str(metadata_v2.relative_to(repo_root)),
        "image_uri": f"local://{image_v2.relative_to(repo_root).as_posix()}",
        "mask_uri": f"local://{mask_v2.relative_to(repo_root).as_posix()}",
    }
    ledger_path.write_text(
        "\n".join([json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)]) + "\n",
        encoding="utf-8",
    )

    runtime = load_runtime_config(repo_root, repo_root / "configs")
    current_rows = reconcile_fetch_storage_stage(repo_root, runtime)

    assert len(current_rows) == 1
    assert current_rows[0]["asset_version"] == 2
    assert not image_v1.exists()
    assert image_v2.exists()


def test_reconcile_fetch_storage_stage_repairs_legacy_bronze_paths(tmp_path: Path) -> None:
    repo_root = build_test_repo(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    run_script(project_root, repo_root, "fetch_openimages.py")

    ledger_path = repo_root / "outputs/bronze/manifests/openimages_fetch.ndjson"
    latest_view_path = repo_root / "outputs/bronze/manifests/openimages_fetch_latest.ndjson"
    first = read_ndjson(ledger_path)[0]
    image_path = repo_root / first["image_path"]
    mask_path = repo_root / first["mask_path"]
    metadata_path = repo_root / first["metadata_path"]

    legacy = {
        **first,
        "image_path": f"raw_data/bronze/openimages/images/{first['category']}/{first['sample_id']}/{image_path.name}",
        "mask_path": f"raw_data/bronze/openimages/masks/{first['category']}/{first['sample_id']}/{mask_path.name}",
        "metadata_path": f"raw_data/bronze/openimages/metadata/{first['category']}/{first['sample_id']}/{metadata_path.name}",
        "image_uri": f"local://raw_data/bronze/openimages/images/{first['category']}/{first['sample_id']}/{image_path.name}",
        "mask_uri": f"local://raw_data/bronze/openimages/masks/{first['category']}/{first['sample_id']}/{mask_path.name}",
    }
    ledger_path.write_text(json.dumps(legacy, sort_keys=True) + "\n", encoding="utf-8")

    runtime = load_runtime_config(repo_root, repo_root / "configs")
    current_rows = reconcile_fetch_storage_stage(repo_root, runtime)

    repaired = read_ndjson(ledger_path)[0]
    assert repaired["image_path"] == first["image_path"]
    assert repaired["mask_path"] == first["mask_path"]
    assert repaired["metadata_path"] == first["metadata_path"]
    assert current_rows[0]["image_path"] == first["image_path"]
    assert read_ndjson(latest_view_path)[0]["image_path"] == first["image_path"]


def test_opencv_prescreen_blocks_noisy_diagrams(tmp_path: Path) -> None:
    noisy_path = tmp_path / "noisy.png"
    image = Image.new("L", (128, 128), 0)
    draw = ImageDraw.Draw(image)
    for x in range(0, 128, 8):
        for y in range(0, 128, 8):
            draw.rectangle((x, y, x + 1, y + 1), fill=255)
    image.save(noisy_path)
    result = validate_with_opencv(
        noisy_path,
        {
            "min_foreground_ratio": 0.03,
            "max_foreground_ratio": 0.32,
            "target_foreground_ratio": 0.16,
            "max_component_count": 10,
            "max_small_contour_ratio": 0.4,
        },
    )
    assert result.passed is False
    assert result.flags


def test_semantic_validator_defaults_to_ollama_without_anthropic_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(OllamaSemanticValidator, "validate_ready", lambda self: None)
    validator = build_outline_validator(
        {
            "backend": "auto",
            "ollama_model_name": "qwen2.5vl:7b",
            "ollama_model_id": "qwen2.5vl:7b",
            "anthropic_model_name": "claude-haiku-4-5",
            "min_score": 0.55,
        }
    )
    assert isinstance(validator, OllamaSemanticValidator)


def test_ollama_validator_validate_ready_fails_when_host_is_down() -> None:
    validator = OllamaSemanticValidator({"host": "http://127.0.0.1:9"})
    with pytest.raises(RuntimeError, match="Ollama validator is not ready"):
        validator.validate_ready()


def test_build_outline_rectifier_supports_silhouette() -> None:
    rectifier = build_outline_rectifier({"backend": "silhouette_outline_rectifier"})
    assert isinstance(rectifier, SilhouetteOutlineRectifier)


def test_build_outline_rectifier_supports_flux(monkeypatch) -> None:
    monkeypatch.setattr(FluxSilhouetteRectifier, "validate_ready", lambda self: None)
    rectifier = build_outline_rectifier(
        {
            "backend": "flux_silhouette_rectifier",
            "gguf_repo": "city96/FLUX.1-schnell-gguf",
            "gguf_file": "flux1-schnell-Q4_K_S.gguf",
            "base_model": "black-forest-labs/FLUX.1-schnell",
        }
    )
    assert isinstance(rectifier, FluxSilhouetteRectifier)


def test_flux_rectifier_writes_output_and_metadata(monkeypatch, tmp_path: Path) -> None:
    subject_image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(subject_image)
    draw.ellipse((16, 16, 112, 112), fill="black")
    destination = tmp_path / "diagram.png"

    rectifier = FluxSilhouetteRectifier(
        {
            "gguf_repo": "city96/FLUX.1-schnell-gguf",
            "gguf_file": "flux1-schnell-Q4_K_S.gguf",
            "base_model": "black-forest-labs/FLUX.1-schnell",
            "strength": 0.90,
            "min_component_area": 20,
            "subject_threshold": 248,
            "silhouette_close_kernel": 9,
        }
    )

    class FakeFluxPipeline:
        def __call__(self, **kwargs):
            out = Image.new("RGB", (512, 512), "white")
            draw = ImageDraw.Draw(out)
            draw.ellipse((40, 40, 470, 470), outline="black", width=4)
            return type("FakeResult", (), {"images": [out]})()

    monkeypatch.setattr(rectifier, "_load_pipeline", lambda: FakeFluxPipeline())
    result = rectifier.run(
        {"sample_id": "test-1", "category": "Dog"},
        subject_image,
        None,
        None,
        destination,
        {},
    )
    assert destination.exists()
    assert result.metadata["backend"] == "flux_silhouette_rectifier"
    assert result.metadata["prompt"]
    assert Image.open(result.diagram_path).size == subject_image.size


def test_flux_rectifier_output_is_black_on_white(monkeypatch, tmp_path: Path) -> None:
    subject_image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(subject_image)
    draw.ellipse((4, 4, 60, 60), fill="black")
    destination = tmp_path / "diagram.png"

    rectifier = FluxSilhouetteRectifier({"min_component_area": 5, "subject_threshold": 248})

    class FakeFluxPipeline:
        def __call__(self, **kwargs):
            out = Image.new("RGB", (512, 512), "white")
            ImageDraw.Draw(out).ellipse((60, 60, 450, 450), outline="black", width=6)
            return type("FakeResult", (), {"images": [out]})()

    monkeypatch.setattr(rectifier, "_load_pipeline", lambda: FakeFluxPipeline())
    result = rectifier.run({"category": "Cat"}, subject_image, None, None, destination, {})
    arr = np.asarray(Image.open(result.diagram_path).convert("L"))
    assert arr.max() == 255, "background must be white"
    assert arr.min() == 0, "lines must be pure black"


def test_ollama_validator_skips_failed_opencv(tmp_path: Path) -> None:
    diagram_path = tmp_path / "diagram.png"
    Image.new("L", (16, 16), 0).save(diagram_path)
    validator = OllamaSemanticValidator({"ollama_model_name": "qwen2.5vl:7b"})
    result = validator.validate(
        {"category": "Bird", "subcategory": "hummingbird", "angle_bucket": "front"},
        diagram_path,
        OpenCVValidationResult(
            passed=False,
            score=0.12,
            flags=["too_many_components"],
            metrics={},
        ),
    )
    assert result.passed is False
    assert result.score < 0.01
    assert "OpenCV pre-screen failed" in result.reason
    assert result.metadata["skipped"] is True


def test_ollama_validator_uses_model_response(monkeypatch, tmp_path: Path) -> None:
    diagram_path = tmp_path / "diagram.png"
    Image.new("L", (16, 16), 255).save(diagram_path)
    validator = OllamaSemanticValidator({"ollama_model_name": "qwen2.5vl:7b", "min_score": 0.55})
    monkeypatch.setattr(
        validator,
        "_request",
        lambda _diagram_path, _prompt: {"response": '{"passed": true, "score": 0.84, "reason": "clean and drawable"}'},
    )
    result = validator.validate(
        {"category": "Bird", "subcategory": "hummingbird", "angle_bucket": "front"},
        diagram_path,
        OpenCVValidationResult(
            passed=True,
            score=0.81,
            flags=[],
            metrics={},
        ),
    )
    assert result.passed is True
    assert abs(result.score - 0.84) < 0.01
    assert result.reason == "clean and drawable"
    assert result.metadata["model_name"] == "qwen2.5vl:7b"


def test_ollama_validator_prompt_is_outline_only() -> None:
    validator = OllamaSemanticValidator({"ollama_model_name": "qwen2.5vl:7b"})
    prompt = validator._build_prompt(  # noqa: SLF001
        {"category": "Bird", "subcategory": "hummingbird", "angle_bucket": "front"},
        OpenCVValidationResult(passed=True, score=0.8, flags=[], metrics={}),
    )
    assert "specific species" in prompt
    assert "hummingbird" not in prompt
    assert "Bird" not in prompt
    assert "black ink strokes" in prompt


def test_run_langgraph_validation_loop_retries_and_preserves_state(tmp_path: Path) -> None:
    subject_path = tmp_path / "subject.png"
    Image.new("RGB", (64, 64), "white").save(subject_path)

    class StubDiagramGenerator(OutlineRectifier):
        call_count = 0

        def __init__(self) -> None:
            super().__init__({})

        def run(self, _sample, subject_image, _subject_mask, _generated_path, destination, _params):
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.call_count += 1
            diagram = Image.new("L", subject_image.size, 255)
            # Succeed only on the second attempt
            if self.call_count >= 2:
                draw = ImageDraw.Draw(diagram)
                draw.ellipse((18, 20, 46, 42), outline=0, width=3)
                draw.line((46, 31, 56, 28), fill=0, width=2)
            diagram.save(destination)
            return type(
                "GeneratedResult",
                (),
                {"diagram_path": destination, "metadata": {"model_name": "stub-diagram-generator"}},
            )()

    class StubValidator:
        def validate(self, _sample, diagram_path, _opencv_result):
            arr = np.asarray(Image.open(diagram_path).convert("L"))
            passed = bool((arr < 128).any())  # has some dark line pixels
            return SemanticValidationResult(
                passed=passed,
                score=0.9 if passed else 0.2,
                reason="ok" if passed else "retry",
                metadata={"model_name": "stub-validator"},
            )

    result = run_langgraph_validation_loop(
        sample={"sample_id": "bird-1", "category": "Bird", "subcategory": "hummingbird", "angle_bucket": "front"},
        subject_path=subject_path,
        subject_mask_path=None,
        diagram_root=tmp_path / "diagrams",
        diagram_generator=StubDiagramGenerator(),
        outline_validator=StubValidator(),
        opencv_settings={
            "min_foreground_ratio": 0.03,
            "max_foreground_ratio": 0.32,
            "target_foreground_ratio": 0.16,
            "max_component_count": 18,
            "max_small_contour_ratio": 0.70,
        },
        initial_params={"strength": 0.90},
        max_attempts=2,
    )

    assert result["diagram_validation_status"] == "accepted"
    assert result["diagram_attempt"] == 2
    assert result["outline_validator_model"] == "stub-validator"
    assert result["diagram_generator_backend"]["model_name"] == "stub-diagram-generator"


def test_select_final_by_angle_keeps_best_score_per_bucket(tmp_path: Path) -> None:
    repo_root = build_test_repo(tmp_path)
    validated_path = repo_root / "outputs/silver/checkpoints/validated_diagrams.ndjson"
    rows = [
        {
            "sample_id": "a",
            "category": "Bird",
            "subcategory": "flamingo",
            "angle_bucket": "front",
            "diagram_path": "outputs/silver/line_diagrams/Bird/a.png",
            "diagram_attempt": 1,
            "diagram_generator_backend": {"model_name": "flux_silhouette_rectifier"},
            "diagram_validation_status": "accepted",
            "diagram_validation_score": 0.61,
            "opencv_flags": [],
            "outline_validator_model": "qwen2.5vl:7b",
            "outline_validator_reason": "ok",
            "selected_for_gold": False,
            "selection_rank": None,
            "lineage": {},
        },
        {
            "sample_id": "b",
            "category": "Bird",
            "subcategory": "flamingo",
            "angle_bucket": "front",
            "diagram_path": "outputs/silver/line_diagrams/Bird/b.png",
            "diagram_attempt": 2,
            "diagram_generator_backend": {"model_name": "flux_silhouette_rectifier"},
            "diagram_validation_status": "accepted",
            "diagram_validation_score": 0.83,
            "opencv_flags": [],
            "outline_validator_model": "qwen2.5vl:7b",
            "outline_validator_reason": "better",
            "selected_for_gold": False,
            "selection_rank": None,
            "lineage": {},
        },
        {
            "sample_id": "c",
            "category": "Bird",
            "subcategory": "flamingo",
            "angle_bucket": "left_profile",
            "diagram_path": "outputs/silver/line_diagrams/Bird/c.png",
            "diagram_attempt": 1,
            "diagram_generator_backend": {"model_name": "flux_silhouette_rectifier"},
            "diagram_validation_status": "accepted",
            "diagram_validation_score": 0.71,
            "opencv_flags": [],
            "outline_validator_model": "qwen2.5vl:7b",
            "outline_validator_reason": "ok",
            "selected_for_gold": False,
            "selection_rank": None,
            "lineage": {},
        },
    ]
    validated_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    runtime = load_runtime_config(repo_root, repo_root / "configs")
    selected = select_final_by_angle(repo_root, runtime)
    front_rows = [row for row in selected if row["angle_bucket"] == "front"]
    assert len(front_rows) == 2
    chosen = [row for row in front_rows if row["selected_for_gold"]]
    assert len(chosen) == 1
    assert chosen[0]["sample_id"] == "b"


def test_silver_and_gold_reruns_reuse_existing_outputs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = build_test_repo(tmp_path)
    initial_scripts = [
        "fetch_openimages.py",
        "curate_bronze.py",
        "normalize_to_silver.py",
        "enrich_and_crop_subjects.py",
        "generate_line_diagrams.py",
        "validate_and_retry_diagrams.py",
        "select_final_by_angle.py",
        "extract_trajectories.py",
        "export_gold_ndjson.py",
        "generate_dataset_report.py",
    ]
    for script_name in initial_scripts:
        run_script(project_root, repo_root, script_name)

    tracked = [
        repo_root / "outputs/silver/checkpoints/line_diagram_attempts.ndjson",
        repo_root / "outputs/silver/checkpoints/validated_diagrams.ndjson",
        repo_root / "outputs/silver/checkpoints/selected_diagrams.ndjson",
        repo_root / "outputs/gold/trajectories/diagram_trajectories.ndjson",
        repo_root / "outputs/gold/records/gold_samples.ndjson",
        repo_root / "outputs/reports/dataset_report.json",
    ]
    before = {path: path.stat().st_mtime_ns for path in tracked}
    time.sleep(0.01)

    for script_name in [
        "generate_line_diagrams.py",
        "validate_and_retry_diagrams.py",
        "select_final_by_angle.py",
        "extract_trajectories.py",
        "export_gold_ndjson.py",
        "generate_dataset_report.py",
    ]:
        run_script(project_root, repo_root, script_name)

    after = {path: path.stat().st_mtime_ns for path in tracked}
    assert after == before
