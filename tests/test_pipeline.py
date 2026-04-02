from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from wildtrace.agentic_pipeline import select_final_by_angle
from wildtrace.config import load_runtime_config
from wildtrace.diagram import (
    DiagramBackend,
    DiagramBackendResult,
    OllamaSemanticValidator,
    OpenCVValidationResult,
    SemanticValidationResult,
    build_outline_validator,
    run_langgraph_validation_loop,
    validate_with_opencv,
)
from wildtrace.io_utils import read_ndjson


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_test_repo(tmp_path: Path, records: list[dict] | None = None) -> Path:
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
        "processed/silver/crops",
        "processed/silver/diagrams",
        "processed/silver/qa",
        "processed/gold/diagrams",
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
                "manifest_path": "raw_data/bronze/manifests/openimages_fetch.ndjson",
                "latest_view_path": "raw_data/bronze/manifests/openimages_fetch_latest.ndjson",
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
                "crops_dir": "processed/silver/crops",
                "diagrams_dir": "processed/silver/diagrams",
                "qa_dir": "processed/silver/qa",
            },
            "gold": {
                "diagrams_dir": "processed/gold/diagrams",
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
            "outline_generator": {
                "backend": "informative_drawings",
                "model_backend": "mock",
                "model_name": "informative_drawings",
                "model_version_or_checkpoint": "mock-v1",
                "max_attempts": 5,
                "default_params": {"blur_radius": 0.8, "threshold": 150},
            },
            "outline_rectifier": {
                "backend": "silhouette_outline_rectifier",
                "model_backend": "local",
                "model_name": "silhouette_outline_rectifier",
                "model_version_or_checkpoint": "local-v1",
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

    gold_path = repo_root / "processed/gold/ndjson/gold_samples.ndjson"
    records = [json.loads(line) for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    record = records[0]
    assert record["category"] == "Cat"
    assert record["subcategory"] == "domestic_cat"
    assert record["selected_for_gold"] is True
    assert record["diagram_validation_status"] == "accepted"
    assert record["trajectory"]["stroke_count"] >= 1
    assert Path(repo_root / record["gold_refs"]["diagram_path"]).exists()
    assert Path(repo_root / "artifacts/reports/dataset_report.md").exists()


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
    curated = read_ndjson(repo_root / "raw_data/bronze/manifests/bronze_curated.ndjson")
    accepted = read_ndjson(repo_root / "raw_data/bronze/manifests/bronze_accepted.ndjson")
    assert len(curated) == 2
    assert len(accepted) == 1
    rejected = [row for row in curated if row["curation_status"] == "rejected"]
    assert rejected
    assert "duplicate_image_checksum" in rejected[0]["curation_flags"]
    assert rejected[0]["deleted_paths"]


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
    assert result.score == 0.0
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
    assert result.score == 0.84
    assert result.reason == "clean and drawable"
    assert result.metadata["model_name"] == "qwen2.5vl:7b"


def test_ollama_validator_prompt_is_outline_only() -> None:
    validator = OllamaSemanticValidator({"ollama_model_name": "qwen2.5vl:7b"})
    prompt = validator._build_prompt(  # noqa: SLF001
        {"category": "Bird", "subcategory": "hummingbird", "angle_bucket": "front"},
        OpenCVValidationResult(passed=True, score=0.8, flags=[], metrics={}),
    )
    assert "specific animal" in prompt
    assert "hummingbird" not in prompt
    assert "Bird" not in prompt
    assert "outer outline" in prompt


def test_run_langgraph_validation_loop_retries_and_preserves_state(tmp_path: Path) -> None:
    subject_path = tmp_path / "subject.png"
    Image.new("RGB", (64, 64), "white").save(subject_path)

    class StubGenerator(DiagramBackend):
        def run(self, image, destination, params):
            destination.parent.mkdir(parents=True, exist_ok=True)
            threshold = int(params["threshold"])
            diagram = Image.new("L", image.size, 0)
            if threshold <= 132:
                draw = ImageDraw.Draw(diagram)
                draw.ellipse((18, 20, 46, 42), outline=255, width=3)
                draw.line((46, 31, 56, 28), fill=255, width=2)
            diagram.save(destination)
            return DiagramBackendResult(diagram_path=destination, metadata={"threshold": threshold})

    class StubValidator:
        def validate(self, sample, diagram_path, opencv_result):
            passed = bool(Image.open(diagram_path).getbbox())
            return SemanticValidationResult(
                passed=passed,
                score=0.9 if passed else 0.2,
                reason="ok" if passed else "retry",
                metadata={"model_name": "stub-validator"},
            )

    class StubRectifier:
        def run(self, subject_image, generated_path, destination, params):
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.open(generated_path).save(destination)
            return type(
                "RectifierResult",
                (),
                {"diagram_path": destination, "metadata": {"model_name": "stub-rectifier"}},
            )()

    result = run_langgraph_validation_loop(
        sample={"sample_id": "bird-1", "category": "Bird", "subcategory": "hummingbird", "angle_bucket": "front"},
        subject_path=subject_path,
        diagram_root=tmp_path / "diagrams",
        outline_generator=StubGenerator({}),
        outline_rectifier=StubRectifier(),
        outline_validator=StubValidator(),
        opencv_settings={
            "min_foreground_ratio": 0.03,
            "max_foreground_ratio": 0.32,
            "target_foreground_ratio": 0.16,
            "max_component_count": 18,
            "max_small_contour_ratio": 0.70,
        },
        initial_params={"threshold": 150, "blur_radius": 0.8},
        max_attempts=2,
    )

    assert result["diagram_validation_status"] == "accepted"
    assert result["diagram_attempt"] == 2
    assert result["outline_validator_model"] == "stub-validator"
    assert result["outline_rectifier_backend"]["model_name"] == "stub-rectifier"


def test_select_final_by_angle_keeps_best_score_per_bucket(tmp_path: Path) -> None:
    repo_root = build_test_repo(tmp_path)
    validated_path = repo_root / "processed/silver/qa/validated_diagrams.ndjson"
    rows = [
        {
            "sample_id": "a",
            "category": "Bird",
            "subcategory": "flamingo",
            "angle_bucket": "front",
            "diagram_path": "processed/silver/diagrams/Bird/a.png",
            "diagram_attempt": 1,
            "outline_generator_backend": {"model_name": "informative_drawings"},
            "outline_rectifier_backend": {"model_name": "silhouette_outline_rectifier"},
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
            "diagram_path": "processed/silver/diagrams/Bird/b.png",
            "diagram_attempt": 2,
            "outline_generator_backend": {"model_name": "informative_drawings"},
            "outline_rectifier_backend": {"model_name": "silhouette_outline_rectifier"},
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
            "diagram_path": "processed/silver/diagrams/Bird/c.png",
            "diagram_attempt": 1,
            "outline_generator_backend": {"model_name": "informative_drawings"},
            "outline_rectifier_backend": {"model_name": "silhouette_outline_rectifier"},
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
