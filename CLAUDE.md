# WildTrace - project index

WildTrace turns Open Images animal photos into robot-drawable line diagrams and stroke
trajectories, then draws them with a simulated UFACTORY xArm 7 in NVIDIA Isaac Sim.

Two largely independent halves:

1. **ETL pipeline** (`wildtrace/`, `scripts/`, `configs/`) - local-first bronze → silver → gold
   medallion pipeline. Each stage is a standalone script so it can become an Airflow DAG task.
2. **Robot simulation** (`isaac-sim/`) - IK / behaviour-cloning / RL drawing on xArm 7, fed by
   gold trajectories. Has its own README, config, tests and Claude skill.

## Setup and commands

- Python 3.12 only (`.python-version`, `pyproject.toml`); dependencies locked in `uv.lock`.
- `uv sync --extra dev` then `uv run pytest` (testpaths = `tests/`, `pythonpath = .`).
- Isaac Sim tests: `uv run pytest isaac-sim/tests` (pure-Python parts; sim itself needs Isaac Sim 6.0.1).
- Runtime env: copy `.env.example` → `.env` (Ollama host/model, BioCLIP device, rectifier models,
  optional `ANTHROPIC_API_KEY` / OpenRouter for semantic validation). GPU (CUDA) expected.
- Helpers: `scripts/install_ollama_and_models.sh`, `scripts/install_omnigen2.sh`.

Every stage script accepts `--repo-root`, `--config-dir`, and repeatable `--category`
(`wildtrace/pipeline.py:parse_common_args`). `--category` narrows expensive work only; it never
drops other categories' rows from existing manifests.

## Pipeline stages (run in order)

| # | Script | Function | Layer |
|---|---|---|---|
| 1 | `scripts/fetch_openimages.py` | `pipeline.fetch_openimages` | bronze: download images/masks/metadata, versioned |
| - | `scripts/reconcile_bronze_fetch_storage.py` | `pipeline.reconcile_fetch_storage` | bronze: repair/clean fetch storage |
| 2 | `scripts/curate_bronze.py` | `bronze_stage.curate_bronze` | bronze: dedupe, quality, angle feasibility, BioCLIP enrichment |
| 3 | `scripts/normalize_to_silver.py` | `silver_stage.normalize_to_silver` | silver: resize/normalize images + masks |
| 4 | `scripts/enrich_and_crop_subjects.py` | `silver_stage.enrich_and_crop_subjects` | silver: isolate + crop subject |
| 5 | `scripts/generate_line_diagrams.py` | `silver_stage.generate_line_diagrams` | silver: outline rectifier (FLUX by default) |
| 6 | `scripts/validate_and_retry_diagrams.py` | `silver_stage.validate_and_retry_diagrams` | silver: OpenCV prescreen + VLM semantic validation, LangGraph retry loop |
| 7 | `scripts/select_final_by_angle.py` | `silver_stage.select_final_by_angle` | best diagram per category/subcategory/angle bucket |
| 8 | `scripts/extract_trajectories.py` | `gold_stage.extract_trajectories` | gold: SVG + normalized stroke trajectories |
| 9 | `scripts/export_gold_ndjson.py` | `gold_stage.export_gold_ndjson` | gold: canonical NDJSON records |
| 10 | `scripts/generate_dataset_report.py` | `gold_stage.generate_dataset_report` | QA report |

Scripts are thin wrappers around `wildtrace/agentic_pipeline.py` `*_main()` functions, which use
`stage_io.run_stage_main`. `pipeline/pipeline.ipynb` drives the same stages interactively.

**Incrementality:** manifests are append-only NDJSON ledgers; each stage computes a config hash +
input hash per row and reuses rows whose hash and files are unchanged (`_*_reusable` helpers).
`sample_id` = logical identity, `checksum` = content identity, `asset_version` bumps on change.
Outputs live under `outputs/` (`configs/storage.yaml`, gitignored); manifest paths in `wildtrace/stage_io.py`.

## Package map (`wildtrace/`)

- `config.py` - loads `configs/*.yaml`, resolves `${ENV:-default}` placeholders.
- `pipeline.py` - CLI args, runtime loading, Open Images discovery/fetch, versioned bronze paths.
- `bronze_stage.py`, `silver_stage.py`, `gold_stage.py` - stage implementations.
- `stage_io.py` - manifest/checkpoint path helpers and `run_stage_main`.
- `io_utils.py` - NDJSON / file helpers. `images.py` - PIL/OpenCV image + mask utils, outline stroke sampling, SVG writing.
- `enrichment.py` - `BioCLIPHuggingFaceBackend` and heuristic fallback (subcategory + tags).
- `viewpoint.py` - mask-feature viewpoint/angle classifier (`LocalScoreViewpointBackend`).
- `diagram.py` (~2k lines) - outline rectifiers (`FluxSilhouetteRectifier`, `FluxKontextRectifier`,
  `SDControlNetLineartRectifier`, `SDXLControlNetRectifier`, `InformativeDrawingsRectifier`,
  `OmniGen2Rectifier`, `SilhouetteOutlineRectifier`), semantic validators (Ollama, OpenRouter,
  Anthropic, Mock), OpenCV validation, and the LangGraph validation/retry graph.
- `eval_metrics.py` - benchmark metrics for generated line diagrams.
- `trajectory_candidates.py` - compile sparse VLM stroke plans into trajectories; raster-distance scoring.

## Configs (`configs/`)

`datasets.yaml` (categories + limits: Cat, Dog, Horse, Bird, Butterfly, Fish, Frog; fetch mode),
`models.yaml` (enrichment, rectifier, validator backends), `quality.yaml` (image gates),
`viewpoints.yaml` (angle buckets), `export.yaml` (stroke/NDJSON limits), `storage.yaml` (output root),
`benchmark.yaml` / `benchmark_sdxl.yaml` (model benchmark runs; select via `WILDTRACE_BENCHMARK_CONFIG`).

## Benchmarks and experiments (`scripts/`)

- Model comparison: `benchmark_models.py`, `benchmark_setups.py`, `benchmark_docx.py`,
  `preflight_sdxl_benchmark.py`, `validator_robustness.py`, `sweep_diagrams.py`.
- Learned trajectories (Molmo VLM): `generate_molmo_stroke_plan.py`, `compile_stroke_plan.py`,
  `compare_trajectory_candidates.py`, `run_molmo_zeroshot_benchmark.py`, `build_vlm_benchmark_table.py`.
- Label/eval sets: `benchmarks/{labelset_v1,evalset_v1,evalset_v2}.json`.

## Isaac Sim (`isaac-sim/`)

- `src/` - `xarm7_loader`, `ik_controller` (Lula IK), `articulation_control`, `drawing_state_machine`
  (approach/lower/draw/lift), `coordinate_mapper` (normalized canvas → world), `trajectory_loader`,
  `path_geometry`, `metrics`, `demonstration`, `behavior_cloning`, `rl_policy`, `project_config`.
- `scripts/` - scene setup, two-stroke drawing, demo recording, BC/PPO training, evaluation, plotting,
  `mcp_drawing_controller.py` (Action Graph controller), `provision_and_draw.py`.
- `mcp_server/wildtrace_sim_server.py` - `wildtrace-sim` MCP server (`sim_status`, `sim_start`,
  `setup_scene`, `draw(category)`, `draw_result`, `sim_shutdown`).
- `.claude/skills/draw-animal/` - skill for "draw a <animal>"; only serves existing gold trajectories.
- `config/xarm7_drawing.yaml` - robot/drive/drawing tuning (Earth gravity, force-mode servo).
- `inputs/` - sample gold diagrams, SVGs and trajectories per category; `data/` - two-stroke samples.
- Simulation only: no physical xArm client or network path.

## Docs

`docs/Plan.md` (overall ETL plan), `docs/PipelinePlan.md` (bronze design), `docs/CheckList.md`,
`docs/benchmarks/` (r2 findings, SDXL ControlNet, benchmark plan), `docs/MolmoTrajectoryPilot.md`,
`docs/MolmoZeroShotBenchmark.md`, `docs/diagrams/` (architecture), `docs/workbook/` (report builder),
`isaac-sim/README.md`, `isaac-sim/DISCOVERY_REPORT.md`, `isaac-sim/WildTrace_RL_Isaac_Sim_Overview.md`.

## Tests

`tests/` - `test_pipeline.py`, `test_eval_metrics.py`, `test_sdxl_rectifier.py`,
`test_trajectory_candidates.py`. `isaac-sim/tests/` - path geometry, trajectory, metrics/robot, BC.

## Conventions for Claude

- Do not add Claude as a co-author in git commits: no `Co-Authored-By` trailer or any other Claude/AI attribution lines in commit messages.
- Do not take credit for the code: no "Generated with Claude Code" or similar attribution in commits, PR descriptions, code comments, or docs.
- Do not use em dashes or en dashes; use a plain hyphen `-` instead.
