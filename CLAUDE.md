# WildTrace

Photos (Open Images animals, faces) → line drawings → pen-stroke trajectories → drawn by a UFACTORY xArm 7,
in Isaac Sim and on the physical lab arm. Overview: `README.md`.

- `wildtrace/`, `scripts/`, `configs/` - bronze → silver → gold pipeline; one script per stage.
- `isaac-sim/` - robot: scenes, controllers (Lula IK, ACT, SmolVLA, PPO), training, results. See `isaac-sim/README.md`.
- `app/` - WildTrace Studio (FastAPI + Next.js). See `app/README.md`, `app/API.md`.

## Commands

- Python 3.12, `uv`. Tests: `uv run --extra dev --extra app pytest tests isaac-sim/tests`.
- `.env` from `.env.example` (model hosts, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`). Never print keys.
- LeRobot (ACT, SmolVLA): separate venv `isaac-sim/.venv-act`.
- Isaac Sim 6.0.1 runs on Windows; WSL drives it through the `wildtrace-sim` MCP server
  (`isaac-sim/mcp_server/wildtrace_sim_server.py`).
- One 16 GB GPU shared by FLUX, training and Isaac Sim: check `nvidia-smi` first; close Isaac when idle.

## Pipeline

`fetch_openimages` → `curate_bronze` → `normalize_to_silver` → `enrich_and_crop_subjects` →
`generate_line_diagrams` (FLUX) → `validate_and_retry_diagrams` (OpenCV + VLM judge, LangGraph) →
`select_final_by_angle` → `extract_trajectories` → `export_gold_ndjson` → `generate_dataset_report`
(`scripts/*.py`, thin wrappers over `wildtrace/agentic_pipeline.py`). Each takes `--repo-root`, `--config-dir`,
repeatable `--category`. Manifests are append-only NDJSON; rows are reused when config and input hashes match.
Outputs go to `outputs/` (gitignored).

## Robot

- Configs: `isaac-sim/config/xarm7_drawing.yaml` (default sim), `xarm7_drawing_real.yaml` (lab-arm twin),
  `xarm7_hardware.yaml` (physical arm), `act_drawing.yaml` (ACT training).
- Gold test set: `isaac-sim/data/act_test/`. ACT checkpoints: `isaac-sim/models/`.
- Results: `isaac-sim/results/` (`act_eval/`, `studio_live/`, ...).

## Conventions

- No Claude/AI attribution anywhere: no `Co-Authored-By` trailer, no "Generated with" line.
- Plain hyphens only, no em or en dashes.
- Commit or push only after the user approves that specific change (exception below).
- Show drafts of GitHub issues and comments before posting.
- Never run `wsl --shutdown`, restart or suspend the machine.
- Do not change the calibration from real-arm experiments (`xarm7_drawing_real.yaml` robot and
  placement, `xarm7_hardware.yaml`, the calibrated URDF).

## Calibration and evaluation runs: always record and push

Only for runs like: real-arm drawing batches (including ones a teammate shares), calibration runs, and
full twin evaluations (gold set, robustness sweeps). Not for one-off sim draws, demos or smoke tests.

1. Copy the run's `*_run_metrics.json` and `*_comparison.png` into the matching `isaac-sim/results/` folder,
   named by drawing and controller.
2. Add a row per run to that folder's results markdown: drawing, controller and checkpoint, scene config,
   finished, path RMSE / p95 / max, sim time, date; failures included, with the reason.
3. Commit only those results files and push right away. No videos, checkpoints or datasets.
