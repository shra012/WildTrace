# WildTrace Studio

A chat and upload app on top of the WildTrace pipeline. You ask for an animal or upload a photo, the app makes a
robot-drawable trajectory, and the xArm 7 draws it in Isaac Sim. While it draws, you see live camera frames, the
pen trace and tracking error. The API contract is in [API.md](API.md).

| Part | Where | Stack |
| --- | --- | --- |
| Backend | `app/backend/wildtrace_app/` | FastAPI, WebSocket events, MJPEG streams |
| Frontend | `app/frontend/` | Next.js, Tailwind, DaisyUI (see its README) |

## Run

```bash
uv sync --extra dev --extra app
uv run --extra app uvicorn wildtrace_app.main:app --app-dir app/backend --port 8000
cd app/frontend && npm install && npm run dev   # http://localhost:3000
```

`.env` at the repo root is read at startup:

- `OPENROUTER_API_KEY` enables LLM chat routing and the image stand-in. Without it, chat falls back to rules.
- `WILDTRACE_CHAT_MODEL` picks the chat model.
- `WILDTRACE_DRAW_CONFIG` picks the scene. The default is the lab-arm twin, `config/xarm7_drawing_real.yaml`.
- `WILDTRACE_CONTROLLER` sets the controller to `act` (the default) or `ik`.
- `WILDTRACE_REPLAY_SECONDS` sets how long a replay plays on screen.
- `WILDTRACE_STUDIO_DIR` sets where jobs, runs and uploads are stored. The default is `outputs/studio`.

## What each request does

| Request | Job | Stages |
| --- | --- | --- |
| "draw a cat", gallery click | `pick` | understand -> pick (gold set first, then generated) |
| "make a new sitting dog" | `animal` | pick_photo -> segment -> outline -> validate -> extract. Outlines come from FLUX-schnell when the GPU is free, and from OpenRouter otherwise (`wildtrace/studio_pipeline.py`). |
| face photo | `face` | detect_faces -> choose_face -> face_landmarks -> extract. MediaPipe runs on the CPU (`wildtrace/face_lineart.py`). |
| "draw it", Draw button | run | live in Isaac Sim, or a replay |

**Group photos.** The app searches the whole photo and overlapping tiles for faces, and saves a head-and-shoulders
crop of each one. The job then waits in `awaiting_input` until you pick a face, either by clicking it or by typing
"2", "the second one" or "left". In the drawing of the chosen face, neighbours' hair and shoulders are left out.

**Live and replay.** Runs are live when Isaac Sim is running. The backend calls the wildtrace-sim MCP server's
`draw_trajectory`, then streams `isaac-sim/outputs/mcp_sessions/live/<prefix>.json` and the camera JPEGs. Otherwise
the run is a replay (`isaac-sim/src/run_replay.py`):

- For a gold drawing with a recorded lab-arm twin run, the replay follows that run's progress, error, metrics and
  camera frames.
- For any other drawing, it animates the planned pen path.

## Tests

`uv run --extra dev --extra app pytest tests/test_studio_app.py tests/test_face_lineart.py isaac-sim/tests/test_live_telemetry.py`
(CPU only; no network).
