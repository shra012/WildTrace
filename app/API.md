# WildTrace Studio API (backend <-> frontend contract)

Backend: FastAPI at `http://localhost:8000` (`app/backend`). Frontend: Next.js at `http://localhost:3000`
(`app/frontend`), calling the backend through `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`).
All JSON. Timestamps are ISO 8601 UTC strings. File URLs are relative to the API base.

## Concepts

- **Job**: turns an input (text or image) into a robot-drawable trajectory by running the WildTrace
  pipeline stages. Kinds: `pick` (existing trajectory chosen from the catalog), `animal` (new
  trajectory from a photo through the outline -> validate -> extract stages), `face` (face photo ->
  landmark + hair line art -> trajectory). A photo with several people pauses a face job in
  `awaiting_input` with one head-and-shoulders crop per face; the user picks one and it continues.
- **Run**: draws a job's trajectory with the xArm 7, live in Isaac Sim (`mode: "live"`) or by
  replaying a recorded run / animating the plan when Isaac is unavailable (`mode: "replay"`).
- **Backends** are chosen like the data pipeline does: FLUX-schnell on the GPU when it is free,
  OpenRouter otherwise; the API reports which one ran each stage.

## Types

```ts
type StageStatus = "pending" | "running" | "done" | "failed" | "skipped";

interface Stage {
  name: string;          // e.g. "understand", "segment", "outline", "validate", "extract", "detect_faces", "choose_face", "face_landmarks"
  label: string;         // human readable, e.g. "Generate line art"
  status: StageStatus;
  backend?: string;      // e.g. "flux-schnell (GPU)", "openrouter: google/gemini-2.5-flash-image", "mediapipe (CPU)"
  started_at?: string;
  finished_at?: string;
  detail?: string;       // one line, shown under the stage
  artifact_url?: string; // image produced by this stage, if any
}

interface Job {
  id: string;
  kind: "pick" | "animal" | "face";
  status: "queued" | "running" | "awaiting_input" | "ready" | "failed";
  title: string;                 // e.g. "Sitting cat", "Your portrait"
  request: { text?: string; image_url?: string; category?: string };
  stages: Stage[];
  artifacts: {
    source_image_url?: string;   // uploaded or picked photo
    lineart_url?: string;        // final line art PNG
    svg_url?: string;            // strokes as SVG (white background, black strokes)
    trajectory_url?: string;     // trajectory JSON (pipeline schema)
  };
  trajectory?: { stroke_count: number; point_count: number; est_draw_seconds: number };
  choices?: FaceChoice[];        // face jobs: every face found, left to right (pick one while awaiting_input)
  error?: string;
  created_at: string;
  updated_at: string;
}

interface FaceChoice {
  index: number;                 // 1-based, left to right in the photo
  image_url: string;             // head-and-shoulders crop of this face
  box: [number, number, number, number]; // face box x0, y0, x1, y1 normalized to the uploaded photo
}

interface Run {
  id: string;
  job_id: string;
  controller: "act" | "ik";
  mode: "live" | "replay";
  status: "starting" | "running" | "finished" | "failed";
  phase?: string;                // controller state, e.g. "DRAW_STROKE_2"
  progress: number;              // 0..1 of pen-down path drawn
  metrics?: { rmse_mm?: number; max_error_mm?: number; sim_seconds?: number; jerk_rms?: number };
  cameras: string[];             // e.g. ["WideFront", "PaperTop"]
  error?: string;
  started_at: string;
  finished_at?: string;
}

interface CatalogItem {
  sample_id: string;
  category: string;              // Bird, Cat, Dog, Fish, Frog, Horse, Face
  subcategory?: string;
  angle?: string;
  stroke_count: number;
  thumbnail_url: string;         // SVG preview
  trajectory_url: string;
}
```

## Endpoints

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/health` | | `{gpu: {busy: bool, utilization: number, note: string}, isaac: {running: bool, scene_ready: bool, config?: string}, backends: {image: string, chat: string, face: string}, policy: {checkpoint: string, device: string}}` |
| POST | `/api/chat` | `{message: string, session_id?: string}` | `{reply: string, job?: Job, run?: Run, suggestions?: string[]}` |
| POST | `/api/jobs/text` | `{prompt: string}` | `Job` |
| POST | `/api/jobs/image` | multipart: `file`, `kind` = `face` \| `animal` \| `auto`, optional `category` | `Job` |
| GET | `/api/jobs` | | `Job[]` (newest first) |
| GET | `/api/jobs/{id}` | | `Job` |
| POST | `/api/jobs/{id}/choose` | `{index: number}` | `Job` (continues a job in `awaiting_input` with that face; 409 otherwise) |
| POST | `/api/jobs/{id}/draw` | `{controller?: "act" \| "ik", mode?: "live" \| "replay"}` | `Run` |
| GET | `/api/runs` | `?active=true&limit=20` | `Run[]` (newest first; `active` keeps starting/running runs) |
| GET | `/api/runs/{id}` | | `Run` |
| GET | `/api/runs/{id}/stream/{camera}` | | `multipart/x-mixed-replace` MJPEG |
| GET | `/api/runs/{id}/trace` | | `{desired: [[x,y],...][], executed: [[x,y],...][]}` in normalized canvas units (0..1, y down) |
| GET | `/api/catalog` | `?category=Cat&limit=48` | `CatalogItem[]` |
| GET | `/files/{path}` | | artifacts (PNG, SVG, JSON, JPEG) |
| WS | `/ws` | | stream of events below |

## WebSocket events (`/ws`)

```ts
{ type: "job", job: Job }                                  // any job change
{ type: "run", run: Run }                                  // any run change
{ type: "telemetry", run_id: string, t: number, phase: string, progress: number,
  tip: [number, number], error_mm: number }               // ~10 Hz while drawing, tip in normalized canvas units
{ type: "log", level: "info" | "warn" | "error", message: string, at: string }
{ type: "health", health: Health }                         // every 10 s
```

## Chat behaviour (for the UI copy)

- "draw a cat" / "show me a horse" -> `pick` job from the catalog, then offers to draw it.
- "make a new sitting dog" -> `animal` job from an unused photo of that animal through the pipeline.
- Message with an attached image -> `face` if a face is found (or the user says face/portrait), else `animal`.
- Several faces -> the job waits in `awaiting_input`; the UI shows `choices` to click, or the user types
  "2", "the second one", "left", "rightmost" in the chat.
- "draw it" / "start" -> starts a run for the latest ready job.
- Replies are short; the job and run objects drive the UI.
