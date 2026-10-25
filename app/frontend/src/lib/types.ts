// Types mirror the backend contract in app/API.md. Keep them in sync with it.

export type StageStatus = "pending" | "running" | "done" | "failed" | "skipped";

export interface Stage {
  name: string;
  label: string;
  status: StageStatus;
  backend?: string;
  started_at?: string;
  finished_at?: string;
  detail?: string;
  artifact_url?: string;
}

export type JobKind = "pick" | "animal" | "face";
export type JobStatus = "queued" | "running" | "awaiting_input" | "ready" | "failed";

/** One face found in a group photo (face jobs). */
export interface FaceChoice {
  /** 1-based, left to right in the photo. */
  index: number;
  /** Head-and-shoulders crop of this face. */
  image_url: string;
  /** Face box x0, y0, x1, y1 normalized to the uploaded photo. */
  box: [number, number, number, number];
}

export interface Job {
  id: string;
  kind: JobKind;
  status: JobStatus;
  title: string;
  request: { text?: string; image_url?: string; category?: string };
  stages: Stage[];
  artifacts: {
    source_image_url?: string;
    lineart_url?: string;
    svg_url?: string;
    trajectory_url?: string;
  };
  trajectory?: { stroke_count: number; point_count: number; est_draw_seconds: number };
  /** Face jobs: every face found, left to right (pick one while awaiting_input). */
  choices?: FaceChoice[];
  error?: string;
  created_at: string;
  updated_at: string;
}

export type Controller = "act" | "ik";
export type RunMode = "live" | "replay";
export type RunStatus = "starting" | "running" | "finished" | "failed";

export interface RunMetrics {
  rmse_mm?: number;
  max_error_mm?: number;
  sim_seconds?: number;
  jerk_rms?: number;
}

export interface Run {
  id: string;
  job_id: string;
  controller: Controller;
  mode: RunMode;
  status: RunStatus;
  phase?: string;
  progress: number;
  metrics?: RunMetrics;
  cameras: string[];
  /** Output file prefix of the run (backend extra). */
  prefix?: string;
  /** Replays: whether a recorded twin run or the planned path is animated (backend extra). */
  source?: "recorded" | "planned";
  error?: string;
  started_at: string;
  finished_at?: string;
}

export interface CatalogItem {
  sample_id: string;
  category: string;
  subcategory?: string;
  angle?: string;
  stroke_count: number;
  thumbnail_url: string;
  trajectory_url: string;
  /** Backend extra: where the trajectory came from. */
  source?: "gold" | "generated" | "studio";
}

export interface Health {
  gpu: { busy: boolean; utilization: number; note: string };
  isaac: { running: boolean; scene_ready: boolean; config?: string };
  backends: { image: string; chat: string; face: string };
  policy: { checkpoint: string; device: string };
}

export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  reply: string;
  job?: Job;
  run?: Run;
  suggestions?: string[];
}

export type ImageJobKind = "face" | "animal" | "auto";

export interface ChooseRequest {
  index: number;
}

export interface DrawRequest {
  controller?: Controller;
  mode?: RunMode;
}

/** A polyline in normalized canvas units (0..1, y down). */
export type Point = [number, number];
export type Polyline = Point[];

export interface Trace {
  desired: Polyline[];
  executed: Polyline[];
  /** Canvas width / height (backend extra; defaults to 1). */
  aspect?: number;
}

export type LogLevel = "info" | "warn" | "error";

export interface TelemetryEvent {
  type: "telemetry";
  run_id: string;
  t: number;
  phase: string;
  progress: number;
  /** null before the pen moves. */
  tip: Point | null;
  error_mm: number | null;
  frame?: boolean;
}

export type StudioEvent =
  | { type: "job"; job: Job }
  | { type: "run"; run: Run }
  | TelemetryEvent
  | { type: "log"; level: LogLevel; message: string; at: string }
  | { type: "health"; health: Health };

export const CATALOG_CATEGORIES = ["Bird", "Cat", "Dog", "Fish", "Frog", "Horse", "Face"] as const;
