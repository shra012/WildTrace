// In-browser mock backend (NEXT_PUBLIC_MOCK=1 or ?mock=1). It implements the same surface as the
// real API client, including WebSocket-style events, so the whole UI can be demoed offline.

import { drawingFor, groupPhoto, maskSvg, pathLength, photoSvg, strokesSvg } from "./drawings";
import { ApiError } from "./errors";
import type {
  CatalogItem,
  ChatRequest,
  ChatResponse,
  Controller,
  DrawRequest,
  FaceChoice,
  Health,
  ImageJobKind,
  Job,
  JobKind,
  Point,
  Polyline,
  Run,
  RunMode,
  Stage,
  StudioEvent,
  Trace,
} from "./types";
import { CATALOG_CATEGORIES } from "./types";

type Listener = (event: StudioEvent) => void;

const IMAGE_BACKEND = "openrouter: google/gemini-2.5-flash-image";
const CHAT_BACKEND = "openrouter: google/gemini-2.5-flash";

interface Drawing {
  category: string;
  strokes: Polyline[];
}

interface MockRun {
  run: Run;
  noisy: Polyline[];
  desired: Polyline[];
  lengths: number[];
  total: number;
  startedAt: number;
  startupMs: number;
  drawMs: number;
  timer?: ReturnType<typeof setInterval>;
  ticks: number;
  err: number;
  errSq: number;
  errN: number;
  errMax: number;
}

interface StagePlan {
  name: string;
  label: string;
  backend: string;
  detail: (d: Drawing) => string;
  artifact?: "photo" | "mask" | "lineart" | "svg" | "colored";
  ms: number;
}

const listeners = new Set<Listener>();
const jobs = new Map<string, Job>();
const drawings = new Map<string, Drawing>();
const runs = new Map<string, MockRun>();
const catalogStrokes = new Map<string, Drawing>();
let catalog: CatalogItem[] = [];
let started = false;
let health: Health = {
  gpu: { busy: true, utilization: 96, note: "Reserved by the XRayGPT test evaluation" },
  isaac: { running: true, scene_ready: true, config: "lab_arm_twin" },
  backends: { image: IMAGE_BACKEND, chat: CHAT_BACKEND, face: "mediapipe (CPU)" },
  policy: { checkpoint: "act_xarm7_lab_twin/best.ckpt", device: "cpu" },
};

// ---------------------------------------------------------------------------------------------
// Utilities

function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function newId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function iso(offsetMs = 0): string {
  return new Date(Date.now() + offsetMs).toISOString();
}

function delay<T>(value: T, ms = 120 + Math.random() * 260): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

function clone<T>(value: T): T {
  return typeof structuredClone === "function" ? structuredClone(value) : JSON.parse(JSON.stringify(value));
}

function emit(event: StudioEvent) {
  for (const listener of listeners) {
    try {
      listener(clone(event));
    } catch {
      // A broken listener must not stop the mock engine.
    }
  }
}

function log(level: "info" | "warn" | "error", message: string) {
  emit({ type: "log", level, message, at: iso() });
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function pointCount(strokes: Polyline[]): number {
  return strokes.reduce((n, s) => n + s.length, 0);
}

function totalLength(strokes: Polyline[]): number {
  return strokes.reduce((n, s) => n + pathLength(s), 0);
}

function estDrawSeconds(strokes: Polyline[]): number {
  return Math.round(totalLength(strokes) * 9 + strokes.length * 1.1);
}

function trajectoryJson(d: Drawing): string {
  const payload = {
    schema: "wildtrace.trajectory.v1",
    category: d.category,
    canvas: { units: "normalized", origin: "top-left" },
    strokes: d.strokes.map((points, i) => ({ id: i, pen: "down", points })),
  };
  return `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(payload))}`;
}

function artifactFor(kind: StagePlan["artifact"], d: Drawing): string | undefined {
  switch (kind) {
    case "photo":
      return photoSvg(d.category, d.strokes);
    case "mask":
      return maskSvg(d.strokes);
    case "lineart":
      return strokesSvg(d.strokes, { width: 7 });
    case "svg":
      return strokesSvg(d.strokes, { width: 4 });
    case "colored":
      return strokesSvg(d.strokes, { width: 5, colored: true });
    default:
      return undefined;
  }
}

function finalArtifacts(d: Drawing, sourceUrl?: string): Job["artifacts"] {
  return {
    source_image_url: sourceUrl ?? photoSvg(d.category, d.strokes),
    lineart_url: strokesSvg(d.strokes, { width: 7 }),
    svg_url: strokesSvg(d.strokes, { width: 4 }),
    trajectory_url: trajectoryJson(d),
  };
}

function trajectoryStats(d: Drawing): Job["trajectory"] {
  return {
    stroke_count: d.strokes.length,
    point_count: pointCount(d.strokes),
    est_draw_seconds: estDrawSeconds(d.strokes),
  };
}

// ---------------------------------------------------------------------------------------------
// Catalog

const CATALOG_SPEC: Record<string, { sub: string[]; angles: string[]; profile: boolean }> = {
  Bird: { sub: ["European robin", "Blue jay", "House sparrow", "Goldfinch"], angles: ["side", "side", "three_quarter", "side"], profile: true },
  Cat: { sub: ["Tabby", "Siamese", "Maine coon", "Bengal"], angles: ["front", "front", "three_quarter", "front"], profile: false },
  Dog: { sub: ["Labrador", "Beagle", "Border collie", "Corgi"], angles: ["front", "three_quarter", "front", "front"], profile: false },
  Fish: { sub: ["Goldfish", "Clownfish", "Trout", "Angelfish"], angles: ["side", "side", "side", "three_quarter"], profile: true },
  Frog: { sub: ["Tree frog", "Bullfrog", "Common frog", "Poison dart frog"], angles: ["front", "front", "three_quarter", "front"], profile: false },
  Horse: { sub: ["Arabian", "Mustang", "Shire", "Appaloosa"], angles: ["side", "side", "three_quarter", "side"], profile: true },
  Face: { sub: ["Portrait", "Portrait", "Portrait"], angles: ["front", "front", "three_quarter"], profile: false },
};

function buildCatalog() {
  const items: CatalogItem[] = [];
  CATALOG_CATEGORIES.forEach((category, ci) => {
    const spec = CATALOG_SPEC[category];
    spec.sub.forEach((sub, i) => {
      const r = rng(ci * 97 + i * 13 + 7);
      const strokes = drawingFor(category, {
        scale: 0.9 + r() * 0.12,
        dx: (r() - 0.5) * 0.06,
        dy: (r() - 0.5) * 0.04,
        rot: (r() - 0.5) * 0.1,
        mirror: spec.profile && i % 2 === 1,
      });
      const hex = Math.floor(r() * 0xffffffff).toString(16).padStart(8, "0");
      const sampleId = category === "Face" ? `face_${hex}` : `oi_${hex}`;
      const d = { category, strokes };
      catalogStrokes.set(sampleId, d);
      items.push({
        sample_id: sampleId,
        category,
        subcategory: sub,
        angle: spec.angles[i],
        stroke_count: strokes.length,
        thumbnail_url: strokesSvg(strokes, { width: 5 }),
        trajectory_url: trajectoryJson(d),
        source: i === 3 ? "generated" : "gold",
      });
    });
  });
  catalog = items;
}

function pickCatalogItem(category: string, sampleId?: string): CatalogItem {
  const byId = sampleId ? catalog.find((c) => c.sample_id.toLowerCase() === sampleId.toLowerCase()) : undefined;
  if (byId) return byId;
  const pool = catalog.filter((c) => c.category === category);
  return pool[Math.floor(Math.random() * pool.length)] ?? catalog[0];
}

// ---------------------------------------------------------------------------------------------
// Jobs

const PICK_PLAN: StagePlan[] = [
  { name: "understand", label: "Understand the request", backend: CHAT_BACKEND, detail: (d) => `Matched "${d.category.toLowerCase()}" in the gold catalog`, ms: 900 },
  { name: "lookup", label: "Pick a gold trajectory", backend: "catalog (local gold)", detail: (d) => `${d.strokes.length} strokes, ${pointCount(d.strokes)} points`, artifact: "svg", ms: 700 },
];

function animalPlan(fromUpload: boolean): StagePlan[] {
  return [
    fromUpload
      ? { name: "understand", label: "Read the photo", backend: CHAT_BACKEND, detail: (d) => `Looks like a ${d.category.toLowerCase()}, clear subject`, ms: 1300 }
      : { name: "understand", label: "Understand the request", backend: CHAT_BACKEND, detail: (d) => `New ${d.category.toLowerCase()} from an unused photo`, ms: 900 },
    fromUpload
      ? { name: "segment", label: "Isolate the subject", backend: "rembg u2net (CPU)", detail: () => "Subject mask, cropped to 512x512", artifact: "mask", ms: 1800 }
      : { name: "fetch", label: "Find an unused photo", backend: "open images (bronze)", detail: () => `Unused photo oi_${Math.floor(Math.random() * 0xffffff).toString(16)}, 1024x768`, artifact: "photo", ms: 1400 },
    ...(fromUpload
      ? []
      : [{ name: "segment", label: "Isolate the subject", backend: "open images mask (CPU)", detail: () => "Mask IoU 0.91, cropped to 512x512", artifact: "mask" as const, ms: 1500 }]),
    { name: "outline", label: "Generate line art", backend: IMAGE_BACKEND, detail: () => "Silhouette outline, attempt 1 of 3", artifact: "lineart", ms: 4200 },
    { name: "validate", label: "Validate the diagram", backend: `opencv + ${CHAT_BACKEND}`, detail: () => `Prescreen ok, semantic score 0.${80 + Math.floor(Math.random() * 15)}`, artifact: "lineart", ms: 2600 },
    { name: "extract", label: "Extract pen strokes", backend: "skeleton tracer (CPU)", detail: (d) => `${d.strokes.length} strokes, ${pointCount(d.strokes)} points`, artifact: "colored", ms: 1600 },
  ];
}

const FACE_PLAN: StagePlan[] = [
  { name: "detect_faces", label: "Find faces", backend: "mediapipe (CPU)", detail: () => "1 face found", ms: 1300 },
  { name: "choose_face", label: "Choose a face", backend: "you", detail: () => "one face", ms: 0 },
  { name: "face_landmarks", label: "Detect face landmarks", backend: "mediapipe (CPU)", detail: () => "478 landmarks, hair outline merged", artifact: "lineart", ms: 2600 },
  { name: "extract", label: "Extract pen strokes", backend: "skeleton tracer (CPU)", detail: (d) => `${d.strokes.length} strokes, ${pointCount(d.strokes)} points`, artifact: "colored", ms: 1200 },
];

/** Faces "found" by the mock detect_faces stage, assigned to job.choices when it finishes. */
const pendingChoices = new Map<string, FaceChoice[]>();

function pendingStages(plan: StagePlan[]): Stage[] {
  return plan.map((p) => ({ name: p.name, label: p.label, status: "pending" }));
}

function touch(job: Job) {
  job.updated_at = iso();
  emit({ type: "job", job });
}

function runPlan(job: Job, plan: StagePlan[], from = 0, sourceUrl?: string) {
  const d = drawings.get(job.id);
  if (!d) return;
  const step = (i: number) => {
    if (!jobs.has(job.id)) return;
    if (i >= plan.length) {
      job.status = "ready";
      job.artifacts = finalArtifacts(d, sourceUrl ?? job.artifacts.source_image_url);
      job.trajectory = trajectoryStats(d);
      touch(job);
      log("info", `Job ready: ${job.title} (${d.strokes.length} strokes)`);
      return;
    }
    const p = plan[i];
    const stage = job.stages[i];
    if (p.name === "choose_face") {
      const choices = job.choices ?? [];
      if (choices.length > 1) {
        job.status = "awaiting_input";
        job.title = `Group photo (${choices.length} faces)`;
        stage.status = "running";
        stage.started_at = iso();
        stage.detail = `${choices.length} faces found, pick one`;
        touch(job);
        log("info", `Found ${choices.length} faces. Pick the one to draw.`);
        return;
      }
      stage.status = "skipped";
      stage.detail = "one face";
      touch(job);
      setTimeout(() => step(i + 1), 200);
      return;
    }
    job.status = "running";
    stage.status = "running";
    stage.backend = p.backend;
    stage.started_at = stage.started_at ?? iso();
    stage.detail = "Working...";
    touch(job);
    if (p.name === "outline" || p.name === "hair_outline") {
      log("warn", "GPU busy (XRayGPT eval), generating line art with OpenRouter instead of FLUX-schnell");
    }
    setTimeout(() => {
      stage.status = "done";
      stage.finished_at = iso();
      stage.detail = p.detail(d);
      stage.artifact_url = artifactFor(p.artifact, d);
      if (p.name === "detect_faces") {
        const found = pendingChoices.get(job.id);
        pendingChoices.delete(job.id);
        job.choices = found ?? [{ index: 1, image_url: job.request.image_url ?? photoSvg("Face", d.strokes), box: [0.3, 0.15, 0.7, 0.75] }];
        stage.detail = `${job.choices.length} face${job.choices.length === 1 ? "" : "s"} found`;
        stage.artifact_url = job.request.image_url;
      }
      touch(job);
      setTimeout(() => step(i + 1), 250);
    }, p.ms * (0.8 + Math.random() * 0.4));
  };
  step(from);
}

function completedStages(plan: StagePlan[], d: Drawing, startMs: number): Stage[] {
  let t = startMs;
  return plan.map((p) => {
    const s: Stage = {
      name: p.name,
      label: p.label,
      status: "done",
      backend: p.backend,
      started_at: new Date(t).toISOString(),
      finished_at: new Date(t + p.ms).toISOString(),
      detail: p.detail(d),
      artifact_url: artifactFor(p.artifact, d),
    };
    t += p.ms + 250;
    return s;
  });
}

function addJob(job: Job, d: Drawing) {
  jobs.set(job.id, job);
  drawings.set(job.id, d);
}

function createPickJob(category: string, title?: string, sampleId?: string, text?: string): Job {
  const item = pickCatalogItem(category, sampleId);
  const d = catalogStrokes.get(item.sample_id) ?? { category, strokes: drawingFor(category) };
  const job: Job = {
    id: newId("job"),
    kind: "pick",
    status: "queued",
    title: title ?? `${capitalize(defaultPose(item.category))} ${item.category.toLowerCase()}`,
    request: { text, category: item.category },
    stages: pendingStages(PICK_PLAN),
    artifacts: {},
    created_at: iso(),
    updated_at: iso(),
  };
  addJob(job, d);
  emit({ type: "job", job });
  runPlan(job, PICK_PLAN);
  return job;
}

function createAnimalJob(category: string, title: string, text?: string, sourceUrl?: string): Job {
  const plan = animalPlan(Boolean(sourceUrl));
  const r = Math.random;
  const d = { category, strokes: drawingFor(category, { scale: 0.92 + r() * 0.08, rot: (r() - 0.5) * 0.08 }) };
  const job: Job = {
    id: newId("job"),
    kind: "animal",
    status: "queued",
    title,
    request: { text, category, image_url: sourceUrl },
    stages: pendingStages(plan),
    artifacts: sourceUrl ? { source_image_url: sourceUrl } : {},
    created_at: iso(),
    updated_at: iso(),
  };
  addJob(job, d);
  emit({ type: "job", job });
  runPlan(job, plan, 0, sourceUrl);
  return job;
}

function resumeFaceJob(job: Job, index: number): Job {
  const choices = job.choices ?? [];
  const stageIdx = job.stages.findIndex((s) => s.name === "choose_face");
  const stage = job.stages[stageIdx];
  stage.status = "done";
  stage.backend = "you";
  stage.finished_at = iso();
  stage.detail = `face ${index} of ${choices.length}`;
  stage.artifact_url = choices.find((c) => c.index === index)?.image_url;
  job.status = "running";
  job.title = `Portrait (face ${index} of ${choices.length})`;
  touch(job);
  setTimeout(() => runPlan(job, FACE_PLAN, stageIdx + 1), 300);
  return job;
}

function createGroupFaceJob(faces = 4): Job {
  const g = groupPhoto(faces);
  const job = createFaceJob(g.photo, g.boxes.map((box, i) => ({ index: i + 1, image_url: g.crops[i], box })));
  return job;
}

function createFaceJob(sourceUrl?: string, choices?: FaceChoice[]): Job {
  const d = { category: "Face", strokes: drawingFor("Face", { scale: 0.96 }) };
  const id = newId("job");
  if (choices) pendingChoices.set(id, choices);
  const job: Job = {
    id,
    kind: "face",
    status: "queued",
    title: "Your portrait",
    request: { image_url: sourceUrl, category: "Face" },
    stages: pendingStages(FACE_PLAN),
    artifacts: sourceUrl ? { source_image_url: sourceUrl } : {},
    created_at: iso(),
    updated_at: iso(),
  };
  addJob(job, d);
  emit({ type: "job", job });
  runPlan(job, FACE_PLAN, 0, sourceUrl);
  return job;
}

function seedJobs() {
  const now = Date.now();

  // Failed fish, 25 minutes ago.
  {
    const plan = animalPlan(false);
    const d = { category: "Fish", strokes: drawingFor("Fish") };
    const stages = completedStages(plan, d, now - 25 * 60_000);
    const vi = stages.findIndex((s) => s.name === "validate");
    stages[vi] = { ...stages[vi], status: "failed", detail: "Semantic score 0.41 < 0.70 after 3 retries: fins missing" };
    for (let i = vi + 1; i < stages.length; i++) {
      stages[i] = { name: stages[i].name, label: stages[i].label, status: "skipped" };
    }
    addJob(
      {
        id: "job_fish01",
        kind: "animal",
        status: "failed",
        title: "Swimming fish",
        request: { text: "make a new swimming fish", category: "Fish" },
        stages,
        artifacts: { source_image_url: photoSvg("Fish", d.strokes), lineart_url: strokesSvg(d.strokes, { width: 7 }) },
        error: "Validation failed 3 times (semantic score 0.41 < 0.70). Try another photo.",
        created_at: new Date(now - 25 * 60_000).toISOString(),
        updated_at: new Date(now - 24 * 60_000).toISOString(),
      },
      d,
    );
  }

  // Ready cat pick, 14 minutes ago.
  {
    const item = pickCatalogItem("Cat");
    const d = catalogStrokes.get(item.sample_id)!;
    addJob(
      {
        id: "job_cat001",
        kind: "pick",
        status: "ready",
        title: "Sitting cat",
        request: { text: "draw a cat", category: "Cat" },
        stages: completedStages(PICK_PLAN, d, now - 14 * 60_000),
        artifacts: finalArtifacts(d),
        trajectory: trajectoryStats(d),
        created_at: new Date(now - 14 * 60_000).toISOString(),
        updated_at: new Date(now - 14 * 60_000 + 2000).toISOString(),
      },
      d,
    );
  }

  // Ready new dog, 6 minutes ago (it is being drawn right now).
  {
    const plan = animalPlan(false);
    const d = { category: "Dog", strokes: drawingFor("Dog", { scale: 0.98 }) };
    addJob(
      {
        id: "job_dog001",
        kind: "animal",
        status: "ready",
        title: "New sitting dog",
        request: { text: "make a new sitting dog", category: "Dog" },
        stages: completedStages(plan, d, now - 6 * 60_000),
        artifacts: finalArtifacts(d),
        trajectory: trajectoryStats(d),
        created_at: new Date(now - 6 * 60_000).toISOString(),
        updated_at: new Date(now - 5 * 60_000).toISOString(),
      },
      d,
    );
  }

  // Group photo waiting for the user to pick a face, 40 seconds ago.
  {
    const d = { category: "Face", strokes: drawingFor("Face", { scale: 0.96 }) };
    const g = groupPhoto(4);
    const choices: FaceChoice[] = g.boxes.map((box, i) => ({ index: i + 1, image_url: g.crops[i], box }));
    const stages = pendingStages(FACE_PLAN);
    stages[0] = {
      ...stages[0],
      status: "done",
      backend: "mediapipe (CPU)",
      started_at: new Date(now - 40_000).toISOString(),
      finished_at: new Date(now - 38_500).toISOString(),
      detail: "4 faces found",
      artifact_url: g.photo,
    };
    stages[1] = { ...stages[1], status: "running", started_at: new Date(now - 38_000).toISOString(), detail: "4 faces found, pick one" };
    addJob(
      {
        id: "job_group1",
        kind: "face",
        status: "awaiting_input",
        title: "Group photo (4 faces)",
        request: { image_url: g.photo, category: "Face" },
        stages,
        artifacts: { source_image_url: g.photo },
        choices,
        created_at: new Date(now - 40_000).toISOString(),
        updated_at: new Date(now - 38_000).toISOString(),
      },
      d,
    );
  }
}

// ---------------------------------------------------------------------------------------------
// Runs

function makeNoisy(strokes: Polyline[], amp: number, seed: number): Polyline[] {
  const r = rng(seed);
  let ox = 0;
  let oy = 0;
  return strokes.map((line) =>
    line.map(([x, y]) => {
      ox = ox * 0.85 + (r() - 0.5) * amp;
      oy = oy * 0.85 + (r() - 0.5) * amp;
      return [x + ox, y + oy] as Point;
    }),
  );
}

function truncate(strokes: Polyline[], lengths: number[], dist: number): Polyline[] {
  const out: Polyline[] = [];
  let remaining = dist;
  for (let i = 0; i < strokes.length && remaining > 0; i++) {
    const line = strokes[i];
    if (remaining >= lengths[i]) {
      out.push(line);
      remaining -= lengths[i];
      continue;
    }
    const part: Polyline = [line[0]];
    for (let j = 1; j < line.length; j++) {
      const seg = Math.hypot(line[j][0] - line[j - 1][0], line[j][1] - line[j - 1][1]);
      if (seg >= remaining) {
        const t = seg === 0 ? 0 : remaining / seg;
        part.push([line[j - 1][0] + (line[j][0] - line[j - 1][0]) * t, line[j - 1][1] + (line[j][1] - line[j - 1][1]) * t]);
        remaining = 0;
        break;
      }
      part.push(line[j]);
      remaining -= seg;
    }
    out.push(part);
  }
  return out;
}

function strokeIndexAt(m: MockRun, p: number): number {
  let dist = p * m.total;
  for (let i = 0; i < m.lengths.length; i++) {
    if (dist <= m.lengths[i]) return i;
    dist -= m.lengths[i];
  }
  return m.lengths.length - 1;
}

function tick(m: MockRun) {
  const run = m.run;
  const elapsed = Date.now() - m.startedAt;
  const prevStatus = run.status;
  const prevPhase = run.phase;
  m.ticks += 1;
  if (elapsed < m.startupMs) {
    run.status = "starting";
    run.phase = elapsed < m.startupMs / 2 ? "HOME" : "APPROACH";
    run.progress = 0;
  } else {
    const p = Math.min(1, (elapsed - m.startupMs) / m.drawMs);
    run.status = "running";
    run.progress = p;
    const k = strokeIndexAt(m, p);
    run.phase = p >= 1 ? "RETRACT" : `DRAW_STROKE_${k + 1}`;
    m.err = Math.max(0.15, Math.min(4.5, m.err * 0.8 + (0.35 + Math.random() * 1.6) * 0.2 + (Math.random() < 0.03 ? 1.8 : 0)));
    m.errSq += m.err * m.err;
    m.errN += 1;
    m.errMax = Math.max(m.errMax, m.err);
    const executed = truncate(m.noisy, m.lengths, p * m.total);
    const last = executed[executed.length - 1];
    const tip = last ? last[last.length - 1] : m.noisy[0][0];
    const simSeconds = Math.round(((elapsed / 1000) * (run.mode === "live" ? 0.92 : 1)) * 10) / 10;
    run.metrics = { ...run.metrics, sim_seconds: simSeconds };
    emit({
      type: "telemetry",
      run_id: run.id,
      t: simSeconds,
      phase: run.phase,
      progress: p,
      tip: [tip[0], tip[1]],
      error_mm: Math.round(m.err * 100) / 100,
    });
    if (p >= 1) {
      run.status = "finished";
      run.phase = "DONE";
      run.finished_at = iso();
      run.metrics = {
        rmse_mm: Math.round(Math.sqrt(m.errSq / Math.max(1, m.errN)) * 100) / 100,
        max_error_mm: Math.round(m.errMax * 100) / 100,
        sim_seconds: simSeconds,
        jerk_rms: Math.round((run.controller === "act" ? 38 + Math.random() * 10 : 22 + Math.random() * 6) * 10) / 10,
      };
      if (m.timer) clearInterval(m.timer);
      emit({ type: "run", run });
      log("info", `Run finished: RMSE ${run.metrics.rmse_mm} mm over ${simSeconds} s`);
      return;
    }
  }
  if (run.status !== prevStatus || run.phase !== prevPhase || m.ticks % 10 === 0) {
    emit({ type: "run", run });
  }
}

function startRun(job: Job, controller: Controller, mode: RunMode, startedAgoMs = 0): Run {
  for (const other of runs.values()) {
    if (other.run.status === "running" || other.run.status === "starting") {
      if (other.timer) clearInterval(other.timer);
      other.run.status = "failed";
      other.run.error = "Stopped: a newer run took over the robot";
      other.run.finished_at = iso();
      emit({ type: "run", run: other.run });
      log("warn", "Previous run stopped so the new one can use the robot");
    }
  }
  const d = drawings.get(job.id) ?? { category: "Cat", strokes: drawingFor("Cat") };
  const amp = controller === "act" ? 0.0022 : 0.0012;
  const noisy = makeNoisy(d.strokes, amp, job.id.length * 31 + runs.size);
  const lengths = noisy.map(pathLength);
  const est = job.trajectory?.est_draw_seconds ?? estDrawSeconds(d.strokes);
  const run: Run = {
    id: newId("run"),
    job_id: job.id,
    controller,
    mode,
    status: "starting",
    phase: "HOME",
    progress: 0,
    cameras: mode === "live" ? ["WideFront", "PaperTop"] : [],
    prefix: `studio_${mode}_${Math.random().toString(36).slice(2, 8)}`,
    source: mode === "replay" ? "planned" : undefined,
    started_at: new Date(Date.now() - startedAgoMs).toISOString(),
  };
  const m: MockRun = {
    run,
    noisy,
    desired: d.strokes,
    lengths,
    total: lengths.reduce((a, b) => a + b, 0),
    startedAt: Date.now() - startedAgoMs,
    startupMs: 2200,
    drawMs: Math.min(48_000, Math.max(20_000, est * 500)),
    ticks: 0,
    err: 0.6,
    errSq: 0,
    errN: 0,
    errMax: 0,
  };
  runs.set(run.id, m);
  m.timer = setInterval(() => tick(m), 100);
  emit({ type: "run", run });
  return run;
}

// ---------------------------------------------------------------------------------------------
// Request understanding (very small stand-in for the chat backend)

const CATEGORY_WORDS: [RegExp, string][] = [
  [/\b(cat|cats|kitten|kitty)\b/, "Cat"],
  [/\b(dog|dogs|puppy|pup|doggo)\b/, "Dog"],
  [/\b(bird|birds|robin|parrot|sparrow|jay)\b/, "Bird"],
  [/\b(fish|goldfish|trout)\b/, "Fish"],
  [/\b(frog|frogs|toad)\b/, "Frog"],
  [/\b(horse|horses|pony|stallion|mare)\b/, "Horse"],
  [/\b(face|portrait|selfie)\b/, "Face"],
];

const POSES = ["sitting", "standing", "lying", "sleeping", "running", "jumping", "flying", "swimming", "perched"];

function detectCategory(text: string): string | undefined {
  for (const [re, cat] of CATEGORY_WORDS) if (re.test(text)) return cat;
  const named = CATALOG_CATEGORIES.find((c) => text.includes(c.toLowerCase()));
  return named;
}

function detectPose(text: string): string | undefined {
  return POSES.find((p) => text.includes(p));
}

function defaultPose(category: string): string {
  return { Bird: "perched", Fish: "swimming", Horse: "galloping", Face: "smiling" }[category] ?? "sitting";
}

function latestReadyJob(): Job | undefined {
  return [...jobs.values()].filter((j) => j.status === "ready").sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
}

function jobFromPrompt(prompt: string): Job {
  const text = prompt.toLowerCase();
  const category = detectCategory(text) ?? "Cat";
  const pose = detectPose(text) ?? defaultPose(category);
  const sampleMatch = prompt.match(/\b((?:oi|face)_[0-9a-f]{4,})\b/i);
  if (/\b(make|create|generate|new)\b/.test(text) && !sampleMatch) {
    if (category === "Face") return createFaceJob();
    return createAnimalJob(category, `New ${pose} ${category.toLowerCase()}`, prompt);
  }
  const title = sampleMatch ? undefined : `${capitalize(pose)} ${category.toLowerCase()}`;
  return createPickJob(category, title, sampleMatch?.[1], prompt);
}

// ---------------------------------------------------------------------------------------------
// Public mock API (mirrors lib/api.ts)

function ensureStarted() {
  if (started || typeof window === "undefined") return;
  started = true;
  buildCatalog();
  seedJobs();
  const dog = jobs.get("job_dog001");
  if (dog) {
    const est = dog.trajectory?.est_draw_seconds ?? 40;
    const drawMs = Math.min(48_000, Math.max(20_000, est * 500));
    startRun(dog, "act", "live", 2200 + drawMs * 0.32);
  }
  setInterval(() => {
    health = {
      ...health,
      gpu: { ...health.gpu, utilization: Math.round(90 + Math.random() * 9) },
    };
    emit({ type: "health", health });
  }, 10_000);
  setTimeout(() => log("warn", "PaperTop camera is streaming at 12 fps (target 30)"), 5000);
}

export function subscribeMock(listener: Listener): () => void {
  ensureStarted();
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function notFound(what: string): never {
  throw new ApiError(`${what} not found`, 404);
}

const ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];

/** Mirrors the backend's chat.parse_face_choice: "2", "the second one", "left", "rightmost". */
function parseFaceChoice(text: string, count: number): number | null {
  const t = text.toLowerCase();
  const num = t.match(/\b(\d{1,2})\b/);
  if (num) {
    const n = Number(num[1]);
    return n >= 1 && n <= count ? n : null;
  }
  const ord = ORDINALS.findIndex((o) => t.includes(o));
  if (ord >= 0 && ord < count) return ord + 1;
  if (/\b(last|rightmost|right)\b/.test(t)) return count;
  if (/\b(leftmost|left)\b/.test(t)) return 1;
  if (/\b(middle|center|centre)\b/.test(t)) return Math.ceil(count / 2);
  return null;
}

function waitingJob(): Job | undefined {
  return [...jobs.values()].find((j) => j.status === "awaiting_input");
}

export const mockApi = {
  async health(): Promise<Health> {
    ensureStarted();
    return delay(clone(health));
  },

  async chat(req: ChatRequest): Promise<ChatResponse> {
    ensureStarted();
    const text = req.message.trim().toLowerCase();
    const waiting = waitingJob();
    if (waiting) {
      const count = waiting.choices?.length ?? 0;
      const index = parseFaceChoice(text, count);
      if (index) {
        const job = resumeFaceJob(waiting, index);
        return delay({ reply: `Drawing face ${index} of ${count}. The line art is on its way.`, job: clone(job), suggestions: ["Draw it"] }, 400);
      }
    }
    if (/\bgroup\b/.test(text)) {
      const job = createGroupFaceJob(3);
      return delay({ reply: "Looking for faces in the group photo...", job: clone(job) }, 400);
    }
    if (/^(draw it|draw that|start|go|yes|draw this|do it|start drawing)\b/.test(text) || /\bdraw it\b/.test(text)) {
      const job = latestReadyJob();
      if (!job) {
        return delay({ reply: "Nothing is ready to draw yet. Ask me for an animal first.", suggestions: ["Draw a cat", "Make a new sitting dog"] });
      }
      const run = startRun(job, "act", health.isaac.running ? "live" : "replay");
      return delay(
        { reply: `Drawing "${job.title}" with the ACT policy (${run.mode}). Watch the Live view.`, job: clone(job), run: clone(run), suggestions: ["Draw a horse", "Make a new flying bird"] },
        500,
      );
    }
    const category = detectCategory(text);
    if (category === "Face" && !/\b(draw|pick|show)\b.*\b(face_[0-9a-f]+)/.test(text)) {
      return delay({ reply: "Attach a photo of a face (paperclip or drag and drop) and I will turn it into a portrait the robot can draw.", suggestions: ["Draw a cat", "Show me a horse"] });
    }
    if (category || /\b(draw|show|make|create|new)\b/.test(text)) {
      const job = jobFromPrompt(req.message);
      const reply =
        job.kind === "pick"
          ? `Here is a ${job.title.toLowerCase()} from the catalog. Say "draw it" when you are ready.`
          : `Making a ${job.title.toLowerCase()}: I will find an unused photo, outline it, validate it and extract pen strokes.`;
      return delay({ reply, job: clone(job), suggestions: job.kind === "pick" ? ["Draw it", "Make a new one", "Show me a frog"] : ["Draw it", "Draw a cat"] }, 450);
    }
    return delay({
      reply: "I can pick an animal from the catalog (\"draw a cat\"), make a new one from a photo (\"make a new sitting dog\"), or turn your face into a portrait (attach a photo).",
      suggestions: ["Draw a cat", "Make a new sitting dog", "Show me a horse"],
    });
  },

  async createTextJob(prompt: string): Promise<Job> {
    ensureStarted();
    return delay(clone(jobFromPrompt(prompt)));
  },

  async createImageJob(file: File, kind: ImageJobKind, category?: string): Promise<Job> {
    ensureStarted();
    const url = URL.createObjectURL(file);
    let resolved: JobKind = kind === "auto" ? "animal" : kind;
    if (kind === "auto" && /(face|selfie|portrait|headshot|group|team|family|friends|people)/i.test(file.name) && !category) resolved = "face";
    if (resolved === "face" && /(group|team|family|friends|people)/i.test(file.name)) {
      URL.revokeObjectURL(url);
      return delay(clone(createGroupFaceJob(4)), 600);
    }
    if (resolved === "face") return delay(clone(createFaceJob(url)), 600);
    const cat = category ?? detectCategory(file.name.toLowerCase()) ?? "Dog";
    return delay(clone(createAnimalJob(cat, `Your ${cat.toLowerCase()}`, undefined, url)), 600);
  },

  async listJobs(): Promise<Job[]> {
    ensureStarted();
    return delay(clone([...jobs.values()].sort((a, b) => b.created_at.localeCompare(a.created_at))));
  },

  async getJob(id: string): Promise<Job> {
    ensureStarted();
    const job = jobs.get(id) ?? notFound("Job");
    return delay(clone(job));
  },

  async chooseFace(id: string, index: number): Promise<Job> {
    ensureStarted();
    const job = jobs.get(id) ?? notFound("Job");
    if (job.status !== "awaiting_input") throw new ApiError("This job is not waiting for a face choice", 409);
    const count = job.choices?.length ?? 0;
    if (index < 1 || index > count) throw new ApiError(`No face ${index}; pick 1 to ${count}`, 400);
    return delay(clone(resumeFaceJob(job, index)), 500);
  },

  async drawJob(id: string, req: DrawRequest): Promise<Run> {
    ensureStarted();
    const job = jobs.get(id) ?? notFound("Job");
    if (job.status !== "ready") throw new ApiError("This job is not ready to draw yet", 409);
    const mode = req.mode ?? "live";
    if (mode === "live" && !health.isaac.running) throw new ApiError("Isaac Sim is not running; use replay mode", 409);
    return delay(clone(startRun(job, req.controller ?? "act", mode)), 400);
  },

  async listRuns(active: boolean): Promise<Run[]> {
    ensureStarted();
    const all = [...runs.values()].map((m) => m.run).reverse();
    return delay(clone(all.filter((r) => !active || r.status === "starting" || r.status === "running")), 80);
  },

  async getRun(id: string): Promise<Run> {
    ensureStarted();
    const m = runs.get(id) ?? notFound("Run");
    return delay(clone(m.run), 80);
  },

  async getTrace(id: string): Promise<Trace> {
    ensureStarted();
    const m = runs.get(id) ?? notFound("Run");
    const executed = m.run.status === "starting" ? [] : truncate(m.noisy, m.lengths, m.run.progress * m.total);
    return delay(clone({ desired: m.desired, executed, aspect: 1 }), 60);
  },

  async catalog(params: { category?: string; limit?: number } = {}): Promise<CatalogItem[]> {
    ensureStarted();
    const items = catalog.filter((c) => !params.category || c.category === params.category);
    return delay(clone(items.slice(0, params.limit ?? 48)), 300);
  },
};
