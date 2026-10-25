// Application state and actions for WildTrace Studio. Components read slices with useStudio()
// and call the exported actions; all network side effects live here.

import { api } from "./api";
import { isMockMode } from "./config";
import { ApiError, errorMessage } from "./errors";
import { createStore, useStore } from "./store";
import type {
  CatalogItem,
  Controller,
  Health,
  ImageJobKind,
  Job,
  Run,
  RunMode,
  StudioEvent,
  TelemetryEvent,
} from "./types";

export type WsStatus = "idle" | "connecting" | "open" | "closed";
export type Focus = "chat" | "live" | "pipeline";
export type ToastLevel = "info" | "success" | "warn" | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  at: string;
  jobId?: string;
  runId?: string;
  imageUrl?: string;
  imageKind?: ImageJobKind;
  pending?: boolean;
  error?: boolean;
  retryText?: string;
}

export interface Toast {
  id: string;
  level: ToastLevel;
  message: string;
}

export interface StudioState {
  started: boolean;
  mock: boolean;
  health: Health | null;
  /** null until the first health check completes. */
  online: boolean | null;
  checkingHealth: boolean;
  nextRetryAt: number | null;
  wsStatus: WsStatus;
  jobs: Record<string, Job>;
  jobList: Job[];
  jobsLoaded: boolean;
  selectedJobId: string | null;
  runs: Record<string, Run>;
  activeRunId: string | null;
  controller: Controller;
  mode: RunMode;
  messages: ChatMessage[];
  chatBusy: boolean;
  suggestions: string[];
  toasts: Toast[];
  focus: Focus;
  galleryOpen: boolean;
  preview: { url: string; title: string } | null;
  drawingJobId: string | null;
  pendingDrawJobId: string | null;
  choosing: { jobId: string; index: number } | null;
}

export const DEFAULT_SUGGESTIONS = ["Draw a cat", "Make a new sitting dog", "Draw my face (upload a photo)"];

export const studio = createStore<StudioState>({
  started: false,
  mock: false,
  health: null,
  online: null,
  checkingHealth: false,
  nextRetryAt: null,
  wsStatus: "idle",
  jobs: {},
  jobList: [],
  jobsLoaded: false,
  selectedJobId: null,
  runs: {},
  activeRunId: null,
  controller: "act",
  mode: "live",
  messages: [],
  chatBusy: false,
  suggestions: DEFAULT_SUGGESTIONS,
  toasts: [],
  focus: "chat",
  galleryOpen: false,
  preview: null,
  drawingJobId: null,
  pendingDrawJobId: null,
  choosing: null,
});

export function useStudio<T>(selector: (s: StudioState) => T): T {
  return useStore(studio, selector);
}

// ---------------------------------------------------------------------------------------------
// Telemetry (kept out of the main store; ~10 Hz)

export const telemetryStore = createStore<{ latest: Record<string, TelemetryEvent> }>({ latest: {} });
const telemetryListeners = new Set<(e: TelemetryEvent) => void>();

export function onTelemetry(listener: (e: TelemetryEvent) => void): () => void {
  telemetryListeners.add(listener);
  return () => {
    telemetryListeners.delete(listener);
  };
}

export function useTelemetry(runId: string | null | undefined): TelemetryEvent | undefined {
  return useStore(telemetryStore, (s) => (runId ? s.latest[runId] : undefined));
}

// ---------------------------------------------------------------------------------------------
// Helpers

function uid(prefix = "m"): string {
  const rand =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10);
  return `${prefix}_${Date.now().toString(36)}${rand}`;
}

function nowIso(): string {
  return new Date().toISOString();
}

const STORAGE_KEY = "wildtrace-studio:v1";
const SESSION_KEY = "wildtrace-studio:session";

function safeGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSet(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage can be unavailable (private mode, quota); persistence is a convenience only.
  }
}

let sessionId: string | null = null;
function getSessionId(): string {
  if (sessionId) return sessionId;
  sessionId = safeGet(SESSION_KEY) ?? uid("s");
  safeSet(SESSION_KEY, sessionId);
  return sessionId;
}

interface Persisted {
  messages: ChatMessage[];
  selectedJobId: string | null;
  activeRunId: string | null;
  controller: Controller;
  mode: RunMode;
}

function loadPersisted(mock: boolean): Partial<Persisted> {
  const raw = safeGet(`${STORAGE_KEY}${mock ? ":mock" : ""}`);
  if (!raw) return {};
  try {
    const data = JSON.parse(raw) as Partial<Persisted>;
    return {
      messages: Array.isArray(data.messages) ? data.messages.filter((m) => !m.pending).slice(-60) : undefined,
      selectedJobId: typeof data.selectedJobId === "string" ? data.selectedJobId : null,
      activeRunId: typeof data.activeRunId === "string" ? data.activeRunId : null,
      controller: data.controller === "ik" ? "ik" : "act",
      mode: data.mode === "replay" ? "replay" : "live",
    };
  } catch {
    return {};
  }
}

let persistTimer: ReturnType<typeof setTimeout> | null = null;
function schedulePersist() {
  if (persistTimer) return;
  persistTimer = setTimeout(() => {
    persistTimer = null;
    const s = studio.get();
    const data: Persisted = {
      messages: s.messages
        .filter((m) => !m.pending)
        .slice(-60)
        .map((m) => (m.imageUrl?.startsWith("blob:") ? { ...m, imageUrl: undefined } : m)),
      selectedJobId: s.selectedJobId,
      activeRunId: s.activeRunId,
      controller: s.controller,
      mode: s.mode,
    };
    safeSet(`${STORAGE_KEY}${s.mock ? ":mock" : ""}`, JSON.stringify(data));
  }, 400);
}

// ---------------------------------------------------------------------------------------------
// Toasts

const toastTimers = new Map<string, ReturnType<typeof setTimeout>>();

export function pushToast(level: ToastLevel, message: string, ttlMs = level === "error" ? 9000 : 5500) {
  const existing = studio.get().toasts.find((t) => t.message === message && t.level === level);
  if (existing) return;
  const toast: Toast = { id: uid("t"), level, message };
  studio.set((s) => ({ toasts: [...s.toasts, toast].slice(-5) }));
  toastTimers.set(
    toast.id,
    setTimeout(() => dismissToast(toast.id), ttlMs),
  );
}

export function dismissToast(id: string) {
  const timer = toastTimers.get(id);
  if (timer) clearTimeout(timer);
  toastTimers.delete(id);
  studio.set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
}

// ---------------------------------------------------------------------------------------------
// Jobs and runs

function sortJobs(jobs: Record<string, Job>): Job[] {
  return Object.values(jobs).sort((a, b) => b.created_at.localeCompare(a.created_at));
}

function isNewer(next: Job, prev?: Job): boolean {
  if (!prev) return true;
  return next.updated_at >= prev.updated_at;
}

export function upsertJob(job: Job) {
  const prev = studio.get().jobs[job.id];
  if (!isNewer(job, prev)) return;
  studio.set((s) => {
    const jobs = { ...s.jobs, [job.id]: job };
    return { jobs, jobList: sortJobs(jobs) };
  });
  const s = studio.get();
  if (s.pendingDrawJobId === job.id) {
    if (job.status === "ready") {
      studio.set({ pendingDrawJobId: null });
      void drawJob(job.id);
    } else if (job.status === "failed") {
      studio.set({ pendingDrawJobId: null });
      pushToast("error", `"${job.title}" failed, so it will not be drawn`);
    }
  }
  if (job.status === "awaiting_input" && (job.choices?.length ?? 0) > 1) announceFaces(job);
  if (prev && prev.status !== job.status) {
    if (job.status === "ready" && prev.status !== "ready") pushToast("success", `"${job.title}" is ready to draw`, 4000);
    if (job.status === "failed") pushToast("error", `"${job.title}" failed${job.error ? `: ${job.error}` : ""}`);
  }
}

function setJobs(list: Job[]) {
  studio.set((s) => {
    const jobs: Record<string, Job> = { ...s.jobs };
    for (const job of list) {
      if (isNewer(job, jobs[job.id])) jobs[job.id] = job;
    }
    const jobList = sortJobs(jobs);
    const selectedJobId = s.selectedJobId && jobs[s.selectedJobId] ? s.selectedJobId : (jobList[0]?.id ?? null);
    return { jobs, jobList, jobsLoaded: true, selectedJobId };
  });
}

function isLive(run?: Run | null): boolean {
  return !!run && (run.status === "starting" || run.status === "running");
}

export function upsertRun(run: Run, makeActive = false) {
  studio.set((s) => {
    const runs = { ...s.runs, [run.id]: run };
    const active = s.activeRunId ? runs[s.activeRunId] : undefined;
    let activeRunId = s.activeRunId;
    if (makeActive || !active) activeRunId = run.id;
    else if (isLive(run) && run.id !== active.id && (!isLive(active) || run.started_at > active.started_at)) activeRunId = run.id;
    return { runs, activeRunId };
  });
}

export function selectJob(id: string | null, focusPipeline = false) {
  studio.set((s) => ({ selectedJobId: id, focus: focusPipeline ? "pipeline" : s.focus }));
  schedulePersist();
  if (id && !studio.get().jobs[id]) void refreshJob(id);
}

export function setActiveRun(id: string) {
  studio.set({ activeRunId: id, focus: "live" });
  schedulePersist();
  if (!studio.get().runs[id]) void refreshRun(id);
}

async function refreshJob(id: string) {
  try {
    upsertJob(await api.getJob(id));
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      studio.set((s) => (s.selectedJobId === id ? { selectedJobId: s.jobList[0]?.id ?? null } : {}));
    }
  }
}

async function refreshRun(id: string) {
  try {
    upsertRun(await api.getRun(id));
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      studio.set((s) => (s.activeRunId === id ? { activeRunId: null } : {}));
    }
  }
}

export async function refreshJobs() {
  try {
    setJobs(await api.listJobs());
  } catch (err) {
    if (err instanceof ApiError && err.offline) markOffline();
  }
}

export function setController(controller: Controller) {
  studio.set({ controller });
  schedulePersist();
}

/** Set once the user picks a mode; until then the mode follows whether Isaac Sim is running. */
let modeChosen = false;

export function setMode(mode: RunMode) {
  modeChosen = true;
  studio.set({ mode });
  schedulePersist();
}

export function setFocus(focus: Focus) {
  studio.set({ focus });
}

export function setGalleryOpen(open: boolean) {
  studio.set({ galleryOpen: open });
}

export function openPreview(url: string, title: string) {
  studio.set({ preview: { url, title } });
}

export function closePreview() {
  studio.set({ preview: null });
}

const FACE_SUGGESTIONS = ["The left one", "The second one", "Rightmost"];
const announced = new Set<string>();

function facesPrompt(job: Job): string {
  const n = job.choices?.length ?? 0;
  return `I found ${n} faces in your photo. Which one should I draw? Click a face below, or type "2", "the second one" or "left".`;
}

/** Posts the "which face?" question once per job (unless a chat reply already carries the job). */
function announceFaces(job: Job) {
  if (announced.has(job.id)) return;
  announced.add(job.id);
  const s = studio.get();
  const asked = s.messages.some((m) => m.jobId === job.id && m.text.startsWith("I found"));
  if (!asked) addMessage({ role: "assistant", text: facesPrompt(job), jobId: job.id });
  studio.set({ suggestions: FACE_SUGGESTIONS });
}

export async function chooseFace(jobId: string, index: number) {
  if (studio.get().choosing) return;
  studio.set({ choosing: { jobId, index } });
  try {
    const job = await api.chooseFace(jobId, index);
    upsertJob(job);
    const n = job.choices?.length;
    addMessage({ role: "assistant", text: `Drawing face ${index}${n ? ` of ${n}` : ""}. The line art is on its way.`, jobId });
    studio.set({ suggestions: ["Draw it", "Draw a cat"] });
  } catch (err) {
    if (err instanceof ApiError && err.offline) markOffline();
    pushToast("error", `Could not pick face ${index}: ${errorMessage(err)}`);
    void refreshJob(jobId);
  } finally {
    studio.set({ choosing: null });
  }
}

function addMessage(msg: Omit<ChatMessage, "id" | "at"> & { id?: string }): string {
  const full: ChatMessage = { id: msg.id ?? uid(), at: nowIso(), ...msg };
  studio.set((s) => ({ messages: [...s.messages, full].slice(-80) }));
  schedulePersist();
  return full.id;
}

function updateMessage(id: string, patch: Partial<ChatMessage>) {
  studio.set((s) => ({ messages: s.messages.map((m) => (m.id === id ? { ...m, ...patch } : m)) }));
  schedulePersist();
}

export function clearChat() {
  studio.set({ messages: [], suggestions: DEFAULT_SUGGESTIONS });
  schedulePersist();
}

export async function drawJob(jobId: string) {
  const s = studio.get();
  if (s.drawingJobId) return;
  const job = s.jobs[jobId];
  studio.set({ drawingJobId: jobId });
  try {
    const run = await api.drawJob(jobId, { controller: s.controller, mode: s.mode });
    upsertRun(run, true);
    studio.set({ focus: "live" });
    schedulePersist();
    addMessage({
      role: "assistant",
      text: `Drawing ${job ? `"${job.title}"` : "the trajectory"} with ${run.controller.toUpperCase()} (${run.mode}).`,
      runId: run.id,
      jobId,
    });
  } catch (err) {
    pushToast("error", `Could not start drawing: ${errorMessage(err)}`);
    if (err instanceof ApiError && err.offline) markOffline();
  } finally {
    studio.set({ drawingJobId: null });
  }
}

/** Draw now if the job is ready, otherwise as soon as it becomes ready. */
export function drawWhenReady(jobId: string) {
  const job = studio.get().jobs[jobId];
  if (job?.status === "ready") void drawJob(jobId);
  else {
    studio.set({ pendingDrawJobId: jobId });
    pushToast("info", "Will start drawing as soon as the trajectory is ready", 4000);
  }
}

function applyChatResult(job?: Job, run?: Run) {
  if (job) {
    upsertJob(job);
    selectJob(job.id);
  }
  if (run) {
    upsertRun(run, true);
    studio.set({ focus: "live" });
  }
}

export async function sendChat(text: string) {
  const message = text.trim();
  if (!message || studio.get().chatBusy) return;
  addMessage({ role: "user", text: message });
  const pendingId = addMessage({ role: "assistant", text: "", pending: true });
  studio.set({ chatBusy: true });
  try {
    const res = await api.chat({ message, session_id: getSessionId() });
    if (res.job?.status === "awaiting_input") announced.add(res.job.id);
    applyChatResult(res.job, res.run);
    updateMessage(pendingId, { pending: false, text: res.reply, jobId: res.job?.id, runId: res.run?.id });
    const waiting = res.job?.status === "awaiting_input" && (res.job.choices?.length ?? 0) > 1;
    studio.set({ suggestions: res.suggestions?.length ? res.suggestions : waiting ? FACE_SUGGESTIONS : DEFAULT_SUGGESTIONS });
  } catch (err) {
    if (err instanceof ApiError && err.offline) markOffline();
    updateMessage(pendingId, {
      pending: false,
      error: true,
      text: `I could not reach the assistant: ${errorMessage(err)}`,
      retryText: message,
    });
  } finally {
    studio.set({ chatBusy: false });
  }
}

export async function retryMessage(id: string) {
  const msg = studio.get().messages.find((m) => m.id === id);
  if (!msg?.retryText) return;
  studio.set((s) => ({ messages: s.messages.filter((m) => m.id !== id) }));
  // Drop the user message that preceded the failed reply; sendChat adds it again.
  studio.set((s) => {
    const last = s.messages[s.messages.length - 1];
    return last && last.role === "user" && last.text === msg.retryText ? { messages: s.messages.slice(0, -1) } : {};
  });
  await sendChat(msg.retryText);
}

const KIND_LABEL: Record<ImageJobKind, string> = { auto: "drawing", face: "portrait", animal: "animal drawing" };

export async function sendImage(file: File, kind: ImageJobKind, category?: string, text?: string) {
  if (studio.get().chatBusy) return;
  const preview = URL.createObjectURL(file);
  addMessage({ role: "user", text: text?.trim() || "", imageUrl: preview, imageKind: kind });
  const pendingId = addMessage({ role: "assistant", text: "", pending: true });
  studio.set({ chatBusy: true });
  try {
    const job = await api.createImageJob(file, kind, category);
    const waiting = job.status === "awaiting_input" && (job.choices?.length ?? 0) > 1;
    // The reply below carries the job; a later switch to awaiting_input posts the face question.
    if (waiting) announced.add(job.id);
    applyChatResult(job);
    studio.set({ focus: waiting ? "chat" : "pipeline" });
    const what = job.kind === "face" ? "portrait" : job.kind === "animal" ? "animal drawing" : KIND_LABEL[kind];
    updateMessage(pendingId, {
      pending: false,
      text: waiting ? facesPrompt(job) : `Got your photo. I am turning it into a ${what}; follow the stages in the Pipeline panel.`,
      jobId: job.id,
    });
    studio.set({ suggestions: waiting ? FACE_SUGGESTIONS : ["Draw it", "Draw a cat"] });
  } catch (err) {
    if (err instanceof ApiError && err.offline) markOffline();
    updateMessage(pendingId, { pending: false, error: true, text: `Upload failed: ${errorMessage(err)}` });
  } finally {
    studio.set({ chatBusy: false });
  }
}

export async function pickFromCatalog(item: CatalogItem, draw: boolean) {
  try {
    const job = await api.createTextJob(`draw ${item.category} ${item.sample_id}`);
    upsertJob(job);
    selectJob(job.id);
    studio.set({ galleryOpen: false, focus: draw ? "live" : "pipeline" });
    addMessage({
      role: "assistant",
      text: `Picked ${item.subcategory ? `${item.subcategory} ` : ""}${item.category.toLowerCase()} ${item.sample_id} from the gallery.`,
      jobId: job.id,
    });
    if (draw) drawWhenReady(job.id);
  } catch (err) {
    if (err instanceof ApiError && err.offline) markOffline();
    pushToast("error", `Could not use ${item.sample_id}: ${errorMessage(err)}`);
  }
}

// ---------------------------------------------------------------------------------------------
// Events (WebSocket or mock bus)

export function handleEvent(event: StudioEvent) {
  switch (event.type) {
    case "job":
      upsertJob(event.job);
      break;
    case "run":
      upsertRun(event.run);
      followRun(event.run);
      break;
    case "telemetry": {
      telemetryStore.set((s) => ({ latest: { ...s.latest, [event.run_id]: event } }));
      for (const listener of telemetryListeners) listener(event);
      break;
    }
    case "log":
      if (event.level === "warn" || event.level === "error") pushToast(event.level, event.message);
      break;
    case "health":
      setHealth(event.health);
      break;
  }
}

/** Show a run that started somewhere else, unless the one on screen is still drawing. */
function followRun(run: Run) {
  const s = studio.get();
  if (run.id === s.activeRunId || (run.status !== "starting" && run.status !== "running")) return;
  if (s.activeRunId && isLive(s.runs[s.activeRunId])) return;
  setActiveRun(run.id);
  selectJob(run.job_id);
}

async function adoptActiveRun() {
  try {
    const [newest] = await api.listRuns(true);
    if (newest) {
      upsertRun(newest);
      followRun(newest);
    }
  } catch {
    // Health checks report the backend being offline; nothing to add here.
  }
}

export function setWsStatus(wsStatus: WsStatus) {
  const prev = studio.get().wsStatus;
  studio.set({ wsStatus });
  if (wsStatus === "open" && prev !== "open") {
    // Catch up on anything missed while disconnected.
    void refreshJobs();
    const active = studio.get().activeRunId;
    if (active) void refreshRun(active);
  }
}

// ---------------------------------------------------------------------------------------------
// Health and polling

let healthTimer: ReturnType<typeof setTimeout> | null = null;
let healthFailures = 0;

function setHealth(health: Health) {
  const wasOffline = studio.get().online === false;
  healthFailures = 0;
  studio.set({ health, online: true, nextRetryAt: null });
  if (!modeChosen && health.isaac) {
    const mode: RunMode = health.isaac.running ? "live" : "replay";
    if (studio.get().mode !== mode) studio.set({ mode });
  }
  if (wasOffline) {
    pushToast("success", "Reconnected to the WildTrace backend", 3500);
    void refreshJobs();
  }
}

function markOffline() {
  if (studio.get().online !== false) {
    studio.set({ online: false });
    scheduleHealth(1500);
  }
}

function scheduleHealth(ms: number) {
  if (healthTimer) clearTimeout(healthTimer);
  healthTimer = setTimeout(() => void checkHealth(), ms);
}

export async function checkHealth() {
  if (healthTimer) clearTimeout(healthTimer);
  healthTimer = null;
  studio.set({ checkingHealth: true });
  try {
    const health = await api.health();
    setHealth(health);
    scheduleHealth(studio.get().wsStatus === "open" ? 20_000 : 10_000);
  } catch {
    healthFailures += 1;
    const wait = Math.min(30_000, 2000 * 2 ** Math.min(4, healthFailures - 1));
    studio.set({ online: false, nextRetryAt: Date.now() + wait });
    scheduleHealth(wait);
  } finally {
    studio.set({ checkingHealth: false });
  }
}

export function retryNow() {
  void checkHealth().then(() => {
    if (studio.get().online) void refreshJobs();
  });
}

let pollTimer: ReturnType<typeof setInterval> | null = null;
let pollTick = 0;

function poll() {
  const s = studio.get();
  if (s.online === false || (typeof document !== "undefined" && document.hidden)) return;
  pollTick += 1;
  const wsOpen = s.wsStatus === "open";
  if (!wsOpen) {
    // Without the event stream, poll whatever is in flight.
    s.jobList
      .filter((j) => j.status === "queued" || j.status === "running")
      .slice(0, 4)
      .forEach((j) => void refreshJob(j.id));
    if (s.activeRunId && (isLive(s.runs[s.activeRunId]) || !s.runs[s.activeRunId])) void refreshRun(s.activeRunId);
    if (pollTick % 6 === 0) void refreshJobs();
  } else {
    if (s.activeRunId && isLive(s.runs[s.activeRunId]) && pollTick % 3 === 0) void refreshRun(s.activeRunId);
    if (pollTick % 12 === 0) void refreshJobs();
  }
}

/** Starts health checks, job loading and polling. Returns a cleanup function. */
export function startStudio(): () => void {
  const mock = isMockMode();
  const persisted = loadPersisted(mock);
  studio.set({
    started: true,
    mock,
    messages: persisted.messages ?? [],
    selectedJobId: persisted.selectedJobId ?? null,
    activeRunId: persisted.activeRunId ?? null,
    controller: persisted.controller ?? "act",
    mode: persisted.mode ?? "live",
  });
  void checkHealth();
  void refreshJobs();
  if (persisted.activeRunId) void refreshRun(persisted.activeRunId);
  void adoptActiveRun();
  pollTimer = setInterval(poll, 2500);
  const onVisible = () => {
    if (!document.hidden && studio.get().online === false) retryNow();
  };
  document.addEventListener("visibilitychange", onVisible);
  window.addEventListener("online", retryNow);
  return () => {
    if (pollTimer) clearInterval(pollTimer);
    if (healthTimer) clearTimeout(healthTimer);
    pollTimer = null;
    healthTimer = null;
    document.removeEventListener("visibilitychange", onVisible);
    window.removeEventListener("online", retryNow);
  };
}
