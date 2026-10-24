// Typed client for the WildTrace Studio backend (contract: app/API.md).
// In mock mode every call is served by the in-browser mock backend instead.

import { apiUrl, isMockMode } from "./config";
import { ApiError } from "./errors";
import { mockApi } from "./mock";
import type {
  CatalogItem,
  ChatRequest,
  ChatResponse,
  DrawRequest,
  Health,
  ImageJobKind,
  Job,
  Run,
  Trace,
} from "./types";

interface RequestOptions {
  method?: "GET" | "POST";
  json?: unknown;
  form?: FormData;
  timeoutMs?: number;
  signal?: AbortSignal;
}

function detailMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // FastAPI validation errors: [{loc, msg, type}]
      return detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d)))
        .join("; ");
    }
  }
  return fallback;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), opts.timeoutMs ?? 15_000);
  const onAbort = () => controller.abort();
  opts.signal?.addEventListener("abort", onAbort);
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      method: opts.method ?? (opts.json !== undefined || opts.form ? "POST" : "GET"),
      headers: opts.json !== undefined ? { "Content-Type": "application/json", Accept: "application/json" } : { Accept: "application/json" },
      body: opts.form ?? (opts.json !== undefined ? JSON.stringify(opts.json) : undefined),
      signal: controller.signal,
      cache: "no-store",
    });
  } catch (err) {
    if (opts.signal?.aborted) throw err;
    const timedOut = controller.signal.aborted;
    throw new ApiError(timedOut ? "The backend did not answer in time" : "Cannot reach the WildTrace backend", 0, true);
  } finally {
    clearTimeout(timeout);
    opts.signal?.removeEventListener("abort", onAbort);
  }
  const text = await res.text();
  let body: unknown = undefined;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!res.ok) {
    throw new ApiError(detailMessage(body, `${res.status} ${res.statusText || "error"}`), res.status);
  }
  return body as T;
}

function enc(id: string): string {
  return encodeURIComponent(id);
}

export const api = {
  health(signal?: AbortSignal): Promise<Health> {
    if (isMockMode()) return mockApi.health();
    return request<Health>("/api/health", { signal, timeoutMs: 6000 });
  },

  chat(req: ChatRequest): Promise<ChatResponse> {
    if (isMockMode()) return mockApi.chat(req);
    return request<ChatResponse>("/api/chat", { json: req, timeoutMs: 90_000 });
  },

  createTextJob(prompt: string): Promise<Job> {
    if (isMockMode()) return mockApi.createTextJob(prompt);
    return request<Job>("/api/jobs/text", { json: { prompt }, timeoutMs: 30_000 });
  },

  createImageJob(file: File, kind: ImageJobKind, category?: string): Promise<Job> {
    if (isMockMode()) return mockApi.createImageJob(file, kind, category);
    const form = new FormData();
    form.append("file", file, file.name || "upload.png");
    form.append("kind", kind);
    if (category) form.append("category", category);
    return request<Job>("/api/jobs/image", { form, timeoutMs: 120_000 });
  },

  listJobs(signal?: AbortSignal): Promise<Job[]> {
    if (isMockMode()) return mockApi.listJobs();
    return request<Job[]>("/api/jobs", { signal });
  },

  getJob(id: string, signal?: AbortSignal): Promise<Job> {
    if (isMockMode()) return mockApi.getJob(id);
    return request<Job>(`/api/jobs/${enc(id)}`, { signal });
  },

  chooseFace(id: string, index: number): Promise<Job> {
    if (isMockMode()) return mockApi.chooseFace(id, index);
    return request<Job>(`/api/jobs/${enc(id)}/choose`, { json: { index }, timeoutMs: 30_000 });
  },

  drawJob(id: string, req: DrawRequest): Promise<Run> {
    if (isMockMode()) return mockApi.drawJob(id, req);
    return request<Run>(`/api/jobs/${enc(id)}/draw`, { json: req, timeoutMs: 60_000 });
  },

  listRuns(active = false): Promise<Run[]> {
    if (isMockMode()) return mockApi.listRuns(active);
    return request<Run[]>(`/api/runs?active=${active}&limit=20`);
  },

  getRun(id: string, signal?: AbortSignal): Promise<Run> {
    if (isMockMode()) return mockApi.getRun(id);
    return request<Run>(`/api/runs/${enc(id)}`, { signal });
  },

  getTrace(id: string, signal?: AbortSignal): Promise<Trace> {
    if (isMockMode()) return mockApi.getTrace(id);
    return request<Trace>(`/api/runs/${enc(id)}/trace`, { signal });
  },

  catalog(params: { category?: string; limit?: number } = {}, signal?: AbortSignal): Promise<CatalogItem[]> {
    if (isMockMode()) return mockApi.catalog(params);
    const q = new URLSearchParams();
    if (params.category) q.set("category", params.category);
    q.set("limit", String(params.limit ?? 48));
    return request<CatalogItem[]>(`/api/catalog?${q.toString()}`, { signal });
  },

  /** MJPEG stream URL for an <img> tag (multipart/x-mixed-replace). */
  streamUrl(runId: string, camera: string, nonce = 0): string {
    const url = apiUrl(`/api/runs/${enc(runId)}/stream/${enc(camera)}`);
    return nonce ? `${url}?n=${nonce}` : url;
  },
};

export type Api = typeof api;
