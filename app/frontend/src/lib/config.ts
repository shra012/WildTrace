// Runtime configuration. NEXT_PUBLIC_* values are inlined at build time.

function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

export const API_BASE = trimSlash(process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000");

/** WebSocket URL: explicit NEXT_PUBLIC_WS_URL, else derived from the API base (http -> ws). */
export const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL || `${API_BASE.replace(/^http(s?):\/\//i, "ws$1://")}/ws`;

const ENV_MOCK = process.env.NEXT_PUBLIC_MOCK === "1" || process.env.NEXT_PUBLIC_MOCK === "true";

let mockCache: boolean | null = null;

/**
 * Mock mode is on when NEXT_PUBLIC_MOCK=1 at build time, or when the page is opened with
 * `?mock=1` (handy for demos of a production build). Only meaningful in the browser.
 */
export function isMockMode(): boolean {
  if (mockCache !== null) return mockCache;
  let fromUrl = false;
  if (typeof window !== "undefined") {
    try {
      const value = new URLSearchParams(window.location.search).get("mock");
      fromUrl = value === "1" || value === "true";
    } catch {
      fromUrl = false;
    }
    mockCache = ENV_MOCK || fromUrl;
    return mockCache;
  }
  return ENV_MOCK;
}

/** Resolves an API-relative file URL (e.g. `/files/x.png`) against the API base. */
export function fileUrl(url?: string | null): string | undefined {
  if (!url) return undefined;
  if (/^(https?:|data:|blob:)/i.test(url)) return url;
  return `${API_BASE}${url.startsWith("/") ? "" : "/"}${url}`;
}

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}
