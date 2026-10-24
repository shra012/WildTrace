export function relativeTime(iso: string | undefined, now: number): string {
  if (!iso || !now) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.round((now - t) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  return new Date(t).toLocaleDateString();
}

export function clockTime(iso: string | undefined): string {
  if (!iso) return "";
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return "";
  return t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function duration(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isFinite(seconds)) return "-";
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

export function stageDuration(start?: string, end?: string, now = 0): string {
  if (!start) return "";
  const a = Date.parse(start);
  const b = end ? Date.parse(end) : now;
  if (Number.isNaN(a) || !b || Number.isNaN(b)) return "";
  return duration(Math.max(0, (b - a) / 1000));
}

export function formatNumber(n: number | undefined, digits = 2): string {
  if (n === undefined || n === null || !Number.isFinite(n)) return "-";
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

/** "openrouter: google/gemini-2.5-flash-image" -> "openrouter"; "flux-schnell (GPU)" -> "flux-schnell". */
export function shortBackend(name: string | undefined): string {
  if (!name) return "";
  return name.split(":")[0].replace(/\s*\(.*\)\s*$/, "").trim();
}

export function phaseLabel(phase: string | undefined): string {
  if (!phase) return "";
  return phase
    .toLowerCase()
    .split("_")
    .map((w) => (/^\d+$/.test(w) ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
}
