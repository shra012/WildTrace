import { cx } from "@/lib/cx";
import { shortBackend } from "@/lib/format";
import type { JobKind, JobStatus, RunStatus } from "@/lib/types";
import { LibraryIcon, PawIcon, UserIcon } from "./icons";

export function JobStatusBadge({ status, size = "sm" }: { status: JobStatus; size?: "xs" | "sm" }) {
  const map: Record<JobStatus, { cls: string; label: string }> = {
    queued: { cls: "badge-ghost", label: "Queued" },
    running: { cls: "badge-info badge-soft", label: "Running" },
    awaiting_input: { cls: "badge-warning", label: "Pick a face" },
    ready: { cls: "badge-success badge-soft", label: "Ready" },
    failed: { cls: "badge-error badge-soft", label: "Failed" },
  };
  const { cls, label } = map[status] ?? map.queued;
  return (
    <span className={cx("badge gap-1 whitespace-nowrap font-medium", size === "xs" ? "badge-xs" : "badge-sm", cls)}>
      {status === "running" && <span className="loading loading-spinner loading-xs scale-75" />}
      {label}
    </span>
  );
}

export function RunStatusBadge({ status }: { status: RunStatus }) {
  const map: Record<RunStatus, { cls: string; label: string }> = {
    starting: { cls: "badge-info badge-soft", label: "Starting" },
    running: { cls: "badge-primary badge-soft", label: "Drawing" },
    finished: { cls: "badge-success badge-soft", label: "Finished" },
    failed: { cls: "badge-error badge-soft", label: "Failed" },
  };
  const { cls, label } = map[status] ?? map.starting;
  return (
    <span className={cx("badge badge-sm gap-1.5 whitespace-nowrap font-medium", cls)}>
      {(status === "running" || status === "starting") && <span className="status status-primary animate-pulse" />}
      {label}
    </span>
  );
}

const KIND: Record<JobKind, { label: string; Icon: typeof PawIcon; cls: string }> = {
  pick: { label: "Catalog pick", Icon: LibraryIcon, cls: "text-info" },
  animal: { label: "New animal", Icon: PawIcon, cls: "text-accent" },
  face: { label: "Portrait", Icon: UserIcon, cls: "text-secondary" },
};

export function KindBadge({ kind }: { kind: JobKind }) {
  const k = KIND[kind] ?? KIND.pick;
  return (
    <span className="badge badge-sm badge-ghost gap-1 whitespace-nowrap">
      <k.Icon size={12} className={k.cls} />
      {k.label}
    </span>
  );
}

export function KindIcon({ kind, size = 16, className }: { kind: JobKind; size?: number; className?: string }) {
  const k = KIND[kind] ?? KIND.pick;
  return <k.Icon size={size} className={cx(k.cls, className)} />;
}

export function backendTone(name: string | undefined): string {
  const n = (name ?? "").toLowerCase();
  if (/flux|gpu|cuda/.test(n)) return "badge-accent";
  if (/openrouter|anthropic|gemini|claude|ollama|gpt/.test(n)) return "badge-secondary";
  return "badge-ghost";
}

export function BackendBadge({ backend }: { backend?: string }) {
  if (!backend) return null;
  return (
    <span className={cx("badge badge-xs badge-soft max-w-full truncate font-mono", backendTone(backend))} title={backend}>
      {backend.length > 34 ? shortBackend(backend) : backend}
    </span>
  );
}
