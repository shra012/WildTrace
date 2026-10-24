"use client";

import { fileUrl } from "@/lib/config";
import { cx } from "@/lib/cx";
import { duration } from "@/lib/format";
import { drawJob, selectJob, useStudio } from "@/lib/studio";
import { JobStatusBadge, KindBadge, KindIcon } from "../badges";
import { LayersIcon, PlayIcon } from "../icons";
import FacePicker from "./FacePicker";

export function jobThumb(job: { artifacts: { svg_url?: string; lineart_url?: string; source_image_url?: string }; request?: { image_url?: string } }) {
  return fileUrl(job.artifacts.svg_url ?? job.artifacts.lineart_url ?? job.artifacts.source_image_url ?? job.request?.image_url);
}

export default function JobCard({ jobId }: { jobId: string }) {
  const job = useStudio((s) => s.jobs[jobId]);
  const selected = useStudio((s) => s.selectedJobId === jobId);
  const drawing = useStudio((s) => s.drawingJobId === jobId);
  const pendingDraw = useStudio((s) => s.pendingDrawJobId === jobId);

  if (!job) {
    return (
      <div className="mt-2 flex w-full max-w-sm items-center gap-3 rounded-box border border-base-content/10 bg-base-200/60 p-3">
        <div className="skeleton h-14 w-14 shrink-0" />
        <div className="flex-1 space-y-2">
          <div className="skeleton h-3 w-2/3" />
          <div className="skeleton h-3 w-1/3" />
        </div>
      </div>
    );
  }

  const done = job.stages.filter((s) => s.status === "done" || s.status === "skipped").length;
  const total = job.stages.length || 1;
  const running = job.stages.find((s) => s.status === "running");
  const thumb = jobThumb(job);

  return (
    <div
      className={cx(
        "animate-rise mt-2 w-full max-w-sm rounded-box border bg-base-200/70 p-3 transition-colors",
        selected ? "border-primary/40" : "border-base-content/10",
      )}
    >
      <div className="flex gap-3">
        <button
          type="button"
          onClick={() => selectJob(job.id, true)}
          className="checker grid h-16 w-16 shrink-0 place-items-center overflow-hidden rounded-lg ring-1 ring-base-content/10"
          aria-label={`Open ${job.title} in the pipeline`}
        >
          {thumb ? <img src={thumb} alt="" className="h-full w-full object-contain" /> : <KindIcon kind={job.kind} size={22} className="opacity-50" />}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-2">
            <div className="min-w-0 flex-1 truncate font-medium leading-snug">{job.title}</div>
            <JobStatusBadge status={job.status} size="xs" />
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <KindBadge kind={job.kind} />
            {job.trajectory && (
              <span className="text-xs text-base-content/60">
                {job.trajectory.stroke_count} strokes, ~{duration(job.trajectory.est_draw_seconds)}
              </span>
            )}
          </div>
          {(job.status === "running" || job.status === "queued") && (
            <div className="mt-2">
              <progress className="progress progress-info h-1.5 w-full" value={done} max={total} />
              <div className="mt-1 truncate text-xs text-base-content/60">
                {running ? running.label : "Queued"} ({done}/{total})
              </div>
            </div>
          )}
          {job.status === "awaiting_input" && (
            <div className="mt-1.5 text-xs text-warning">{job.choices?.length ?? 0} faces found. Pick the one to draw.</div>
          )}
          {job.status === "failed" && job.error && <div className="mt-1.5 line-clamp-2 text-xs text-error">{job.error}</div>}
        </div>
      </div>
      {job.status === "awaiting_input" && (job.choices?.length ?? 0) > 0 && (
        <div className="mt-3">
          <FacePicker job={job} compact />
        </div>
      )}
      <div className="mt-3 flex gap-2">
        <button type="button" className="btn btn-ghost btn-xs gap-1" onClick={() => selectJob(job.id, true)}>
          <LayersIcon size={13} /> Pipeline
        </button>
        <button
          type="button"
          className="btn btn-primary btn-xs ml-auto gap-1"
          disabled={job.status !== "ready" || drawing}
          onClick={() => drawJob(job.id)}
          title={job.status === "ready" ? "Draw with the robot" : "Available when the trajectory is ready"}
        >
          {drawing ? <span className="loading loading-spinner loading-xs" /> : <PlayIcon size={12} />}
          {pendingDraw ? "Draws when ready" : "Draw"}
        </button>
      </div>
    </div>
  );
}
