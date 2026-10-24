"use client";

import { formatNumber, phaseLabel } from "@/lib/format";
import { setActiveRun, useStudio } from "@/lib/studio";
import { RunStatusBadge } from "../badges";
import { RobotIcon, VideoIcon } from "../icons";

export default function RunCard({ runId }: { runId: string }) {
  const run = useStudio((s) => s.runs[runId]);
  const job = useStudio((s) => (run ? s.jobs[run.job_id] : undefined));
  if (!run) return null;
  const live = run.status === "running" || run.status === "starting";
  return (
    <div className="animate-rise mt-2 w-full max-w-sm rounded-box border border-base-content/10 bg-base-200/70 p-3">
      <div className="flex items-center gap-2">
        <span className="grid h-8 w-8 place-items-center rounded-lg bg-primary/15 text-primary">
          <RobotIcon size={16} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">{job?.title ?? "Robot run"}</div>
          <div className="text-xs text-base-content/60">
            {`${run.controller.toUpperCase()}, ${run.mode}${run.phase ? `, ${phaseLabel(run.phase)}` : ""}`}
          </div>
        </div>
        <RunStatusBadge status={run.status} />
      </div>
      <progress className="progress progress-primary mt-3 h-1.5 w-full" value={Math.round(run.progress * 100)} max={100} />
      <div className="mt-2 flex items-center gap-2 text-xs text-base-content/60">
        <span className="font-mono">{Math.round(run.progress * 100)}%</span>
        {run.metrics?.rmse_mm !== undefined && <span>RMSE {formatNumber(run.metrics.rmse_mm)} mm</span>}
        {run.error && <span className="truncate text-error">{run.error}</span>}
        <button type="button" className="btn btn-ghost btn-xs ml-auto gap-1" onClick={() => setActiveRun(run.id)}>
          <VideoIcon size={13} /> {live ? "Watch live" : "View"}
        </button>
      </div>
    </div>
  );
}
