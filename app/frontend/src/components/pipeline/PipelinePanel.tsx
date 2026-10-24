"use client";

import { fileUrl } from "@/lib/config";
import { cx } from "@/lib/cx";
import { clockTime, duration, relativeTime, stageDuration } from "@/lib/format";
import { drawJob, openPreview, selectJob, setFocus, useStudio } from "@/lib/studio";
import type { Job, Stage, StageStatus } from "@/lib/types";
import { useNow } from "@/hooks/useNow";
import { BackendBadge, JobStatusBadge, KindBadge, KindIcon } from "../badges";
import FacePicker from "../cards/FacePicker";
import { jobThumb } from "../cards/JobCard";
import {
  AlertIcon,
  CheckIcon,
  ChevronDownIcon,
  CircleIcon,
  DownloadIcon,
  ImageIcon,
  LayersIcon,
  MinusIcon,
  PlayIcon,
  VideoIcon,
  XIcon,
} from "../icons";

function StageIcon({ status }: { status: StageStatus }) {
  const base = "grid h-6 w-6 place-items-center rounded-full";
  switch (status) {
    case "done":
      return (
        <span className={cx(base, "bg-success text-success-content")}>
          <CheckIcon size={14} strokeWidth={2.6} />
        </span>
      );
    case "running":
      return (
        <span className={cx(base, "animate-pulse-ring bg-info/20 text-info")}>
          <span className="loading loading-spinner loading-xs" />
        </span>
      );
    case "failed":
      return (
        <span className={cx(base, "bg-error text-error-content")}>
          <XIcon size={14} strokeWidth={2.6} />
        </span>
      );
    case "skipped":
      return (
        <span className={cx(base, "bg-base-300 text-base-content/50")}>
          <MinusIcon size={14} />
        </span>
      );
    default:
      return (
        <span className={cx(base, "border border-base-content/20 text-base-content/30")}>
          <CircleIcon size={8} />
        </span>
      );
  }
}

function StageItem({ stage, job, first, last, now }: { stage: Stage; job: Job; first: boolean; last: boolean; now: number }) {
  const art = fileUrl(stage.artifact_url);
  const lineCls = (s: StageStatus) => (s === "done" ? "bg-success/60" : s === "failed" ? "bg-error/60" : "bg-base-content/15");
  const choosing = stage.name === "choose_face" && job.status === "awaiting_input" && (job.choices?.length ?? 0) > 0;
  const took = stageDuration(stage.started_at, stage.finished_at, stage.status === "running" ? now : 0);
  return (
    <li className="flex gap-3">
      <div className="flex w-5 shrink-0 flex-col items-center">
        <div className={cx("h-1.5 w-0.5", first ? "bg-transparent" : lineCls(stage.status))} />
        <StageIcon status={stage.status} />
        <div className={cx("w-0.5 flex-1", last ? "bg-transparent" : lineCls(stage.status))} />
      </div>
      <div className={cx("mb-4 min-w-0 flex-1 pt-0.5", stage.status === "pending" && "opacity-60")}>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className={cx("text-sm font-medium", stage.status === "skipped" && "line-through decoration-base-content/30")}>{stage.label}</span>
          <BackendBadge backend={stage.backend} />
        </div>
        {stage.detail && (
          <div className={cx("mt-0.5 text-xs", stage.status === "failed" ? "text-error" : choosing ? "text-warning" : "text-base-content/60")}>{stage.detail}</div>
        )}
        {(stage.started_at || took) && (
          <div className="mt-0.5 font-mono text-[11px] text-base-content/40">
            {clockTime(stage.started_at)}
            {took && ` (${took})`}
          </div>
        )}
        {choosing && (
          <div className="mt-2 rounded-box border border-warning/30 bg-warning/5 p-2.5">
            <div className="mb-2 text-xs font-medium">Which face should the robot draw?</div>
            <FacePicker job={job} compact />
          </div>
        )}
        {art && !choosing && (
          <button
            type="button"
            className="checker mt-2 block h-20 w-20 overflow-hidden rounded-lg ring-1 ring-base-content/10 transition-transform hover:scale-105"
            onClick={() => openPreview(art, `${job.title}: ${stage.label}`)}
            aria-label={`Preview ${stage.label} output`}
          >
            <img src={art} alt="" className="h-full w-full object-contain" loading="lazy" />
          </button>
        )}
      </div>
    </li>
  );
}

function ArtifactTile({ url, label, title }: { url?: string; label: string; title: string }) {
  const u = fileUrl(url);
  return (
    <div className="min-w-0">
      <button
        type="button"
        disabled={!u}
        onClick={() => u && openPreview(u, `${title}: ${label}`)}
        className={cx(
          "group relative block aspect-square w-full overflow-hidden rounded-lg ring-1 ring-base-content/10",
          u ? "checker cursor-zoom-in" : "bg-base-200",
        )}
      >
        {u ? (
          <img src={u} alt={label} className="h-full w-full object-contain transition-transform group-hover:scale-105" loading="lazy" />
        ) : (
          <span className="grid h-full place-items-center text-base-content/30">
            <ImageIcon size={20} />
          </span>
        )}
      </button>
      <div className="mt-1 truncate text-center text-[11px] text-base-content/60">{label}</div>
    </div>
  );
}

function DownloadLink({ url, label }: { url?: string; label: string }) {
  const u = fileUrl(url);
  if (!u) return null;
  return (
    <a className="btn btn-xs btn-outline gap-1 border-base-content/20" href={u} target="_blank" rel="noreferrer" download>
      <DownloadIcon size={12} /> {label}
    </a>
  );
}

function JobDetail({ job }: { job: Job }) {
  const now = useNow(job.status === "running" ? 1000 : 30_000);
  const controller = useStudio((s) => s.controller);
  const mode = useStudio((s) => s.mode);
  const drawing = useStudio((s) => s.drawingJobId === job.id);
  const runForJob = useStudio((s) => {
    const r = s.activeRunId ? s.runs[s.activeRunId] : undefined;
    return r && r.job_id === job.id ? r : undefined;
  });
  const requestImg = fileUrl(job.request.image_url);
  const done = job.stages.filter((s) => s.status === "done" || s.status === "skipped").length;

  return (
    <div className="animate-fade-in space-y-5">
      <div className="flex gap-3">
        <div className="checker grid h-14 w-14 shrink-0 place-items-center overflow-hidden rounded-lg ring-1 ring-base-content/10">
          {jobThumb(job) ? <img src={jobThumb(job)} alt="" className="h-full w-full object-contain" /> : <KindIcon kind={job.kind} size={20} />}
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-base font-semibold leading-tight">{job.title}</h3>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <JobStatusBadge status={job.status} />
            <KindBadge kind={job.kind} />
            <span className="text-xs text-base-content/50">{relativeTime(job.created_at, now)}</span>
          </div>
        </div>
      </div>

      {(job.request.text || requestImg) && (
        <div className="flex items-center gap-2 rounded-box bg-base-200/60 p-2.5 text-sm">
          {requestImg && (
            <button type="button" onClick={() => openPreview(requestImg, "Uploaded photo")} className="shrink-0">
              <img src={requestImg} alt="Uploaded" className="h-10 w-10 rounded object-cover" />
            </button>
          )}
          {job.request.text ? <span className="min-w-0 italic text-base-content/75">&quot;{job.request.text}&quot;</span> : <span className="text-base-content/60">Uploaded photo</span>}
        </div>
      )}

      {job.status === "failed" && job.error && (
        <div role="alert" className="alert alert-error alert-soft items-start py-2.5 text-sm">
          <AlertIcon size={18} className="mt-0.5" />
          <span>{job.error}</span>
        </div>
      )}

      <div>
        <div className="mb-2 flex items-center justify-between">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-base-content/55">Stages</h4>
          <span className="font-mono text-[11px] text-base-content/45">
            {done}/{job.stages.length}
          </span>
        </div>
        {job.stages.length === 0 ? (
          <div className="text-sm text-base-content/50">No stages reported yet.</div>
        ) : (
          <ul>
            {job.stages.map((s, i) => (
              <StageItem key={`${s.name}-${i}`} stage={s} job={job} first={i === 0} last={i === job.stages.length - 1} now={now} />
            ))}
          </ul>
        )}
      </div>

      <div>
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-base-content/55">Artifacts</h4>
        <div className="grid grid-cols-3 gap-2.5">
          <ArtifactTile url={job.artifacts.source_image_url ?? job.request.image_url} label="Source photo" title={job.title} />
          <ArtifactTile url={job.artifacts.lineart_url} label="Line art" title={job.title} />
          <ArtifactTile url={job.artifacts.svg_url} label="SVG strokes" title={job.title} />
        </div>
      </div>

      {job.trajectory && (
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-base-content/55">Trajectory</h4>
          <div className="stats stats-horizontal w-full overflow-hidden border border-base-content/10 bg-base-200/50">
            <div className="stat px-3 py-2">
              <div className="stat-title text-[11px]">Strokes</div>
              <div className="stat-value font-mono text-xl">{job.trajectory.stroke_count}</div>
            </div>
            <div className="stat px-3 py-2">
              <div className="stat-title text-[11px]">Points</div>
              <div className="stat-value font-mono text-xl">{job.trajectory.point_count.toLocaleString()}</div>
            </div>
            <div className="stat px-3 py-2">
              <div className="stat-title text-[11px]">Est. draw</div>
              <div className="stat-value font-mono text-xl">{duration(job.trajectory.est_draw_seconds)}</div>
            </div>
          </div>
        </div>
      )}

      {(job.artifacts.lineart_url || job.artifacts.svg_url || job.artifacts.trajectory_url) && (
        <div className="flex flex-wrap gap-2">
          <DownloadLink url={job.artifacts.lineart_url} label="Line art" />
          <DownloadLink url={job.artifacts.svg_url} label="SVG" />
          <DownloadLink url={job.artifacts.trajectory_url} label="Trajectory JSON" />
        </div>
      )}

      <div className="sticky bottom-0 -mx-4 border-t border-base-content/10 bg-base-100/95 px-4 py-3 backdrop-blur">
        {runForJob ? (
          <button type="button" className="btn btn-outline btn-sm w-full gap-1.5" onClick={() => setFocus("live")}>
            <VideoIcon size={14} /> View the run ({Math.round(runForJob.progress * 100)}%)
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary btn-sm w-full gap-1.5"
            disabled={job.status !== "ready" || drawing}
            onClick={() => drawJob(job.id)}
          >
            {drawing ? <span className="loading loading-spinner loading-xs" /> : <PlayIcon size={13} />}
            {job.status === "ready" ? `Draw with ${controller.toUpperCase()} (${mode})` : job.status === "awaiting_input" ? "Pick a face first" : job.status === "failed" ? "Cannot draw a failed job" : "Waiting for the trajectory"}
          </button>
        )}
      </div>
    </div>
  );
}

function JobRow({ job, selected, now, onPick }: { job: Job; selected: boolean; now: number; onPick: () => void }) {
  const thumb = jobThumb(job);
  return (
    <button
      type="button"
      onClick={onPick}
      className={cx(
        "flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors",
        selected ? "bg-primary/10 ring-1 ring-primary/30" : "hover:bg-base-200",
      )}
    >
      <span className="checker grid h-9 w-9 shrink-0 place-items-center overflow-hidden rounded-md ring-1 ring-base-content/10">
        {thumb ? <img src={thumb} alt="" className="h-full w-full object-contain" loading="lazy" /> : <KindIcon kind={job.kind} size={14} />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{job.title}</span>
        <span className="block truncate text-[11px] text-base-content/50">{relativeTime(job.created_at, now) || job.kind}</span>
      </span>
      <JobStatusBadge status={job.status} size="xs" />
    </button>
  );
}

function JobSwitcher() {
  const jobs = useStudio((s) => s.jobList);
  const selectedId = useStudio((s) => s.selectedJobId);
  const selected = useStudio((s) => (s.selectedJobId ? s.jobs[s.selectedJobId] : undefined));
  const now = useNow(30_000);
  if (!jobs.length) return null;
  const pick = (id: string) => {
    selectJob(id);
    (document.activeElement as HTMLElement | null)?.blur();
  };
  return (
    <div className="dropdown dropdown-end">
      <button type="button" tabIndex={0} className="btn btn-ghost btn-xs max-w-48 gap-1 font-normal">
        <span className="truncate">{selected?.title ?? "Choose a job"}</span>
        <ChevronDownIcon size={13} />
      </button>
      <div tabIndex={0} className="dropdown-content z-40 mt-2 max-h-96 w-80 overflow-y-auto rounded-box border border-base-content/10 bg-base-100 p-2 shadow-xl scroll-thin">
        <div className="px-2 pb-1.5 pt-1 text-[11px] font-semibold uppercase tracking-wider text-base-content/50">Recent jobs</div>
        {jobs.slice(0, 40).map((j) => (
          <JobRow key={j.id} job={j} selected={j.id === selectedId} now={now} onPick={() => pick(j.id)} />
        ))}
      </div>
    </div>
  );
}

function RecentJobs() {
  const jobs = useStudio((s) => s.jobList);
  const selectedId = useStudio((s) => s.selectedJobId);
  const now = useNow(30_000);
  if (jobs.length < 2) return null;
  return (
    <div className="mt-6">
      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-base-content/55">Recent jobs</h4>
      <div className="space-y-0.5">
        {jobs.slice(0, 8).map((j) => (
          <JobRow key={j.id} job={j} selected={j.id === selectedId} now={now} onPick={() => selectJob(j.id)} />
        ))}
      </div>
    </div>
  );
}

function PipelineSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex gap-3">
        <div className="skeleton h-14 w-14" />
        <div className="flex-1 space-y-2 pt-1">
          <div className="skeleton h-4 w-2/3" />
          <div className="skeleton h-3 w-1/3" />
        </div>
      </div>
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="flex gap-3">
          <div className="skeleton h-6 w-6 shrink-0 rounded-full" />
          <div className="flex-1 space-y-1.5">
            <div className="skeleton h-3 w-1/2" />
            <div className="skeleton h-3 w-3/4" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function PipelinePanel({ className }: { className?: string }) {
  const job = useStudio((s) => (s.selectedJobId ? s.jobs[s.selectedJobId] : undefined));
  const loaded = useStudio((s) => s.jobsLoaded);
  const online = useStudio((s) => s.online);
  const hasJobs = useStudio((s) => s.jobList.length > 0);

  return (
    <section className={cx("panel min-w-0 flex-1", className)} aria-label="Pipeline">
      <div className="panel-header">
        <LayersIcon size={16} className="text-primary" />
        <h2 className="panel-title">Pipeline</h2>
        <div className="flex-1" />
        <JobSwitcher />
        <button type="button" className="btn btn-ghost btn-xs hidden gap-1 lg:inline-flex xl:hidden" onClick={() => setFocus("live")}>
          <VideoIcon size={13} /> Live
        </button>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 pt-4">
        {job ? (
          <JobDetail key={job.id} job={job} />
        ) : !loaded && online !== false ? (
          <PipelineSkeleton />
        ) : (
          <div className="animate-fade-in flex flex-col items-center py-10 text-center">
            <div className="mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-primary/15 text-primary">
              <LayersIcon size={22} />
            </div>
            <div className="font-medium">{hasJobs ? "Select a job" : online === false ? "Jobs unavailable while offline" : "No jobs yet"}</div>
            <p className="mt-1 max-w-xs text-sm text-base-content/60">
              {hasJobs ? "Choose one of the recent jobs below." : "Each request becomes a job: its pipeline stages, artifacts and trajectory show up here."}
            </p>
          </div>
        )}
        <div className="pb-4">
          <RecentJobs />
        </div>
      </div>
    </section>
  );
}
