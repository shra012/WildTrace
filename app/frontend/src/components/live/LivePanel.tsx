"use client";

import { useState, type ReactNode } from "react";
import { fileUrl } from "@/lib/config";
import { cx } from "@/lib/cx";
import { duration, formatNumber, phaseLabel } from "@/lib/format";
import FacePicker from "../cards/FacePicker";
import { drawJob, setController, setFocus, setGalleryOpen, setMode, useStudio, useTelemetry } from "@/lib/studio";
import type { Job, Run, Trace } from "@/lib/types";
import { useRunTrace } from "@/hooks/useRunTrace";
import { RunStatusBadge } from "../badges";
import { jobThumb } from "../cards/JobCard";
import { AlertIcon, GaugeIcon, GridIcon, LayersIcon, PenIcon, PlayIcon, RouteIcon, TargetIcon, VideoIcon, ClockIcon } from "../icons";
import CameraFeed from "./CameraFeed";
import MockCamera from "./MockCamera";
import TraceCanvas from "./TraceCanvas";

function isLive(run?: Run | null): boolean {
  return !!run && (run.status === "starting" || run.status === "running");
}

function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string; title?: string }[]; onChange: (v: T) => void; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-base-content/60">{label}</span>
      <div className="join" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={value === o.value}
            title={o.title}
            className={cx("btn join-item btn-xs sm:btn-sm", value === o.value ? "btn-primary" : "btn-ghost bg-base-200")}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function RunControls({ job }: { job?: Job }) {
  const controller = useStudio((s) => s.controller);
  const mode = useStudio((s) => s.mode);
  const drawing = useStudio((s) => !!s.drawingJobId);
  const isaac = useStudio((s) => s.health?.isaac);
  const online = useStudio((s) => s.online);
  const ready = job?.status === "ready";
  const isaacDown = isaac ? !isaac.running : false;

  return (
    <div className="space-y-2 border-b border-base-content/10 px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Segmented
          label="Controller"
          value={controller}
          onChange={setController}
          options={[
            { value: "act", label: "ACT", title: "Learned ACT policy" },
            { value: "ik", label: "IK", title: "Inverse kinematics baseline" },
          ]}
        />
        <Segmented
          label="Mode"
          value={mode}
          onChange={setMode}
          options={[
            { value: "live", label: "Live", title: "Draw live in Isaac Sim" },
            { value: "replay", label: "Replay", title: "Replay a recorded run or animate the plan" },
          ]}
        />
        <button
          type="button"
          className="btn btn-primary btn-sm ml-auto min-w-0 gap-1.5"
          disabled={!ready || drawing || online === false}
          onClick={() => job && drawJob(job.id)}
          title={!job ? "Select a job first" : ready ? `Draw ${job.title}` : "Available when the trajectory is ready"}
        >
          {drawing ? <span className="loading loading-spinner loading-xs" /> : <PlayIcon size={14} />}
          <span className="max-w-40 truncate">{job ? `Draw ${job.title}` : "Draw"}</span>
        </button>
      </div>
      {mode === "live" && isaacDown && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-warning">
          <AlertIcon size={14} />
          Isaac Sim is not running, so a live run cannot start.
          <button type="button" className="btn btn-xs btn-ghost text-warning underline" onClick={() => setMode("replay")}>
            Use replay
          </button>
        </div>
      )}
    </div>
  );
}

function Stat({ icon, label, value, unit, tone }: { icon: ReactNode; label: string; value: string; unit?: string; tone?: string }) {
  return (
    <div className="min-w-0 rounded-box border border-base-content/10 bg-base-200/60 px-3 py-2">
      <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-base-content/55">
        {icon}
        <span className="truncate">{label}</span>
      </div>
      <div className={cx("mt-0.5 truncate font-mono text-lg font-semibold tabular-nums", tone)}>
        {value}
        {unit && value !== "-" && <span className="ml-1 text-xs font-normal text-base-content/50">{unit}</span>}
      </div>
    </div>
  );
}

function RunStatus({ run, job }: { run: Run; job?: Job }) {
  const tele = useTelemetry(run.id);
  const live = isLive(run);
  const progress = live && tele ? Math.max(run.progress, tele.progress) : run.progress;
  const phase = (live && tele?.phase) || run.phase;
  const m = run.metrics ?? {};

  return (
    <div className="space-y-3 border-b border-base-content/10 px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">{job ? job.title : "Robot run"}</div>
          <div className="text-xs text-base-content/55">
            {run.controller.toUpperCase()}, {run.mode}
            {run.mode === "replay" && run.source ? ` (${run.source === "recorded" ? "recorded twin run" : "planned path"})` : ""}
          </div>
        </div>
        {phase && (
          <span className="badge badge-sm badge-ghost font-mono" title={phase}>
            {phaseLabel(phase)}
          </span>
        )}
        <RunStatusBadge status={run.status} />
      </div>
      <div className="flex items-center gap-3">
        <progress className={cx("progress h-2 flex-1", run.status === "failed" ? "progress-error" : "progress-primary")} value={Math.round(progress * 1000)} max={1000} />
        <span className="w-12 text-right font-mono text-sm tabular-nums">{Math.round(progress * 100)}%</span>
      </div>
      {run.error && (
        <div role="alert" className="alert alert-error alert-soft py-2 text-sm">
          <AlertIcon size={16} />
          {run.error}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat icon={<TargetIcon size={12} />} label="RMSE" value={formatNumber(m.rmse_mm)} unit="mm" />
        <Stat icon={<RouteIcon size={12} />} label="Max error" value={formatNumber(m.max_error_mm)} unit="mm" />
        <Stat icon={<ClockIcon size={12} />} label="Sim time" value={m.sim_seconds !== undefined ? duration(m.sim_seconds) : "-"} />
        {live ? (
          <Stat
            icon={<GaugeIcon size={12} />}
            label="Live error"
            value={formatNumber(tele?.error_mm ?? undefined)}
            unit="mm"
            tone={tele?.error_mm != null && tele.error_mm > 3 ? "text-warning" : undefined}
          />
        ) : (
          <Stat icon={<GaugeIcon size={12} />} label="Jerk RMS" value={formatNumber(m.jerk_rms, 1)} />
        )}
      </div>
    </div>
  );
}

function Legend() {
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-3 rounded-md bg-base-100/80 px-2.5 py-1.5 text-[11px] shadow backdrop-blur">
      <span className="flex items-center gap-1.5">
        <span className="h-0.5 w-4 rounded bg-[rgba(32,41,64,0.3)]" /> Desired
      </span>
      <span className="flex items-center gap-1.5">
        <span className="h-0.5 w-4 rounded bg-[#16233f] ring-1 ring-white/40" /> Executed
      </span>
      <span className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full bg-[#0ea5a0]" /> Pen tip
      </span>
    </div>
  );
}

function EmptyLive({ job }: { job?: Job }) {
  const drawing = useStudio((s) => !!s.drawingJobId);
  const thumb = job ? jobThumb(job) : undefined;
  if (job && job.status === "awaiting_input" && (job.choices?.length ?? 0) > 0) {
    return (
      <div className="animate-fade-in mx-auto flex h-full w-full max-w-3xl flex-col gap-4 overflow-y-auto p-6 [&>*:first-child]:mt-auto [&>*:last-child]:mb-auto">
        <div className="text-center">
          <div className="text-lg font-semibold">Which face should the robot draw?</div>
          <div className="mt-1 text-sm text-base-content/60">
            {job.choices!.length} faces found in your photo. Click one, or type &quot;the second one&quot; in the chat.
          </div>
        </div>
        <FacePicker job={job} />
      </div>
    );
  }
  if (job) {
    const ready = job.status === "ready";
    return (
      <div className="animate-fade-in flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
        <div className="checker relative aspect-square w-full max-w-72 overflow-hidden rounded-box shadow-lg ring-1 ring-base-content/10">
          {thumb ? <img src={thumb} alt={job.title} className="h-full w-full object-contain p-3" /> : <div className="skeleton h-full w-full" />}
        </div>
        <div>
          <div className="font-medium">{job.title}</div>
          <div className="mt-1 text-sm text-base-content/60">
            {ready
              ? "Ready to draw. Pick a controller and mode above, then press Draw."
              : job.status === "awaiting_input"
                ? "Waiting for you to pick a face."
                : job.status === "failed"
                  ? "This job failed, so there is nothing to draw."
                  : "Preparing the trajectory..."}
          </div>
        </div>
        {ready && (
          <button type="button" className="btn btn-primary gap-2" disabled={drawing} onClick={() => drawJob(job.id)}>
            {drawing ? <span className="loading loading-spinner loading-sm" /> : <PenIcon size={16} />}
            Draw it
          </button>
        )}
      </div>
    );
  }
  return (
    <div className="animate-fade-in flex h-full flex-col items-center justify-center gap-3 p-6 text-center">
      <div className="grid h-14 w-14 place-items-center rounded-2xl bg-primary/15 text-primary">
        <VideoIcon size={26} />
      </div>
      <div className="font-medium">Nothing is being drawn</div>
      <p className="max-w-xs text-sm text-base-content/60">Ask the assistant for an animal or pick one from the gallery, then press Draw to watch the xArm 7 here.</p>
      <button type="button" className="btn btn-sm btn-outline gap-1.5" onClick={() => setGalleryOpen(true)}>
        <GridIcon size={14} /> Open gallery
      </button>
    </div>
  );
}

function Viewer({ run, trace, loading, error }: { run: Run; trace: Trace | null; loading: boolean; error: string | null }) {
  const mock = useStudio((s) => s.mock);
  const [tab, setTab] = useState<string>("trace");
  const live = isLive(run);
  const view = tab !== "trace" && run.cameras.includes(tab) ? tab : "trace";

  return (
    <div className="flex min-h-[320px] flex-1 flex-col gap-2 p-3">
      <div className="flex items-center gap-2">
        <div role="tablist" className="tabs tabs-box tabs-sm bg-base-200/70">
          <button type="button" role="tab" className={cx("tab gap-1.5", view === "trace" && "tab-active")} onClick={() => setTab("trace")}>
            <PenIcon size={13} /> Trace
          </button>
          {run.cameras.map((c) => (
            <button key={c} type="button" role="tab" className={cx("tab gap-1.5", view === c && "tab-active")} onClick={() => setTab(c)}>
              <VideoIcon size={13} /> {c}
            </button>
          ))}
        </div>
        {run.cameras.length === 0 && (
          <span className="truncate text-xs text-base-content/50">{run.mode === "replay" ? "Planned replay: no camera feed" : "No cameras for this run"}</span>
        )}
      </div>
      <div className="relative min-h-0 flex-1 overflow-hidden rounded-box bg-base-300/70 ring-1 ring-base-content/10">
        {view === "trace" ? (
          <>
            <TraceCanvas runId={run.id} trace={trace} live={live} />
            {!trace && loading && (
              <div className="absolute inset-0 grid place-items-center">
                <span className="loading loading-spinner loading-md opacity-60" />
              </div>
            )}
            {!trace && error && (
              <div className="absolute inset-x-3 top-3">
                <div role="alert" className="alert alert-warning alert-soft py-2 text-sm">
                  <AlertIcon size={16} /> Trace unavailable: {error}
                </div>
              </div>
            )}
            <Legend />
          </>
        ) : (
          <>
            {mock ? <MockCamera runId={run.id} camera={view} trace={trace} live={live} /> : <CameraFeed runId={run.id} camera={view} live={live} />}
            <div className="absolute bottom-3 right-3 h-28 w-28 overflow-hidden rounded-lg bg-base-300/80 shadow-xl ring-1 ring-white/20 sm:h-36 sm:w-36">
              <TraceCanvas runId={run.id} trace={trace} live={live} mini />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function LivePanel({ className }: { className?: string }) {
  const run = useStudio((s) => (s.activeRunId ? s.runs[s.activeRunId] : undefined));
  const selectedJob = useStudio((s) => (s.selectedJobId ? s.jobs[s.selectedJobId] : undefined));
  const runJob = useStudio((s) => (run ? s.jobs[run.job_id] : undefined));
  const live = isLive(run);
  const { trace, loading, error } = useRunTrace(run?.id, live);
  const showRun = !!run;
  const controlsJob = selectedJob ?? runJob;
  const lineart = runJob ? fileUrl(runJob.artifacts.svg_url) : undefined;

  return (
    <section className={cx("panel min-w-0 flex-1", className)} aria-label="Live view">
      <div className="panel-header">
        <VideoIcon size={16} className="text-primary" />
        <h2 className="panel-title">Live</h2>
        {live &&
          (run?.mode === "replay" ? (
            <span className="badge badge-xs badge-info gap-1 font-semibold text-info-content">REPLAY</span>
          ) : (
            <span className="badge badge-xs badge-error gap-1 font-semibold text-error-content">LIVE</span>
          ))}
        {run && <span className="hidden truncate font-mono text-[11px] text-base-content/45 sm:inline">{run.id}</span>}
        <div className="flex-1" />
        {lineart && (
          <img src={lineart} alt="" className="checker hidden h-7 w-7 rounded object-contain ring-1 ring-base-content/10 sm:block" />
        )}
        <button type="button" className="btn btn-ghost btn-xs hidden gap-1 lg:inline-flex xl:hidden" onClick={() => setFocus("pipeline")}>
          <LayersIcon size={13} /> Pipeline
        </button>
      </div>
      <div className="scroll-thin flex min-h-0 flex-1 flex-col overflow-y-auto">
        <RunControls job={controlsJob} />
        {showRun && run ? (
          <>
            <RunStatus run={run} job={runJob} />
            <Viewer key={run.id} run={run} trace={trace} loading={loading} error={error} />
          </>
        ) : (
          <div className="min-h-[320px] flex-1">
            <EmptyLive job={selectedJob} />
          </div>
        )}
      </div>
    </section>
  );
}
