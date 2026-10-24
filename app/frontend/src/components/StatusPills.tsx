"use client";

import type { ReactNode } from "react";
import { cx } from "@/lib/cx";
import { shortBackend } from "@/lib/format";
import { useStudio } from "@/lib/studio";
import { ChatIcon, ChipIcon, RobotIcon, SparklesIcon, WifiOffIcon } from "./icons";

function Pill({ tip, className, children }: { tip: string; className?: string; children: ReactNode }) {
  return (
    <div className="tooltip tooltip-bottom before:max-w-64 before:whitespace-normal before:text-left" data-tip={tip}>
      <span className={cx("badge badge-sm h-6 gap-1.5 whitespace-nowrap border-base-content/10 font-medium", className)}>{children}</span>
    </div>
  );
}

export default function StatusPills({ vertical = false }: { vertical?: boolean }) {
  const health = useStudio((s) => s.health);
  const online = useStudio((s) => s.online);
  const ws = useStudio((s) => s.wsStatus);
  const mock = useStudio((s) => s.mock);

  if (online === null && !health) {
    return (
      <div className="flex items-center gap-2 text-xs opacity-70">
        <span className="loading loading-dots loading-xs" /> Connecting
      </div>
    );
  }

  const wrap = cx("flex gap-1.5", vertical ? "flex-col items-start" : "flex-wrap items-center justify-end");

  if (online === false) {
    return (
      <div className={wrap}>
        <Pill tip="The backend is not answering /api/health" className="badge-error badge-soft">
          <WifiOffIcon size={13} /> Backend offline
        </Pill>
      </div>
    );
  }

  const gpu = health?.gpu;
  const isaac = health?.isaac;
  const image = health?.backends.image ?? "";
  const flux = /flux/i.test(image);

  return (
    <div className={wrap}>
      <Pill
        tip={mock ? "Mock mode: events come from the in-browser demo backend" : ws === "open" ? "Live event stream connected" : "Event stream reconnecting; polling meanwhile"}
        className="badge-ghost"
      >
        <span className={cx("status", ws === "open" ? "status-success" : "status-warning animate-pulse")} />
        {mock ? "Mock" : ws === "open" ? "Live" : "Polling"}
      </Pill>
      {gpu && (
        <Pill tip={gpu.note || (gpu.busy ? "GPU busy" : "GPU free")} className={gpu.busy ? "badge-warning badge-soft" : "badge-success badge-soft"}>
          <ChipIcon size={13} />
          GPU {gpu.busy ? "busy" : "free"} <span className="font-mono opacity-80">{Math.round(gpu.utilization)}%</span>
        </Pill>
      )}
      {isaac && (
        <Pill
          tip={`Isaac Sim ${isaac.running ? "running" : "stopped"}${isaac.scene_ready ? ", scene ready" : ""}${isaac.config ? ` (${isaac.config})` : ""}${health?.policy ? `. Policy ${health.policy.checkpoint} on ${health.policy.device}` : ""}`}
          className={isaac.running ? (isaac.scene_ready ? "badge-success badge-soft" : "badge-info badge-soft") : "badge-ghost"}
        >
          <RobotIcon size={13} />
          Isaac {isaac.running ? (isaac.scene_ready ? "ready" : "running") : "stopped"}
        </Pill>
      )}
      {image && (
        <Pill tip={`Image backend: ${image}`} className={flux ? "badge-accent badge-soft" : "badge-secondary badge-soft"}>
          <SparklesIcon size={13} />
          {shortBackend(image)}
        </Pill>
      )}
      {health?.backends.chat && (
        <Pill tip={`Chat backend: ${health.backends.chat}${health.backends.face ? `. Face: ${health.backends.face}` : ""}`} className="badge-ghost">
          <ChatIcon size={13} />
          {shortBackend(health.backends.chat)}
        </Pill>
      )}
    </div>
  );
}
