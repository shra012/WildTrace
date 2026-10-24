"use client";

import { cx } from "@/lib/cx";
import { setFocus, setGalleryOpen, useStudio, type Focus } from "@/lib/studio";
import { ChatIcon, GridIcon, LayersIcon, VideoIcon } from "./icons";

export default function MobileDock() {
  const focus = useStudio((s) => s.focus);
  const runLive = useStudio((s) => {
    const r = s.activeRunId ? s.runs[s.activeRunId] : undefined;
    return !!r && (r.status === "running" || r.status === "starting");
  });
  const jobBusy = useStudio((s) => {
    const j = s.selectedJobId ? s.jobs[s.selectedJobId] : undefined;
    return !!j && (j.status === "running" || j.status === "queued" || j.status === "awaiting_input");
  });

  const item = (key: Focus, label: string, Icon: typeof ChatIcon, dot?: boolean) => (
    <button type="button" className={cx(focus === key && "dock-active text-primary")} onClick={() => setFocus(key)} aria-label={label}>
      <span className="indicator">
        {dot && <span className="indicator-item status status-primary animate-pulse" />}
        <Icon size={20} />
      </span>
      <span className="dock-label">{label}</span>
    </button>
  );

  return (
    <nav className="dock dock-sm z-30 border-t border-base-content/10 bg-base-100/90 backdrop-blur lg:hidden" aria-label="Sections">
      {item("chat", "Chat", ChatIcon)}
      {item("live", "Live", VideoIcon, runLive)}
      {item("pipeline", "Pipeline", LayersIcon, jobBusy)}
      <button type="button" onClick={() => setGalleryOpen(true)} aria-label="Gallery">
        <GridIcon size={20} />
        <span className="dock-label">Gallery</span>
      </button>
    </nav>
  );
}
