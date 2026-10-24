"use client";

import { useEffect } from "react";
import { WS_URL } from "@/lib/config";
import { cx } from "@/lib/cx";
import { subscribeMock } from "@/lib/mock";
import { handleEvent, setGalleryOpen, setWsStatus, startStudio, useStudio } from "@/lib/studio";
import type { StudioEvent } from "@/lib/types";
import { useWebSocket } from "@/hooks/useWebSocket";
import ChatPanel from "./chat/ChatPanel";
import GalleryPanel from "./gallery/GalleryPanel";
import LivePanel from "./live/LivePanel";
import MobileDock from "./MobileDock";
import Navbar from "./Navbar";
import OfflineBanner from "./OfflineBanner";
import PipelinePanel from "./pipeline/PipelinePanel";
import PreviewModal from "./PreviewModal";
import Toasts from "./Toasts";

function StudioEvents() {
  const started = useStudio((s) => s.started);
  const mock = useStudio((s) => s.mock);

  useWebSocket<StudioEvent>({ url: WS_URL, enabled: started && !mock, onMessage: handleEvent, onStatus: setWsStatus });

  useEffect(() => {
    if (!started || !mock) return;
    setWsStatus("open");
    return subscribeMock(handleEvent);
  }, [started, mock]);

  return null;
}

export default function Studio() {
  const focus = useStudio((s) => s.focus);
  const galleryOpen = useStudio((s) => s.galleryOpen);

  useEffect(() => startStudio(), []);

  return (
    <div className="drawer drawer-end h-dvh">
      <input
        id="gallery-drawer"
        type="checkbox"
        className="drawer-toggle"
        checked={galleryOpen}
        onChange={(e) => setGalleryOpen(e.target.checked)}
        aria-label="Toggle gallery"
      />
      <div className="drawer-content flex h-dvh min-h-0 flex-col">
        <StudioEvents />
        <Navbar />
        <OfflineBanner />
        <main className="flex min-h-0 flex-1 gap-3 p-2 pb-[4.75rem] sm:p-3 sm:pb-[4.75rem] lg:grid lg:grid-cols-[minmax(320px,370px)_minmax(0,1fr)] lg:pb-3 xl:grid-cols-[minmax(320px,370px)_minmax(0,1fr)_minmax(340px,400px)]">
          <ChatPanel className={cx(focus === "chat" ? "flex" : "hidden", "lg:flex")} />
          <LivePanel className={cx(focus === "live" ? "flex" : "hidden", focus === "pipeline" ? "lg:hidden" : "lg:flex", "xl:flex")} />
          <PipelinePanel className={cx(focus === "pipeline" ? "flex" : "hidden", "xl:flex")} />
        </main>
        <MobileDock />
      </div>
      <div className="drawer-side z-50">
        <label htmlFor="gallery-drawer" aria-label="Close gallery" className="drawer-overlay" />
        <GalleryPanel />
      </div>
      <Toasts />
      <PreviewModal />
    </div>
  );
}
