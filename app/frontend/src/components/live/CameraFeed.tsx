"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { RefreshIcon, VideoIcon } from "../icons";

/** MJPEG stream from /api/runs/{id}/stream/{camera} with a retry when the stream drops. */
export default function CameraFeed({ runId, camera, live }: { runId: string; camera: string; live: boolean }) {
  const [state, setState] = useState<{ key: string; nonce: number; status: "loading" | "ok" | "error" }>({ key: "", nonce: 0, status: "loading" });
  const key = `${runId}/${camera}`;
  const current = state.key === key ? state : { key, nonce: 0, status: "loading" as const };

  return (
    <div className="relative h-full w-full bg-black">
      {current.status !== "error" && (
        <img
          key={`${key}#${current.nonce}`}
          src={api.streamUrl(runId, camera, current.nonce)}
          alt={`${camera} camera`}
          className="h-full w-full object-contain"
          onLoad={() => setState({ key, nonce: current.nonce, status: "ok" })}
          onError={() => setState({ key, nonce: current.nonce, status: "error" })}
        />
      )}
      {current.status === "loading" && (
        <div className="absolute inset-0 grid place-items-center text-white/70">
          <div className="flex flex-col items-center gap-2 text-sm">
            <span className="loading loading-spinner loading-md" />
            Connecting to {camera}...
          </div>
        </div>
      )}
      {current.status === "error" && (
        <div className="absolute inset-0 grid place-items-center p-4 text-center text-white/80">
          <div className="flex flex-col items-center gap-2">
            <VideoIcon size={28} className="opacity-60" />
            <div className="text-sm font-medium">{live ? "Camera stream unavailable" : "The stream has ended"}</div>
            <div className="max-w-xs text-xs opacity-70">
              {live ? "Isaac Sim may still be warming up the camera." : "Camera frames are only streamed while a run is drawing."}
            </div>
            <button type="button" className="btn btn-sm btn-outline mt-1 gap-1 border-white/30 text-white" onClick={() => setState({ key, nonce: current.nonce + 1, status: "loading" })}>
              <RefreshIcon size={14} /> Retry
            </button>
          </div>
        </div>
      )}
      <CameraBadge camera={camera} live={live && current.status === "ok"} />
    </div>
  );
}

export function CameraBadge({ camera, live, mock = false }: { camera: string; live: boolean; mock?: boolean }) {
  return (
    <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-1.5 rounded-md bg-black/55 px-2 py-1 font-mono text-[11px] text-white backdrop-blur">
      {live && <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-red-500" />}
      {camera}
      {mock && <span className="opacity-60">(mock feed)</span>}
    </div>
  );
}
