"use client";

import type { Trace } from "@/lib/types";
import { CameraBadge } from "./CameraFeed";
import TraceCanvas from "./TraceCanvas";

/** Stand-in for an MJPEG camera in mock mode: the trace rendered as if filmed in the scene. */
export default function MockCamera({ runId, camera, trace, live }: { runId: string; camera: string; trace: Trace | null; live: boolean }) {
  const wide = /wide|front/i.test(camera);
  return (
    <div className="relative h-full w-full overflow-hidden bg-[#1d1f24]">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_35%,#4b4f57,transparent_70%)]" />
      {wide ? (
        <div className="absolute inset-0 [perspective:900px]">
          <div className="absolute inset-x-[-10%] bottom-[-20%] top-[35%] origin-bottom bg-[linear-gradient(#6b5a48,#4a3d31)] [transform:rotateX(58deg)]" />
          <div className="absolute inset-x-[12%] bottom-[2%] top-[18%] origin-bottom [transform:rotateX(52deg)]">
            <TraceCanvas runId={runId} trace={trace} live={live} />
          </div>
          <div className="absolute right-[14%] top-[6%] h-[38%] w-[7%] rounded-t-full bg-gradient-to-b from-zinc-200 to-zinc-500 opacity-80 shadow-2xl" />
        </div>
      ) : (
        <div className="absolute inset-0 p-2">
          <TraceCanvas runId={runId} trace={trace} live={live} />
        </div>
      )}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_55%,rgba(0,0,0,0.55))]" />
      <CameraBadge camera={camera} live={live} mock />
    </div>
  );
}
