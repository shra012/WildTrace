"use client";

import { useEffect, useRef } from "react";
import { cx } from "@/lib/cx";
import { onTelemetry } from "@/lib/studio";
import type { Point, Trace } from "@/lib/types";

interface Props {
  runId: string | null;
  trace: Trace | null;
  live: boolean;
  /** >1 follows the pen tip at that magnification (wrist-camera style). */
  zoom?: number;
  /** Draw a small, low-detail version (picture in picture). */
  mini?: boolean;
  className?: string;
}

const PAPER = "#fbfaf7";
const DESIRED = "rgba(32, 41, 64, 0.17)";
const INK = "#16233f";
const TIP = "#0ea5a0";

interface TailPoint {
  p: Point;
  phase: string;
}

/**
 * Live 2D trace: desired path in light gray, executed path in ink, pen tip as a pulsing dot.
 * The executed path comes from /trace polling; telemetry tips extend it between polls.
 */
export default function TraceCanvas({ runId, trace, live, zoom = 1, mini = false, className }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef({ trace, live, zoom, mini });
  const tailRef = useRef<TailPoint[]>([]);
  const tipRef = useRef<Point | null>(null);
  const dirtyRef = useRef(true);

  useEffect(() => {
    propsRef.current = { trace, live, zoom, mini };
    // The fresh trace already contains everything the telemetry tail covered.
    tailRef.current = tailRef.current.slice(-1);
    const ex = trace?.executed;
    const lastStroke = ex && ex.length ? ex[ex.length - 1] : null;
    if (lastStroke && lastStroke.length && !tipRef.current) tipRef.current = lastStroke[lastStroke.length - 1];
    dirtyRef.current = true;
  }, [trace, live, zoom, mini]);

  useEffect(() => {
    tailRef.current = [];
    tipRef.current = null;
    dirtyRef.current = true;
    if (!runId) return;
    return onTelemetry((e) => {
      if (e.run_id !== runId || !e.tip) return;
      tipRef.current = e.tip;
      const tail = tailRef.current;
      tail.push({ p: e.tip, phase: e.phase });
      if (tail.length > 600) tail.splice(0, tail.length - 600);
      dirtyRef.current = true;
    });
  }, [runId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    let W = 0;
    let H = 0;
    let dpr = 1;

    const resize = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      const r = wrap.getBoundingClientRect();
      W = Math.max(1, Math.round(r.width * dpr));
      H = Math.max(1, Math.round(r.height * dpr));
      canvas.width = W;
      canvas.height = H;
      dirtyRef.current = true;
    };

    const draw = (t: number) => {
      const { trace: tr, live: isLive, zoom: z, mini: small } = propsRef.current;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const aspect = tr?.aspect && tr.aspect > 0 ? tr.aspect : 1;
      const m = (small ? 6 : 18) * dpr;
      const pw = Math.max(1, Math.min(W - 2 * m, (H - 2 * m) * aspect));
      const ph = pw / aspect;
      const ox = (W - pw) / 2;
      const oy = (H - ph) / 2;
      const X = (p: Point) => ox + p[0] * pw;
      const Y = (p: Point) => oy + p[1] * ph;
      const tip = tipRef.current;

      ctx.save();
      if (z > 1 && tip) {
        ctx.translate(W / 2, H / 2);
        ctx.scale(z, z);
        ctx.translate(-X(tip), -Y(tip));
      }

      // Paper with a faint grid.
      ctx.save();
      ctx.shadowColor = "rgba(0,0,0,0.28)";
      ctx.shadowBlur = (small ? 6 : 22) * dpr;
      ctx.shadowOffsetY = (small ? 2 : 6) * dpr;
      ctx.fillStyle = PAPER;
      ctx.fillRect(ox, oy, pw, ph);
      ctx.restore();
      if (!small) {
        ctx.strokeStyle = "rgba(0,0,0,0.045)";
        ctx.lineWidth = 1;
        const step = Math.max(pw, ph) / 16;
        ctx.beginPath();
        for (let x = ox + step; x < ox + pw - 1; x += step) {
          ctx.moveTo(x, oy);
          ctx.lineTo(x, oy + ph);
        }
        for (let y = oy + step; y < oy + ph - 1; y += step) {
          ctx.moveTo(ox, y);
          ctx.lineTo(ox + pw, y);
        }
        ctx.stroke();
      }

      const lw = Math.max(1.2 * dpr, Math.min(pw, ph) / (small ? 160 : 260)) / (z > 1 ? Math.sqrt(z) : 1);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      const strokeLines = (lines: Point[][] | undefined, color: string, width: number) => {
        if (!lines?.length) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        for (const line of lines) {
          if (!line.length) continue;
          ctx.moveTo(X(line[0]), Y(line[0]));
          if (line.length === 1) ctx.lineTo(X(line[0]) + 0.01, Y(line[0]));
          for (let i = 1; i < line.length; i++) ctx.lineTo(X(line[i]), Y(line[i]));
        }
        ctx.stroke();
      };

      strokeLines(tr?.desired, DESIRED, lw * 1.25);
      strokeLines(tr?.executed, INK, lw * 1.3);

      // Telemetry tail: only pen-down samples of the same stroke are joined.
      const tail = tailRef.current;
      if (isLive && tail.length > 1) {
        ctx.strokeStyle = INK;
        ctx.lineWidth = lw * 1.3;
        ctx.beginPath();
        for (let i = 1; i < tail.length; i++) {
          const a = tail[i - 1];
          const b = tail[i];
          if (a.phase !== b.phase || !/DRAW/i.test(b.phase)) continue;
          ctx.moveTo(X(a.p), Y(a.p));
          ctx.lineTo(X(b.p), Y(b.p));
        }
        ctx.stroke();
      }

      if (tip && (isLive || (tr?.executed?.length ?? 0) > 0)) {
        const r = Math.max(3 * dpr, lw * 2.2);
        if (isLive) {
          const pulse = (t / 900) % 1;
          ctx.beginPath();
          ctx.arc(X(tip), Y(tip), r + pulse * r * 2.4, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(14,165,160,${0.55 * (1 - pulse)})`;
          ctx.lineWidth = Math.max(1, lw * 0.8);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(X(tip), Y(tip), r, 0, Math.PI * 2);
        ctx.fillStyle = TIP;
        ctx.fill();
        ctx.lineWidth = Math.max(1, lw * 0.6);
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }
      ctx.restore();
    };

    const frame = (t: number) => {
      if (dirtyRef.current || propsRef.current.live) {
        dirtyRef.current = false;
        draw(t);
      }
      raf = requestAnimationFrame(frame);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <div ref={wrapRef} className={cx("relative h-full w-full", className)}>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-label="Pen trace" role="img" />
    </div>
  );
}
