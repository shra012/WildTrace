"use client";

import { useState } from "react";
import { fileUrl } from "@/lib/config";
import { cx } from "@/lib/cx";
import { chooseFace, useStudio } from "@/lib/studio";
import type { Job } from "@/lib/types";
import { UserIcon } from "../icons";

/**
 * Face picker for a face job waiting in awaiting_input: crop thumbnails labelled Face 1..N and,
 * optionally, the uploaded photo with numbered box overlays. Clicking a face calls /choose.
 */
export default function FacePicker({ job, compact = false }: { job: Job; compact?: boolean }) {
  const choosing = useStudio((s) => (s.choosing?.jobId === job.id ? s.choosing.index : null));
  const [hover, setHover] = useState<number | null>(null);
  const [showPhoto, setShowPhoto] = useState(!compact);
  const choices = job.choices ?? [];
  const photo = fileUrl(job.request.image_url ?? job.artifacts.source_image_url);
  const waiting = job.status === "awaiting_input";
  if (choices.length === 0) return null;

  return (
    <div className="animate-rise space-y-2.5">
      {photo && showPhoto && (
        <div className={cx("relative overflow-hidden rounded-lg ring-1 ring-base-content/10", !compact && "mx-auto w-fit")}>
          <img
            src={photo}
            alt="Uploaded group photo"
            className={cx("block h-auto", compact ? "w-full" : "max-h-[42vh] w-auto max-w-full")}
          />
          {choices.map((c) => {
            const [x0, y0, x1, y1] = c.box;
            const active = hover === c.index || choosing === c.index;
            return (
              <button
                key={c.index}
                type="button"
                disabled={!waiting || choosing !== null}
                onClick={() => chooseFace(job.id, c.index)}
                onMouseEnter={() => setHover(c.index)}
                onMouseLeave={() => setHover(null)}
                className={cx(
                  "absolute rounded-md border-2 transition-all",
                  active ? "border-primary bg-primary/15 shadow-[0_0_0_3px_rgb(0_0_0/0.25)]" : "border-white/85 hover:border-primary",
                )}
                style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%` }}
                aria-label={`Choose face ${c.index}`}
              >
                <span
                  className={cx(
                    "absolute -left-0.5 -top-0.5 grid h-5 min-w-5 place-items-center rounded-br-md rounded-tl-md px-1 text-[11px] font-bold",
                    active ? "bg-primary text-primary-content" : "bg-white/90 text-black",
                  )}
                >
                  {c.index}
                </span>
              </button>
            );
          })}
        </div>
      )}
      <div className={cx("grid gap-2", compact ? "grid-cols-2 sm:grid-cols-3" : "mx-auto max-w-xl grid-cols-[repeat(auto-fit,minmax(0,8.5rem))] justify-center")}>
        {choices.map((c) => {
          const busy = choosing === c.index;
          return (
            <button
              key={c.index}
              type="button"
              disabled={!waiting || choosing !== null}
              onClick={() => chooseFace(job.id, c.index)}
              onMouseEnter={() => setHover(c.index)}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(c.index)}
              onBlur={() => setHover(null)}
              className={cx(
                "group relative overflow-hidden rounded-lg bg-base-300 text-left ring-1 transition-all",
                busy ? "ring-2 ring-primary" : hover === c.index ? "ring-primary/70" : "ring-base-content/10",
                waiting && choosing === null && "hover:-translate-y-0.5 hover:shadow-md",
                choosing !== null && !busy && "opacity-50",
              )}
              aria-label={`Draw face ${c.index}`}
            >
              <div className="aspect-[4/5] w-full">
                {c.image_url ? (
                  <img src={fileUrl(c.image_url)} alt={`Face ${c.index}`} className="h-full w-full object-cover" loading="lazy" />
                ) : (
                  <div className="grid h-full place-items-center">
                    <UserIcon size={20} className="opacity-40" />
                  </div>
                )}
              </div>
              <div className="flex items-center gap-1 whitespace-nowrap bg-base-100/90 px-1.5 py-1 text-[11px] font-medium">
                {busy ? <span className="loading loading-spinner loading-xs" /> : <UserIcon size={11} className="text-primary" />}
                Face {c.index}
              </div>
              {busy && <div className="absolute inset-0 bg-primary/10" />}
            </button>
          );
        })}
      </div>
      {photo && (
        <button type="button" className="btn btn-ghost btn-xs" onClick={() => setShowPhoto((v) => !v)}>
          {showPhoto ? "Hide photo" : "Show photo with face boxes"}
        </button>
      )}
    </div>
  );
}
