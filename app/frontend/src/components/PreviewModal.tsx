"use client";

import { useEffect } from "react";
import { closePreview, useStudio } from "@/lib/studio";
import { ExternalIcon, XIcon } from "./icons";

export default function PreviewModal() {
  const preview = useStudio((s) => s.preview);

  useEffect(() => {
    if (!preview) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closePreview();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [preview]);

  if (!preview) return null;
  const canOpen = !preview.url.startsWith("data:");
  return (
    <div className="modal modal-open modal-bottom sm:modal-middle z-[70]" role="dialog" aria-modal="true" aria-label={preview.title}>
      <div className="modal-box max-w-3xl animate-rise p-0">
        <div className="flex items-center gap-2 border-b border-base-content/10 px-4 py-3">
          <h3 className="min-w-0 flex-1 truncate font-semibold">{preview.title}</h3>
          {canOpen && (
            <a className="btn btn-ghost btn-sm gap-1" href={preview.url} target="_blank" rel="noreferrer">
              <ExternalIcon size={14} /> Open
            </a>
          )}
          <button type="button" className="btn btn-ghost btn-sm btn-circle" onClick={closePreview} aria-label="Close">
            <XIcon size={16} />
          </button>
        </div>
        <div className="checker flex max-h-[75vh] items-center justify-center overflow-auto p-4">
          <img src={preview.url} alt={preview.title} className="max-h-[70vh] w-auto max-w-full object-contain" />
        </div>
      </div>
      <button type="button" className="modal-backdrop" onClick={closePreview} aria-label="Close preview" />
    </div>
  );
}
