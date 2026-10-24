"use client";

import { cx } from "@/lib/cx";
import { dismissToast, useStudio } from "@/lib/studio";
import { AlertIcon, CheckIcon, SparklesIcon, XIcon } from "./icons";

const TONE = {
  info: { cls: "alert-info", Icon: SparklesIcon },
  success: { cls: "alert-success", Icon: CheckIcon },
  warn: { cls: "alert-warning", Icon: AlertIcon },
  error: { cls: "alert-error", Icon: AlertIcon },
} as const;

export default function Toasts() {
  const toasts = useStudio((s) => s.toasts);
  if (!toasts.length) return null;
  return (
    <div className="toast toast-end toast-bottom z-[60] mb-16 max-w-[calc(100vw-1rem)] lg:mb-0" aria-live="polite">
      {toasts.map((t) => {
        const tone = TONE[t.level];
        return (
          <div key={t.id} role="status" className={cx("alert alert-soft animate-rise w-80 max-w-full items-start py-2.5 text-sm shadow-lg", tone.cls)}>
            <tone.Icon size={18} className="mt-0.5 shrink-0" />
            <span className="min-w-0 flex-1 break-words">{t.message}</span>
            <button type="button" className="btn btn-ghost btn-xs btn-circle -mr-1" onClick={() => dismissToast(t.id)} aria-label="Dismiss">
              <XIcon size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
