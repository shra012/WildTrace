"use client";

import { fileUrl } from "@/lib/config";
import { cx } from "@/lib/cx";
import { relativeTime } from "@/lib/format";
import { openPreview, retryMessage, type ChatMessage } from "@/lib/studio";
import JobCard from "../cards/JobCard";
import RunCard from "../cards/RunCard";
import { RefreshIcon, RobotIcon } from "../icons";

const KIND_LABEL = { auto: "Auto", face: "Face", animal: "Animal" } as const;

export default function MessageBubble({ msg, now }: { msg: ChatMessage; now: number }) {
  const mine = msg.role === "user";
  const img = fileUrl(msg.imageUrl);
  return (
    <div className={cx("chat animate-rise", mine ? "chat-end" : "chat-start")}>
      {!mine && (
        <div className="chat-image">
          <div className="grid h-8 w-8 place-items-center rounded-full bg-primary/15 text-primary ring-1 ring-primary/25">
            <RobotIcon size={16} />
          </div>
        </div>
      )}
      <div
        className={cx(
          "chat-bubble max-w-[88%] text-sm leading-relaxed",
          mine ? "chat-bubble-primary" : msg.error ? "chat-bubble-error" : "bg-base-200 text-base-content",
        )}
      >
        {img && (
          <button type="button" className="mb-1.5 block overflow-hidden rounded-lg" onClick={() => openPreview(img, "Your photo")}>
            <img src={img} alt="Attached" className="max-h-44 w-auto max-w-full object-cover" />
          </button>
        )}
        {img && msg.imageKind && (
          <div className="mb-1 text-[11px] uppercase tracking-wider opacity-70">Photo, {KIND_LABEL[msg.imageKind]}</div>
        )}
        {msg.pending ? (
          <span className="loading loading-dots loading-sm" aria-label="Thinking" />
        ) : (
          msg.text && <p className="whitespace-pre-wrap break-words">{msg.text}</p>
        )}
        {msg.error && msg.retryText && (
          <button type="button" className="btn btn-xs mt-2 gap-1" onClick={() => retryMessage(msg.id)}>
            <RefreshIcon size={12} /> Retry
          </button>
        )}
      </div>
      {(msg.jobId || msg.runId) && !mine && (
        <div className="chat-footer block w-full max-w-sm opacity-100">
          {msg.jobId && !msg.runId && <JobCard jobId={msg.jobId} />}
          {msg.runId && <RunCard runId={msg.runId} />}
        </div>
      )}
      <div className="chat-footer mt-0.5 text-[11px] opacity-50">{relativeTime(msg.at, now)}</div>
    </div>
  );
}
