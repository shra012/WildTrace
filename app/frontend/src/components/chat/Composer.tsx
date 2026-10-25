"use client";

import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import { cx } from "@/lib/cx";
import { pushToast, sendChat, sendImage, useStudio } from "@/lib/studio";
import { CATALOG_CATEGORIES, type ImageJobKind } from "@/lib/types";
import { ImageIcon, PaperclipIcon, SendIcon, XIcon } from "../icons";

export interface ComposerHandle {
  attach(file: File): void;
  pickFile(kind?: ImageJobKind): void;
}

const MAX_BYTES = 15 * 1024 * 1024;
const KINDS: { value: ImageJobKind; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "face", label: "Face" },
  { value: "animal", label: "Animal" },
];
const ANIMALS = CATALOG_CATEGORIES.filter((c) => c !== "Face");

function formatBytes(n: number): string {
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Composer({ ref }: { ref?: Ref<ComposerHandle> }) {
  const busy = useStudio((s) => s.chatBusy);
  const [text, setText] = useState("");
  const [attachment, setAttachment] = useState<{ file: File; url: string } | null>(null);
  const [kind, setKind] = useState<ImageJobKind>("auto");
  const [category, setCategory] = useState<string>("");
  const fileInput = useRef<HTMLInputElement>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    urlRef.current = attachment?.url ?? null;
  }, [attachment]);

  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  const attach = (file: File) => {
    if (!file.type.startsWith("image/")) {
      pushToast("warn", "Only image files can be attached");
      return;
    }
    if (file.size > MAX_BYTES) {
      pushToast("warn", `That image is ${formatBytes(file.size)}; the limit is 15 MB`);
      return;
    }
    setAttachment((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      return { file, url: URL.createObjectURL(file) };
    });
    textarea.current?.focus();
  };

  const clearAttachment = () => {
    setAttachment((prev) => {
      if (prev) URL.revokeObjectURL(prev.url);
      return null;
    });
    setCategory("");
  };

  useImperativeHandle(ref, () => ({
    attach,
    pickFile(k?: ImageJobKind) {
      if (k) setKind(k);
      fileInput.current?.click();
    },
  }));

  const canSend = !busy && (text.trim().length > 0 || !!attachment);

  const submit = () => {
    if (!canSend) return;
    if (attachment) {
      void sendImage(attachment.file, kind, kind === "animal" && category ? category : undefined, text);
      // The chat message keeps its own object URL, so do not revoke this one.
      setAttachment(null);
      setCategory("");
    } else {
      void sendChat(text);
    }
    setText("");
  };

  const rows = Math.min(5, Math.max(1, text.split("\n").length));

  return (
    <div className="border-t border-base-content/10 bg-base-100/60 p-3">
      {attachment && (
        <div className="animate-rise mb-2.5 rounded-box border border-base-content/10 bg-base-200/70 p-2">
          <div className="flex items-center gap-2.5">
            <img src={attachment.url} alt="Attachment preview" className="h-12 w-12 shrink-0 rounded-lg object-cover ring-1 ring-base-content/10" />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium">{attachment.file.name || "Pasted image"}</div>
              <div className="text-xs text-base-content/60">{formatBytes(attachment.file.size)}</div>
            </div>
            <button type="button" className="btn btn-ghost btn-xs btn-circle" onClick={clearAttachment} aria-label="Remove attachment">
              <XIcon size={14} />
            </button>
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className="text-xs text-base-content/60">Treat as</span>
            <div className="join" role="radiogroup" aria-label="Image kind">
              {KINDS.map((k) => (
                <button
                  key={k.value}
                  type="button"
                  role="radio"
                  aria-checked={kind === k.value}
                  className={cx("btn join-item btn-xs", kind === k.value ? "btn-primary" : "btn-ghost bg-base-100")}
                  onClick={() => setKind(k.value)}
                >
                  {k.label}
                </button>
              ))}
            </div>
            {kind === "animal" && (
              <select className="select select-xs w-32" value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Animal category">
                <option value="">Any animal</option>
                {ANIMALS.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            )}
          </div>
        </div>
      )}
      <div className="flex items-end gap-2">
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) attach(f);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          className="btn btn-ghost btn-square btn-sm mb-0.5"
          onClick={() => fileInput.current?.click()}
          aria-label="Attach an image"
          title="Attach an image (or drop / paste one)"
        >
          {attachment ? <ImageIcon size={18} className="text-primary" /> : <PaperclipIcon size={18} />}
        </button>
        <textarea
          ref={textarea}
          className="textarea textarea-sm min-h-0 flex-1 resize-none py-2 text-sm leading-snug"
          rows={rows}
          placeholder={attachment ? "Add a note (optional)" : "Ask for a drawing, e.g. draw a cat"}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          onPaste={(e) => {
            const item = Array.from(e.clipboardData.items).find((i) => i.kind === "file" && i.type.startsWith("image/"));
            const f = item?.getAsFile();
            if (f) {
              e.preventDefault();
              attach(f);
            }
          }}
          aria-label="Message"
        />
        <button type="button" className="btn btn-primary btn-square btn-sm mb-0.5" onClick={submit} disabled={!canSend} aria-label="Send">
          {busy ? <span className="loading loading-spinner loading-xs" /> : <SendIcon size={16} />}
        </button>
      </div>
      <div className="mt-1.5 hidden text-[11px] text-base-content/40 sm:block">Enter to send, Shift+Enter for a new line. Drop or paste a photo to attach it.</div>
    </div>
  );
}
