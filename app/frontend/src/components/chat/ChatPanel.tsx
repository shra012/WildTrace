"use client";

import { useEffect, useRef, useState, type DragEvent } from "react";
import { cx } from "@/lib/cx";
import { shortBackend } from "@/lib/format";
import { clearChat, sendChat, useStudio } from "@/lib/studio";
import { useNow } from "@/hooks/useNow";
import { ChatIcon, ImageIcon, LibraryIcon, PawIcon, TrashIcon, UserIcon } from "../icons";
import Composer, { type ComposerHandle } from "./Composer";
import MessageBubble from "./MessageBubble";

function hasFiles(e: DragEvent) {
  return Array.from(e.dataTransfer?.types ?? []).includes("Files");
}

function Welcome() {
  const items = [
    { Icon: LibraryIcon, title: "Pick from the catalog", text: '"Draw a cat" picks a validated gold trajectory.' },
    { Icon: PawIcon, title: "Make a new animal", text: '"Make a new sitting dog" runs outline, validate and extract on a fresh photo.' },
    { Icon: UserIcon, title: "Draw your face", text: "Attach a photo and get a portrait the robot can draw." },
  ];
  return (
    <div className="animate-fade-in mx-auto flex max-w-sm flex-col items-center px-2 py-6 text-center">
      <div className="mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-primary/15 text-primary">
        <ChatIcon size={22} />
      </div>
      <h2 className="text-balance text-base font-semibold">What should the robot draw?</h2>
      <p className="mt-1 text-sm text-base-content/60">Ask in plain words, or drop a photo here.</p>
      <ul className="mt-5 w-full space-y-2 text-left">
        {items.map(({ Icon, title, text }) => (
          <li key={title} className="flex gap-3 rounded-box border border-base-content/10 bg-base-200/50 p-3">
            <Icon size={18} className="mt-0.5 shrink-0 text-primary" />
            <div>
              <div className="text-sm font-medium">{title}</div>
              <div className="text-xs text-base-content/60">{text}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ChatPanel({ className }: { className?: string }) {
  const messages = useStudio((s) => s.messages);
  const suggestions = useStudio((s) => s.suggestions);
  const busy = useStudio((s) => s.chatBusy);
  const chatBackend = useStudio((s) => s.health?.backends.chat);
  const now = useNow(30_000);
  const listRef = useRef<HTMLDivElement>(null);
  const composer = useRef<ComposerHandle>(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const onSuggestion = (s: string) => {
    if (/face|photo|upload/i.test(s) && /\(|upload|photo/i.test(s)) {
      composer.current?.pickFile("face");
      return;
    }
    void sendChat(s);
  };

  return (
    <section
      className={cx("panel relative min-w-0 flex-1", className)}
      aria-label="Chat"
      onDragEnter={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => {
        if (hasFiles(e)) e.preventDefault();
      }}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDrop={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        const f = Array.from(e.dataTransfer.files).find((file) => file.type.startsWith("image/")) ?? e.dataTransfer.files[0];
        if (f) composer.current?.attach(f);
      }}
    >
      <div className="panel-header">
        <ChatIcon size={16} className="text-primary" />
        <h2 className="panel-title">Assistant</h2>
        {chatBackend && <span className="badge badge-ghost badge-xs font-mono" title={chatBackend}>{shortBackend(chatBackend)}</span>}
        <div className="flex-1" />
        {messages.length > 0 && (
          <button type="button" className="btn btn-ghost btn-xs gap-1 text-base-content/60" onClick={clearChat} title="Clear conversation">
            <TrashIcon size={13} /> Clear
          </button>
        )}
      </div>

      <div ref={listRef} className="scroll-thin min-h-0 flex-1 space-y-1 overflow-y-auto px-3 py-3">
        {messages.length === 0 ? <Welcome /> : messages.map((m) => <MessageBubble key={m.id} msg={m} now={now} />)}
      </div>

      <div className="flex gap-1.5 overflow-x-auto px-3 pb-2 scroll-thin">
        {suggestions.map((s) => (
          <button key={s} type="button" className="btn btn-xs btn-outline shrink-0 rounded-full border-base-content/20 font-normal" disabled={busy} onClick={() => onSuggestion(s)}>
            {s}
          </button>
        ))}
      </div>

      <Composer ref={composer} />

      {dragging && (
        <div className="animate-fade-in pointer-events-none absolute inset-2 z-10 grid place-items-center rounded-box border-2 border-dashed border-primary bg-base-100/85 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-2 text-primary">
            <ImageIcon size={32} />
            <div className="font-medium">Drop a photo to attach it</div>
            <div className="text-xs text-base-content/60">Faces become portraits, animals go through the pipeline</div>
          </div>
        </div>
      )}
    </section>
  );
}
