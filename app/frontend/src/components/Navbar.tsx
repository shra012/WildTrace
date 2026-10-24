"use client";

import { setGalleryOpen, useStudio } from "@/lib/studio";
import { ActivityIcon, GridIcon } from "./icons";
import StatusPills from "./StatusPills";
import ThemeToggle from "./ThemeToggle";

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 64 64" aria-hidden="true" className="shrink-0">
      <rect width="64" height="64" rx="16" className="fill-primary/15" />
      <path d="M12 44c6-14 12-20 18-14s10 4 14-6 6-10 8-8" fill="none" className="stroke-primary" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="52" cy="16" r="5" className="fill-accent" />
    </svg>
  );
}

export default function Navbar() {
  const mock = useStudio((s) => s.mock);
  const online = useStudio((s) => s.online);
  return (
    <header className="navbar min-h-14 gap-2 border-b border-base-content/10 bg-base-100/75 px-3 backdrop-blur sm:px-4">
      <div className="flex min-w-0 flex-1 items-center gap-2.5">
        <Logo />
        <div className="min-w-0 leading-tight">
          <div className="truncate text-[15px] font-semibold tracking-tight">
            WildTrace <span className="font-normal text-base-content/60">Studio</span>
          </div>
          <div className="hidden truncate text-[11px] text-base-content/50 sm:block">Photos and prompts to pen strokes, drawn by a simulated xArm 7</div>
        </div>
        {mock && <span className="badge badge-accent badge-sm badge-soft ml-1">Mock data</span>}
      </div>
      <div className="hidden md:block">
        <StatusPills />
      </div>
      <div className="dropdown dropdown-end md:hidden">
        <button type="button" tabIndex={0} className="btn btn-ghost btn-sm gap-1.5" aria-label="System status">
          <span className={`status ${online === false ? "status-error" : online ? "status-success" : "status-warning"}`} />
          <ActivityIcon size={16} />
        </button>
        <div tabIndex={0} className="dropdown-content z-40 mt-2 w-64 rounded-box border border-base-content/10 bg-base-100 p-3 shadow-xl">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider opacity-60">System status</div>
          <StatusPills vertical />
        </div>
      </div>
      <button type="button" className="btn btn-ghost btn-sm gap-1.5" onClick={() => setGalleryOpen(true)}>
        <GridIcon size={16} />
        <span className="hidden sm:inline">Gallery</span>
      </button>
      <ThemeToggle />
    </header>
  );
}
