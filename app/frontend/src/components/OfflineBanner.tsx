"use client";

import { API_BASE } from "@/lib/config";
import { retryNow, useStudio } from "@/lib/studio";
import { useNow } from "@/hooks/useNow";
import { RefreshIcon, WifiOffIcon } from "./icons";

export default function OfflineBanner() {
  const online = useStudio((s) => s.online);
  const checking = useStudio((s) => s.checkingHealth);
  const nextRetryAt = useStudio((s) => s.nextRetryAt);
  const now = useNow(1000);
  if (online !== false) return null;
  const secs = nextRetryAt && now ? Math.max(0, Math.ceil((nextRetryAt - now) / 1000)) : null;
  return (
    <div className="px-2 pt-2 sm:px-3 sm:pt-3">
      <div role="alert" className="alert alert-warning alert-soft animate-rise flex-wrap gap-x-3 gap-y-2 py-2.5 text-sm">
        <WifiOffIcon size={20} />
        <div className="min-w-0 flex-1">
          <div className="font-semibold">The WildTrace backend is offline</div>
          <div className="text-xs opacity-80">
            Cannot reach <code className="font-mono">{API_BASE}</code>.{" "}
            {checking ? "Checking now..." : secs !== null ? `Retrying in ${secs}s.` : "Retrying automatically."} Everything you see stays usable. For a demo
            without a backend, run with <code className="font-mono">NEXT_PUBLIC_MOCK=1</code> or open <code className="font-mono">?mock=1</code>.
          </div>
        </div>
        <button type="button" className="btn btn-sm btn-warning" onClick={retryNow} disabled={checking}>
          {checking ? <span className="loading loading-spinner loading-xs" /> : <RefreshIcon size={14} />}
          Retry now
        </button>
      </div>
    </div>
  );
}
