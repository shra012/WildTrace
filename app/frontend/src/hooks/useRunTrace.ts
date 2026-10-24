"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Trace } from "@/lib/types";

interface TraceState {
  runId: string | null;
  trace: Trace | null;
  error: string | null;
}

/**
 * Polls /api/runs/{id}/trace while the run is live (and once more when it ends), so the executed
 * path grows on screen. Telemetry tips fill in between polls.
 */
export function useRunTrace(runId: string | null | undefined, live: boolean, intervalMs = 700) {
  const [state, setState] = useState<TraceState>({ runId: null, trace: null, error: null });

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let inFlight: AbortController | null = null;

    const load = async () => {
      inFlight?.abort();
      const ctrl = new AbortController();
      inFlight = ctrl;
      try {
        const trace = await api.getTrace(runId, ctrl.signal);
        if (!cancelled) setState({ runId, trace, error: null });
      } catch (err) {
        if (!cancelled && !ctrl.signal.aborted) {
          setState((s) => ({ runId, trace: s.runId === runId ? s.trace : null, error: err instanceof Error ? err.message : String(err) }));
        }
      }
      if (!cancelled && live) timer = setTimeout(load, document.hidden ? intervalMs * 4 : intervalMs);
    };
    timer = setTimeout(load, 0);

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
      inFlight?.abort();
    };
  }, [runId, live, intervalMs]);

  const current = state.runId === runId;
  return {
    trace: current ? state.trace : null,
    error: current ? state.error : null,
    loading: !!runId && (!current || (!state.trace && !state.error)),
  };
}
