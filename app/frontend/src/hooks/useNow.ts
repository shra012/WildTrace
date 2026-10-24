"use client";

import { useSyncExternalStore } from "react";

interface Ticker {
  now: number;
  listeners: Set<() => void>;
  timer: ReturnType<typeof setInterval> | null;
}

const tickers = new Map<number, Ticker>();

function ticker(intervalMs: number): Ticker {
  let t = tickers.get(intervalMs);
  if (!t) {
    t = { now: Date.now(), listeners: new Set(), timer: null };
    tickers.set(intervalMs, t);
  }
  return t;
}

function subscribe(intervalMs: number, listener: () => void): () => void {
  const t = ticker(intervalMs);
  t.listeners.add(listener);
  if (!t.timer) {
    // `now` is not refreshed here: a changed snapshot during subscribe makes React re-render and
    // resubscribe, which loops. The first tick refreshes it.
    t.timer = setInterval(() => {
      t.now = Date.now();
      for (const l of t.listeners) l();
    }, intervalMs);
  }
  return () => {
    t.listeners.delete(listener);
    if (t.listeners.size === 0 && t.timer) {
      clearInterval(t.timer);
      t.timer = null;
    }
  };
}

// One subscribe function per interval: useSyncExternalStore resubscribes whenever it changes identity.
const subscribers = new Map<number, (listener: () => void) => () => void>();

function subscriberFor(intervalMs: number) {
  let fn = subscribers.get(intervalMs);
  if (!fn) {
    fn = (listener) => subscribe(intervalMs, listener);
    subscribers.set(intervalMs, fn);
  }
  return fn;
}

/** Current time, refreshed every `intervalMs`. Returns 0 during server rendering. */
export function useNow(intervalMs = 30_000): number {
  return useSyncExternalStore(
    subscriberFor(intervalMs),
    () => ticker(intervalMs).now,
    () => 0,
  );
}
