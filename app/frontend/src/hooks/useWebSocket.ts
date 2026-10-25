"use client";

import { useEffect, useRef, useState } from "react";

export type SocketStatus = "idle" | "connecting" | "open" | "closed";

interface Options<T> {
  url: string;
  enabled?: boolean;
  onMessage: (data: T) => void;
  onStatus?: (status: SocketStatus) => void;
  /** Reconnect if nothing arrives for this long while open (the backend sends health every 10 s). */
  staleAfterMs?: number;
}

/**
 * WebSocket with automatic reconnect (exponential backoff with jitter, capped at 30 s), stale
 * connection detection, and immediate retry when the browser comes back online or the tab
 * becomes visible again.
 */
export function useWebSocket<T = unknown>({ url, enabled = true, onMessage, onStatus, staleAfterMs = 45_000 }: Options<T>): SocketStatus {
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const onMessageRef = useRef(onMessage);
  const onStatusRef = useRef(onStatus);

  useEffect(() => {
    onMessageRef.current = onMessage;
    onStatusRef.current = onStatus;
  });

  useEffect(() => {
    if (!enabled) return;
    let ws: WebSocket | null = null;
    let disposed = false;
    let attempt = 0;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let staleTimer: ReturnType<typeof setTimeout> | null = null;

    const report = (next: SocketStatus) => {
      if (disposed) return;
      setStatus(next);
      onStatusRef.current?.(next);
    };

    const armStale = () => {
      if (staleTimer) clearTimeout(staleTimer);
      staleTimer = setTimeout(() => ws?.close(4000, "stale"), staleAfterMs);
    };

    const scheduleReconnect = () => {
      if (disposed || retryTimer) return;
      const base = Math.min(30_000, 1000 * 2 ** Math.min(attempt, 5));
      const wait = base * (0.75 + Math.random() * 0.5);
      attempt += 1;
      retryTimer = setTimeout(() => {
        retryTimer = null;
        connect();
      }, wait);
    };

    const connect = () => {
      if (disposed) return;
      report("connecting");
      let socket: WebSocket;
      try {
        socket = new WebSocket(url);
      } catch {
        report("closed");
        scheduleReconnect();
        return;
      }
      ws = socket;
      socket.onopen = () => {
        attempt = 0;
        report("open");
        armStale();
      };
      socket.onmessage = (ev) => {
        armStale();
        if (typeof ev.data !== "string") return;
        try {
          onMessageRef.current(JSON.parse(ev.data) as T);
        } catch {
          // Ignore malformed frames.
        }
      };
      socket.onerror = () => {
        // onclose follows and handles the reconnect.
      };
      socket.onclose = () => {
        if (staleTimer) clearTimeout(staleTimer);
        if (ws === socket) ws = null;
        report("closed");
        scheduleReconnect();
      };
    };

    const kick = () => {
      if (disposed || ws) return;
      if (retryTimer) clearTimeout(retryTimer);
      retryTimer = null;
      attempt = 0;
      connect();
    };
    const onVisible = () => {
      if (!document.hidden) kick();
    };

    // Defer the first connect so React strict-mode double effects do not open two sockets.
    const startTimer = setTimeout(connect, 0);
    window.addEventListener("online", kick);
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      disposed = true;
      clearTimeout(startTimer);
      if (retryTimer) clearTimeout(retryTimer);
      if (staleTimer) clearTimeout(staleTimer);
      window.removeEventListener("online", kick);
      document.removeEventListener("visibilitychange", onVisible);
      if (ws) {
        ws.onclose = null;
        ws.close(1000, "unmount");
      }
      ws = null;
    };
  }, [url, enabled, staleAfterMs]);

  return enabled ? status : "idle";
}
