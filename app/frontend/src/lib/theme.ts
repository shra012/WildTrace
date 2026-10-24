"use client";

import { useSyncExternalStore } from "react";

export type ThemeName = "wildtrace-dark" | "wildtrace-light";
const KEY = "wildtrace-theme";
const listeners = new Set<() => void>();

function current(): ThemeName {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "wildtrace-light" ? "wildtrace-light" : "wildtrace-dark";
}

export function setTheme(theme: ThemeName) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // Not persisted; the toggle still works for this page view.
  }
  for (const l of listeners) l();
}

export function useTheme(): ThemeName {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => {
        listeners.delete(l);
      };
    },
    current,
    () => "wildtrace-dark",
  );
}
