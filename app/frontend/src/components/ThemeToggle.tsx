"use client";

import { setTheme, useTheme } from "@/lib/theme";
import { MoonIcon, SunIcon } from "./icons";

export default function ThemeToggle() {
  const theme = useTheme();
  const dark = theme === "wildtrace-dark";
  return (
    <button
      type="button"
      className="btn btn-ghost btn-sm btn-circle"
      onClick={() => setTheme(dark ? "wildtrace-light" : "wildtrace-dark")}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      title={dark ? "Light theme" : "Dark theme"}
    >
      <span key={theme} className="animate-fade-in">{dark ? <SunIcon /> : <MoonIcon />}</span>
    </button>
  );
}
