"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { fileUrl } from "@/lib/config";
import { cx } from "@/lib/cx";
import { errorMessage } from "@/lib/errors";
import { pickFromCatalog, setGalleryOpen, useStudio } from "@/lib/studio";
import { CATALOG_CATEGORIES, type CatalogItem } from "@/lib/types";
import { GridIcon, LayersIcon, PlayIcon, RefreshIcon, XIcon } from "../icons";

const FILTERS = ["All", ...CATALOG_CATEGORIES] as const;

interface Loaded {
  key: string;
  items: CatalogItem[] | null;
  error: string | null;
}

const SOURCE_TONE: Record<string, string> = {
  gold: "badge-warning",
  generated: "badge-secondary",
  studio: "badge-info",
};

export default function GalleryPanel() {
  const open = useStudio((s) => s.galleryOpen);
  const [category, setCategory] = useState<(typeof FILTERS)[number]>("All");
  const [nonce, setNonce] = useState(0);
  const [loaded, setLoaded] = useState<Loaded>({ key: "", items: null, error: null });
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState<"use" | "draw" | null>(null);
  const key = `${category}#${nonce}`;

  useEffect(() => {
    if (!open) return;
    const ctrl = new AbortController();
    api
      .catalog({ category: category === "All" ? undefined : category, limit: category === "All" ? 96 : 48 }, ctrl.signal)
      .then((items) => setLoaded({ key, items, error: null }))
      .catch((err) => {
        if (!ctrl.signal.aborted) setLoaded({ key, items: null, error: errorMessage(err) });
      });
    return () => ctrl.abort();
  }, [open, category, key]);

  const current = loaded.key === key ? loaded : null;
  const items = current?.items ?? null;
  const item = items?.find((i) => i.sample_id === selected) ?? null;

  const use = async (draw: boolean) => {
    if (!item) return;
    setBusy(draw ? "draw" : "use");
    await pickFromCatalog(item, draw);
    setBusy(null);
  };

  return (
    <aside className="flex h-full w-full max-w-[100vw] flex-col bg-base-100 sm:w-[560px]" aria-label="Gallery">
      <div className="flex items-center gap-2 border-b border-base-content/10 px-4 py-3">
        <GridIcon size={18} className="text-primary" />
        <h2 className="font-semibold">Gallery</h2>
        <span className="text-xs text-base-content/50">Ready-made trajectories</span>
        <div className="flex-1" />
        <button type="button" className="btn btn-ghost btn-sm btn-circle" onClick={() => setNonce((n) => n + 1)} aria-label="Refresh" title="Refresh">
          <RefreshIcon size={15} />
        </button>
        <button type="button" className="btn btn-ghost btn-sm btn-circle" onClick={() => setGalleryOpen(false)} aria-label="Close gallery">
          <XIcon size={16} />
        </button>
      </div>

      <div className="scroll-thin flex gap-1.5 overflow-x-auto border-b border-base-content/10 px-4 py-2.5">
        {FILTERS.map((c) => (
          <button
            key={c}
            type="button"
            className={cx("btn btn-xs shrink-0 rounded-full", category === c ? "btn-primary" : "btn-ghost bg-base-200")}
            onClick={() => {
              setCategory(c);
              setSelected(null);
            }}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto p-4">
        {!current ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {Array.from({ length: 9 }, (_, i) => (
              <div key={i} className="space-y-2">
                <div className="skeleton aspect-square w-full" />
                <div className="skeleton h-3 w-2/3" />
              </div>
            ))}
          </div>
        ) : current.error ? (
          <div role="alert" className="alert alert-warning alert-soft">
            <span className="flex-1 text-sm">Could not load the catalog: {current.error}</span>
            <button type="button" className="btn btn-sm" onClick={() => setNonce((n) => n + 1)}>
              Retry
            </button>
          </div>
        ) : !items?.length ? (
          <div className="py-16 text-center text-sm text-base-content/60">No trajectories for {category === "All" ? "any category" : category} yet.</div>
        ) : (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {items.map((it) => {
              const active = it.sample_id === selected;
              return (
                <button
                  key={it.sample_id}
                  type="button"
                  onClick={() => setSelected(active ? null : it.sample_id)}
                  className={cx(
                    "animate-fade-in group overflow-hidden rounded-box border bg-base-200/50 text-left transition-all hover:-translate-y-0.5 hover:shadow-lg",
                    active ? "border-primary ring-2 ring-primary/40" : "border-base-content/10",
                  )}
                >
                  <div className="checker relative aspect-square w-full">
                    <img src={fileUrl(it.thumbnail_url)} alt={`${it.category} ${it.sample_id}`} className="h-full w-full object-contain p-2" loading="lazy" />
                    {it.source && it.source !== "gold" && (
                      <span className={cx("badge badge-xs absolute right-1.5 top-1.5", SOURCE_TONE[it.source] ?? "badge-ghost")}>{it.source}</span>
                    )}
                  </div>
                  <div className="space-y-0.5 px-2.5 py-2">
                    <div className="flex items-center gap-1.5">
                      <span className="text-sm font-medium">{it.category}</span>
                      <span className="ml-auto font-mono text-[11px] text-base-content/50">{it.stroke_count} strokes</span>
                    </div>
                    <div className="truncate text-[11px] text-base-content/55">{[it.subcategory, it.angle?.replace(/_/g, " ")].filter(Boolean).join(", ") || " "}</div>
                    <div className="truncate font-mono text-[10px] text-base-content/40">{it.sample_id}</div>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {item && (
        <div className="animate-rise border-t border-base-content/10 bg-base-200/70 p-4">
          <div className="flex items-center gap-3">
            <img src={fileUrl(item.thumbnail_url)} alt="" className="checker h-16 w-16 rounded-lg object-contain ring-1 ring-base-content/10" />
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium">
                {item.subcategory ? `${item.subcategory} ` : ""}
                {item.category.toLowerCase()}
              </div>
              <div className="truncate text-xs text-base-content/55">
                {item.sample_id}, {item.stroke_count} strokes{item.source ? `, ${item.source}` : ""}
              </div>
            </div>
          </div>
          <div className="mt-3 flex gap-2">
            <button type="button" className="btn btn-sm flex-1 gap-1.5" disabled={!!busy} onClick={() => use(false)}>
              {busy === "use" ? <span className="loading loading-spinner loading-xs" /> : <LayersIcon size={14} />}
              Use this
            </button>
            <button type="button" className="btn btn-primary btn-sm flex-1 gap-1.5" disabled={!!busy} onClick={() => use(true)}>
              {busy === "draw" ? <span className="loading loading-spinner loading-xs" /> : <PlayIcon size={13} />}
              Use and draw
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
