// Small procedural line drawings used by mock mode (catalog thumbnails, job artifacts, traces).
// Coordinates are normalized canvas units (0..1, y down), matching the trajectory contract.

import type { Point, Polyline } from "./types";

const TAU = Math.PI * 2;

interface EllipseOpts {
  a0?: number;
  a1?: number;
  n?: number;
  rot?: number;
}

function ellipse(cx: number, cy: number, rx: number, ry: number, opts: EllipseOpts = {}): Polyline {
  const { a0 = 0, a1 = TAU, rot = 0 } = opts;
  const n = opts.n ?? Math.max(10, Math.round(((rx + ry) / 2) * 260 * ((a1 - a0) / TAU)));
  const out: Polyline = [];
  const cr = Math.cos(rot);
  const sr = Math.sin(rot);
  for (let i = 0; i <= n; i++) {
    const t = a0 + ((a1 - a0) * i) / n;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    out.push([cx + x * cr - y * sr, cy + x * sr + y * cr]);
  }
  return out;
}

/** Catmull-Rom spline through the control points. */
function smooth(points: Point[], samples = 8): Polyline {
  if (points.length < 3) return points.map((p) => [p[0], p[1]] as Point);
  const out: Polyline = [];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    for (let s = 0; s < samples; s++) {
      const t = s / samples;
      const t2 = t * t;
      const t3 = t2 * t;
      const f = (a: number, b: number, c: number, d: number) =>
        0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
      out.push([f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])]);
    }
  }
  const last = points[points.length - 1];
  out.push([last[0], last[1]]);
  return out;
}

function mirrorX(points: Point[]): Point[] {
  return points.map(([x, y]) => [1 - x, y] as Point);
}

function cat(): Polyline[] {
  return [
    ellipse(0.5, 0.32, 0.14, 0.12),
    [[0.385, 0.26], [0.4, 0.13], [0.46, 0.215]],
    [[0.54, 0.215], [0.6, 0.13], [0.615, 0.26]],
    ellipse(0.45, 0.31, 0.02, 0.025),
    ellipse(0.55, 0.31, 0.02, 0.025),
    [[0.49, 0.355], [0.51, 0.355], [0.5, 0.37], [0.49, 0.355]],
    smooth([[0.47, 0.392], [0.485, 0.402], [0.5, 0.372], [0.515, 0.402], [0.53, 0.392]], 4),
    [[0.42, 0.37], [0.31, 0.35]],
    [[0.42, 0.385], [0.31, 0.405]],
    [[0.58, 0.37], [0.69, 0.35]],
    [[0.58, 0.385], [0.69, 0.405]],
    smooth([[0.42, 0.43], [0.36, 0.55], [0.33, 0.72], [0.35, 0.84], [0.5, 0.86], [0.65, 0.84], [0.67, 0.72], [0.64, 0.55], [0.58, 0.43]]),
    [[0.45, 0.62], [0.45, 0.85]],
    [[0.55, 0.62], [0.55, 0.85]],
    smooth([[0.65, 0.82], [0.76, 0.8], [0.82, 0.7], [0.8, 0.6], [0.76, 0.57]]),
  ];
}

function dog(): Polyline[] {
  const ear: Point[] = [[0.4, 0.23], [0.33, 0.27], [0.32, 0.38], [0.37, 0.4], [0.4, 0.32]];
  return [
    ellipse(0.5, 0.3, 0.12, 0.11),
    smooth(ear, 6),
    smooth(mirrorX(ear), 6),
    ellipse(0.5, 0.36, 0.055, 0.04),
    ellipse(0.5, 0.345, 0.018, 0.012),
    ellipse(0.455, 0.285, 0.014, 0.016),
    ellipse(0.545, 0.285, 0.014, 0.016),
    smooth([[0.49, 0.4], [0.488, 0.43], [0.512, 0.43], [0.51, 0.4]], 5),
    smooth([[0.42, 0.43], [0.5, 0.455], [0.58, 0.43]], 6),
    smooth([[0.42, 0.42], [0.37, 0.55], [0.34, 0.72], [0.34, 0.85], [0.5, 0.87], [0.66, 0.85], [0.66, 0.72], [0.63, 0.55], [0.58, 0.42]]),
    [[0.44, 0.6], [0.44, 0.86]],
    [[0.56, 0.6], [0.56, 0.86]],
    smooth([[0.66, 0.84], [0.75, 0.83], [0.8, 0.78], [0.83, 0.7]], 6),
  ];
}

function bird(): Polyline[] {
  return [
    ellipse(0.52, 0.5, 0.17, 0.11, { rot: -0.35 }),
    ellipse(0.36, 0.36, 0.075, 0.07),
    [[0.29, 0.345], [0.235, 0.37], [0.29, 0.385]],
    ellipse(0.345, 0.35, 0.012, 0.012),
    smooth([[0.45, 0.45], [0.55, 0.43], [0.65, 0.47], [0.7, 0.55], [0.58, 0.55], [0.48, 0.51]]),
    [[0.66, 0.56], [0.84, 0.63], [0.82, 0.68], [0.64, 0.6]],
    [[0.5, 0.6], [0.49, 0.7]],
    [[0.55, 0.6], [0.555, 0.7]],
    smooth([[0.15, 0.72], [0.45, 0.7], [0.75, 0.71], [0.9, 0.69]]),
    [[0.7, 0.71], [0.78, 0.64]],
    ellipse(0.8, 0.62, 0.035, 0.015, { rot: -0.6 }),
  ];
}

function fish(): Polyline[] {
  return [
    smooth([[0.2, 0.5], [0.3, 0.38], [0.48, 0.33], [0.62, 0.38], [0.72, 0.48]]),
    smooth([[0.2, 0.5], [0.3, 0.6], [0.48, 0.65], [0.62, 0.6], [0.72, 0.52]]),
    [[0.72, 0.48], [0.86, 0.36], [0.83, 0.5], [0.86, 0.64], [0.72, 0.52]],
    ellipse(0.29, 0.47, 0.02, 0.02),
    smooth([[0.36, 0.4], [0.39, 0.5], [0.36, 0.6]]),
    smooth([[0.45, 0.335], [0.5, 0.25], [0.6, 0.28], [0.6, 0.37]]),
    smooth([[0.44, 0.53], [0.5, 0.6], [0.52, 0.54]]),
    ellipse(0.5, 0.47, 0.03, 0.035, { a0: -Math.PI / 2, a1: Math.PI / 2 }),
    ellipse(0.58, 0.47, 0.03, 0.035, { a0: -Math.PI / 2, a1: Math.PI / 2 }),
    ellipse(0.14, 0.38, 0.015, 0.015),
    ellipse(0.11, 0.3, 0.022, 0.022),
  ];
}

function frog(): Polyline[] {
  const backLeg: Point[] = [[0.27, 0.66], [0.17, 0.72], [0.2, 0.82], [0.28, 0.84]];
  const frontLeg: Point[] = [[0.36, 0.72], [0.33, 0.82], [0.28, 0.85]];
  return [
    smooth([[0.25, 0.55], [0.3, 0.42], [0.4, 0.38], [0.5, 0.39], [0.6, 0.38], [0.7, 0.42], [0.75, 0.55], [0.7, 0.7], [0.5, 0.75], [0.3, 0.7], [0.25, 0.55]]),
    ellipse(0.38, 0.36, 0.065, 0.06),
    ellipse(0.62, 0.36, 0.065, 0.06),
    ellipse(0.38, 0.36, 0.025, 0.03),
    ellipse(0.62, 0.36, 0.025, 0.03),
    smooth([[0.33, 0.53], [0.42, 0.58], [0.5, 0.59], [0.58, 0.58], [0.67, 0.53]]),
    smooth(frontLeg, 5),
    smooth(mirrorX(frontLeg), 5),
    smooth(backLeg, 6),
    smooth(mirrorX(backLeg), 6),
    smooth([[0.12, 0.86], [0.3, 0.9], [0.5, 0.91], [0.7, 0.9], [0.88, 0.86]]),
  ];
}

function horse(): Polyline[] {
  return [
    smooth([[0.58, 0.12], [0.55, 0.22], [0.47, 0.27], [0.36, 0.36], [0.25, 0.5], [0.2, 0.58], [0.22, 0.64], [0.3, 0.66], [0.38, 0.62], [0.46, 0.6], [0.52, 0.66], [0.58, 0.8], [0.62, 0.9]]),
    [[0.55, 0.22], [0.6, 0.1], [0.63, 0.23]],
    smooth([[0.63, 0.23], [0.7, 0.35], [0.76, 0.55], [0.8, 0.75], [0.84, 0.9]]),
    smooth([[0.64, 0.26], [0.72, 0.31], [0.74, 0.4]], 5),
    smooth([[0.69, 0.36], [0.77, 0.42], [0.78, 0.5]], 5),
    smooth([[0.73, 0.48], [0.8, 0.55], [0.81, 0.63]], 5),
    ellipse(0.46, 0.36, 0.022, 0.014, { rot: -0.4 }),
    smooth([[0.25, 0.55], [0.27, 0.57], [0.26, 0.59]], 4),
    [[0.22, 0.62], [0.28, 0.62]],
    smooth([[0.38, 0.62], [0.44, 0.52], [0.52, 0.48]], 6),
  ];
}

function face(): Polyline[] {
  const brow: Point[] = [[0.39, 0.41], [0.43, 0.395], [0.47, 0.405]];
  return [
    ellipse(0.5, 0.5, 0.2, 0.27),
    smooth([[0.29, 0.48], [0.28, 0.32], [0.36, 0.2], [0.5, 0.16], [0.64, 0.2], [0.72, 0.32], [0.71, 0.48]]),
    smooth([[0.32, 0.33], [0.42, 0.3], [0.52, 0.24], [0.6, 0.3], [0.69, 0.34]]),
    smooth(brow, 5),
    smooth(mirrorX(brow), 5),
    ellipse(0.43, 0.45, 0.035, 0.016),
    ellipse(0.57, 0.45, 0.035, 0.016),
    ellipse(0.43, 0.45, 0.009, 0.011),
    ellipse(0.57, 0.45, 0.009, 0.011),
    smooth([[0.5, 0.46], [0.49, 0.55], [0.47, 0.58], [0.5, 0.592], [0.53, 0.58]], 5),
    smooth([[0.43, 0.66], [0.5, 0.69], [0.57, 0.66]], 6),
    ellipse(0.295, 0.5, 0.025, 0.05, { a0: Math.PI / 2, a1: (3 * Math.PI) / 2 }),
    ellipse(0.705, 0.5, 0.025, 0.05, { a0: -Math.PI / 2, a1: Math.PI / 2 }),
    [[0.44, 0.765], [0.44, 0.88]],
    [[0.56, 0.765], [0.56, 0.88]],
  ];
}

const BUILDERS: Record<string, () => Polyline[]> = {
  Cat: cat,
  Dog: dog,
  Bird: bird,
  Fish: fish,
  Frog: frog,
  Horse: horse,
  Face: face,
};

export interface DrawingVariant {
  scale?: number;
  dx?: number;
  dy?: number;
  rot?: number;
  mirror?: boolean;
}

export function drawingFor(category: string, v: DrawingVariant = {}): Polyline[] {
  const build = BUILDERS[category] ?? cat;
  const { scale = 1, dx = 0, dy = 0, rot = 0, mirror = false } = v;
  const c = Math.cos(rot);
  const s = Math.sin(rot);
  return build().map((line) =>
    line.map(([x0, y0]) => {
      const x = (mirror ? 1 - x0 : x0) - 0.5;
      const y = y0 - 0.5;
      const xr = (x * c - y * s) * scale + 0.5 + dx;
      const yr = (x * s + y * c) * scale + 0.5 + dy;
      return [round(xr), round(yr)] as Point;
    }),
  );
}

function round(n: number): number {
  return Math.round(n * 10000) / 10000;
}

export function pathLength(line: Polyline): number {
  let len = 0;
  for (let i = 1; i < line.length; i++) {
    len += Math.hypot(line[i][0] - line[i - 1][0], line[i][1] - line[i - 1][1]);
  }
  return len;
}

function pointsAttr(line: Polyline, size: number): string {
  return line.map(([x, y]) => `${(x * size).toFixed(1)},${(y * size).toFixed(1)}`).join(" ");
}

function svgDataUri(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

/** White background, black strokes, like the pipeline's gold SVG. */
export function strokesSvg(strokes: Polyline[], opts: { colored?: boolean; width?: number } = {}): string {
  const size = 512;
  const width = opts.width ?? 4;
  const body = strokes
    .map((line, i) => {
      const color = opts.colored ? `hsl(${Math.round((i * 360) / Math.max(1, strokes.length))} 70% 42%)` : "#111";
      return `<polyline points="${pointsAttr(line, size)}" stroke="${color}"/>`;
    })
    .join("");
  return svgDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">` +
      `<rect width="${size}" height="${size}" fill="#fff"/>` +
      `<g fill="none" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round">${body}</g></svg>`,
  );
}

/** White silhouette of the first (outline) stroke on black, like a segmentation mask. */
export function maskSvg(strokes: Polyline[]): string {
  const size = 512;
  const fills = strokes
    .slice(0, 2)
    .map((line) => `<polygon points="${pointsAttr(line, size)}" fill="#fff"/>`)
    .join("");
  return svgDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">` +
      `<rect width="${size}" height="${size}" fill="#000"/>${fills}</svg>`,
  );
}

const PHOTO_TONES: Record<string, [string, string, string]> = {
  Cat: ["#c9a27e", "#6b4f3a", "#e9dccb"],
  Dog: ["#d1a15f", "#5a3d22", "#cfe0c3"],
  Bird: ["#e0703a", "#3d5a80", "#b9d7ea"],
  Fish: ["#f0a030", "#1f4e79", "#7fc4d8"],
  Frog: ["#5fa34a", "#2c4a1e", "#cfe6b8"],
  Horse: ["#8a5a3b", "#3b2618", "#e6d3b3"],
  Face: ["#e2b192", "#3a2a22", "#d9d4ec"],
};

/** A soft, photo-like placeholder (blurred subject over a gradient backdrop). */
export function photoSvg(category: string, strokes: Polyline[]): string {
  const size = 512;
  const [subject, dark, backdrop] = PHOTO_TONES[category] ?? PHOTO_TONES.Cat;
  const fills = strokes
    .slice(0, 3)
    .map((line, i) => `<polygon points="${pointsAttr(line, size)}" fill="${i === 0 ? subject : dark}" opacity="${i === 0 ? 1 : 0.55}"/>`)
    .join("");
  return svgDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">` +
      `<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${backdrop}"/><stop offset="1" stop-color="${dark}" stop-opacity="0.55"/></linearGradient>` +
      `<filter id="b"><feGaussianBlur stdDeviation="5"/></filter></defs>` +
      `<rect width="${size}" height="${size}" fill="url(#g)"/>` +
      `<ellipse cx="256" cy="470" rx="230" ry="40" fill="${dark}" opacity="0.25" filter="url(#b)"/>` +
      `<g filter="url(#b)">${fills}</g></svg>`,
  );
}

const SKIN = ["#f1c7a5", "#c68b62", "#e6b08a", "#8d5a3c", "#f3d2b8"];
const HAIR = ["#2b1d14", "#6b4423", "#121212", "#b07a3c", "#4a3b32"];
const SHIRT = ["#4f7cac", "#c0616b", "#5d9b74", "#d9a441", "#7b68b0"];

/** A group photo placeholder with `n` people, their face boxes and head-and-shoulders crops. */
export function groupPhoto(n: number): { photo: string; crops: string[]; boxes: [number, number, number, number][] } {
  const W = 800;
  const H = 500;
  const people: string[] = [];
  const boxes: [number, number, number, number][] = [];
  const centers: number[] = [];
  for (let i = 0; i < n; i++) {
    const cx = ((i + 0.5) / n) * W + (i % 2 ? 6 : -6);
    const cy = 200 + (i % 2 ? 18 : 0);
    const rx = 46;
    const ry = 58;
    centers.push(cx);
    people.push(
      `<ellipse cx="${cx}" cy="${cy + 230}" rx="105" ry="140" fill="${SHIRT[i % SHIRT.length]}"/>` +
        `<rect x="${cx - 18}" y="${cy + 40}" width="36" height="50" fill="${SKIN[i % SKIN.length]}"/>` +
        `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="${SKIN[i % SKIN.length]}"/>` +
        `<path d="M${cx - rx - 4} ${cy - 5} Q${cx - rx} ${cy - ry - 26} ${cx} ${cy - ry - 18} Q${cx + rx} ${cy - ry - 26} ${cx + rx + 4} ${cy - 5} Q${cx + 10} ${cy - ry + 10} ${cx - rx - 4} ${cy - 5}Z" fill="${HAIR[i % HAIR.length]}"/>` +
        `<circle cx="${cx - 16}" cy="${cy + 2}" r="4" fill="#222"/><circle cx="${cx + 16}" cy="${cy + 2}" r="4" fill="#222"/>` +
        `<path d="M${cx - 15} ${cy + 28} Q${cx} ${cy + 40} ${cx + 15} ${cy + 28}" stroke="#7a3b2e" stroke-width="4" fill="none" stroke-linecap="round"/>`,
    );
    const pad = 12;
    boxes.push([
      round((cx - rx - pad) / W),
      round((cy - ry - 26) / H),
      round((cx + rx + pad) / W),
      round((cy + ry + pad) / H),
    ]);
  }
  const scene =
    `<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#cfe3ef"/><stop offset="1" stop-color="#9fb8a0"/></linearGradient></defs>` +
    `<rect width="${W}" height="${H}" fill="url(#bg)"/>` +
    `<rect y="${H * 0.72}" width="${W}" height="${H * 0.28}" fill="#7f9a72" opacity="0.6"/>` +
    people.join("");
  const photo = svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${scene}</svg>`);
  const crops = centers.map((cx, i) => {
    const top = 200 + (i % 2 ? 18 : 0) - 120;
    return svgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${cx - 120} ${top} 240 260" width="240" height="260">${scene}</svg>`);
  });
  return { photo, crops, boxes };
}
