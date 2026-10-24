#!/usr/bin/env python3
"""Assemble time-lapse MP4s from frames recorded by video_capture_node.py.

    python3 isaac-sim/scripts/make_drawing_videos.py --frames /mnt/c/wildtrace/videos/frog_dense \
        --label Frog --sim-seconds 150.6 --every 5 --out isaac-sim/outputs/videos [--target-seconds 60] [--also-10x]

Needs an ffmpeg with libx264: run under
`uv run --no-project --with imageio-ffmpeg --with pillow python ...` (the
bundled binary is used automatically). Captions are drawn with PIL and
overlaid, since that binary has no drawtext filter.

One video per camera folder. Playback fps is chosen so each video lasts about
--target-seconds; the caption states the real speed-up (sim time / video time).
The last frame is held for 2 s so the finished drawing is readable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

VIEW_NAMES = {"WideFront": "Wide view, front", "WideBack": "Wide view, back", "PaperZoom": "Close-up on the drawing"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _ffmpeg() -> str:
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _caption_png(text: str, width: int) -> str:
    bar = Image.new("RGBA", (width, 54), (0, 0, 0, 140))
    ImageDraw.Draw(bar).text((24, 13), text, font=ImageFont.truetype(FONT, 24), fill=(255, 255, 255, 255))
    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    bar.save(path)
    return path


def encode(frames: Path, out: Path, fps: float, caption: str) -> None:
    width = Image.open(next(iter(sorted(frames.glob("*.jpg"))))).width
    bar = _caption_png(caption, width)
    graph = "[0:v]tpad=stop_mode=clone:stop_duration=2[v];[v][1:v]overlay=0:H-h,format=yuv420p"
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-framerate", f"{fps:.4f}", "-i", str(frames / "%05d.jpg"),
           "-i", bar, "-filter_complex", graph, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-r", "30", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    os.unlink(bar)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sim-seconds", type=float, required=True)
    ap.add_argument("--every", type=int, required=True, help="physics ticks between saved frames (60 Hz)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-seconds", type=float, default=60.0)
    ap.add_argument("--also-10x", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for cam_dir in sorted(p for p in args.frames.iterdir() if p.is_dir()):
        n = len(list(cam_dir.glob("*.jpg")))
        if not n:
            continue
        sim_span = n * args.every / 60.0
        fps = max(n / args.target_seconds, 8.0)
        speed = sim_span / (n / fps)
        view = VIEW_NAMES.get(cam_dir.name, cam_dir.name)
        stem = f"wildtrace_{args.label.lower()}_{cam_dir.name}"
        encode(cam_dir, args.out / f"{stem}.mp4", fps,
               f"WildTrace · {args.label} · {view} · {speed:.1f}× speed · xArm7 in Isaac Sim")
        print(f"{stem}.mp4  frames={n}  fps={fps:.2f}  length={n / fps + 2:.1f}s  speed={speed:.1f}x")
        if args.also_10x:
            fps10 = 10 * 60.0 / args.every
            encode(cam_dir, args.out / f"{stem}_10x.mp4", fps10,
                   f"WildTrace · {args.label} · {view} · 10× speed · xArm7 in Isaac Sim")
            print(f"{stem}_10x.mp4  length={n / fps10 + 2:.1f}s")


if __name__ == "__main__":
    main()
