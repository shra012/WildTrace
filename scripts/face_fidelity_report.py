#!/usr/bin/env python3
"""Compare ears-off/glasses-off portraits with ears-on, judge-selected glasses.

    uv run --extra app python scripts/face_fidelity_report.py
    uv run --extra app python scripts/face_fidelity_report.py --backend openrouter

Only the fixed Open Images test faces are used; uploaded Studio photos are excluded.
Outputs stay separate from the source work and the physical-arm input trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import textwrap

from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wildtrace.config import load_runtime_config
from wildtrace.face_fidelity import HOSTED_MODEL, LOCAL_MODEL, PROMPT, SCHEMA_HASH, _checklist, judge_face_fidelity
from wildtrace.face_lineart import generate_face_trajectory

SAMPLES = ("58265341041a89bb", "fb1b8bcaf2415905", "cdb9aedb5778896b", "f4d9516ce50ff090",
           "e00984bb56fa1ba6", "d40b77d1", "562adcc8", "2ed5d5c4")


def verdict(source, lineart, backend, ollama_host):
    cache = Path(lineart).with_name("fidelity.json")
    expected = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "lineart_sha256": hashlib.sha256(Path(lineart).read_bytes()).hexdigest(),
                "backend": backend, "model": LOCAL_MODEL if backend == "ollama" else HOSTED_MODEL,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "response_schema_sha256": SCHEMA_HASH}
    if backend == "ollama":
        expected["ollama_host"] = ollama_host
    if cache.exists():
        existing = json.loads(cache.read_text())
        if all(existing.get("provenance", {}).get(k) == v for k, v in expected.items()):
            return {**_checklist(existing), "provenance": {**existing["provenance"], "schema_version": 2}}
    result = judge_face_fidelity(source, lineart, backend=backend, source_is_open_images=True, ollama_host=ollama_host)
    cache.write_text(json.dumps(result, indent=2))
    return result


def renderer_checks(trajectory_path, fidelity):
    labels = json.loads(Path(trajectory_path).read_text())["stroke_labels"]
    count = labels.count("glasses")
    observed = fidelity["features"]["glasses"]["drawing"]
    return {"glasses_strokes": count, "judge_glasses_disagreement":
            observed == "present" and count == 0 or observed == "absent" and count > 0}


def sheet(source: Path, before, after, destination: Path, *, credit: str):
    images = [Image.open(p).convert("RGB") for p in (source, before["lineart_path"], after["lineart_path"])]
    notes = [f"Photo; {credit}"]
    for label, result in (("Before", before), ("After", after)):
        verdict, stats = result["verdict"], result["stats"]
        notes.append(f"{label}: resemblance {verdict['resemblance']}/5; {stats['stroke_count']} strokes; "
                     f"estimated {stats['est_draw_seconds']:.0f} s. "
                     f"Missing: {', '.join(verdict['missing']) or 'none'}. "
                     f"Extra: {', '.join(verdict['extra']) or 'none'}.")
        if result.get("renderer_checks", {}).get("judge_glasses_disagreement"):
            notes[-1] += " Judge's glasses label disagrees with the frame strokes."
    wrapped = [textwrap.wrap(note, width=45) for note in notes]
    canvas = Image.new("RGB", (960, 340 + 16 * max(map(len, wrapped))), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (image, lines) in enumerate(zip(images, wrapped)):
        thumb = ImageOps.contain(image, (310, 310))
        canvas.paste(thumb, (column * 320 + (320 - thumb.width) // 2, (320 - thumb.height) // 2))
        draw.multiline_text((column * 320 + 5, 325), "\n".join(lines), fill="black", spacing=3)
    canvas.save(destination)


def write_markdown(results, backend, out):
    lines = ["# Face fidelity comparison", "", f"Judge backend: {backend}. Resemblance scores are model judgements.", "",
             "Time estimates use a 160 x 120 mm drawing area, 14 mm/s pen speed and 6 seconds per stroke.", "",
             "| Face | Before / after resemblance | Before / after strokes | Before / after estimated seconds | Glasses |", "|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['sample_id']} | Error: {r['error']} | | | |")
            continue
        a, b = r["before"], r["after"]
        glasses = "on" if r["include_glasses"] is True else "off" if r["include_glasses"] is False else "uncertain (off)"
        lines.append(f"| {r['sample_id']} | {a['verdict']['resemblance']} / {b['verdict']['resemblance']} | "
                     f"{a['stats']['stroke_count']} / {b['stats']['stroke_count']} | "
                     f"{a['stats']['est_draw_seconds']} / {b['stats']['est_draw_seconds']} | {glasses} |")
    for r in results:
        if "error" not in r:
            lines += ["", f"## {r['sample_id']}", "", f"![{r['sample_id']}]({r['sample_id']}/comparison.png)", ""]
            for label in ("before", "after"):
                v = r[label]["verdict"]
                lines += [f"{label.title()} missing: {', '.join(v['missing']) or 'none'}. "
                          f"Extra: {', '.join(v['extra']) or 'none'}.", ""]
                checks = r[label]["renderer_checks"]
                lines += [f"{label.title()} frame strokes: {checks['glasses_strokes']}. "
                          f"Judge/frame-stroke disagreement: {checks['judge_glasses_disagreement']}.", ""]
            credit = r["credit"]
            lines += [f"Source: {credit.get('author', 'Open Images')}; "
                      f"[original]({credit.get('original_landing_url', credit['url'])}), "
                      f"[license]({credit.get('license', '')}).", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ollama", "openrouter"), default="ollama")
    parser.add_argument("--out", type=Path, default=REPO / "outputs/face_fidelity/paper_faces")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    load_runtime_config(REPO)  # Load the existing API configuration without printing credentials.
    work = REPO / "outputs/faces_source/work"
    records = [json.loads(line) for line in (REPO / "outputs/faces_source/manifest.ndjson").read_text().splitlines() if line.strip()]
    public = {r["sample_id"]: r for r in records if str(r.get("url", "")).startswith("https://open-images-dataset.s3.amazonaws.com/")}
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for prefix in SAMPLES[:args.limit]:
        matches = sorted(p for p in work.glob(prefix + "*") if p.is_dir())
        if len(matches) != 1 or matches[0].name not in public:
            raise RuntimeError(f"Expected one documented Open Images source for {prefix}")
        sample = matches[0].name
        source = matches[0] / "source.png"
        record = {"sample_id": sample, "source": str(source.relative_to(REPO)), "credit": public[sample]}
        directory = args.out / sample
        try:
            before = generate_face_trajectory(image_path=source, out_dir=directory / "before",
                                             settings={"include_ears": False, "include_glasses": False})
            before["verdict"] = verdict(source, before["lineart_path"], args.backend, args.ollama_host)
            glasses = before["verdict"]["include_glasses"]
            after = generate_face_trajectory(image_path=source, out_dir=directory / "after",
                                            settings={"include_ears": True, "include_glasses": glasses is True})
            after["verdict"] = verdict(source, after["lineart_path"], args.backend, args.ollama_host)
            for key, value in (("before", before), ("after", after)):
                value["renderer_checks"] = renderer_checks(value["trajectory_path"], value["verdict"])
                record[key] = {"stats": value["stats"], "verdict": value["verdict"], "lineart_path": str(value["lineart_path"]),
                               "renderer_checks": value["renderer_checks"]}
            record["include_glasses"] = glasses
            record["comparison_path"] = str(directory / "comparison.png")
            sheet(source, before, after, directory / "comparison.png", credit=public[sample].get("author", "Open Images"))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
        (args.out / "report.json").write_text(json.dumps({"backend": args.backend, "samples": results}, indent=2, default=str))
        print(sample, record.get("error", "done"), flush=True)
    write_markdown(results, args.backend, args.out)
    return int(any("error" in r for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
