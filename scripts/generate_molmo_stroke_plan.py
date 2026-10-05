#!/usr/bin/env python3
"""Generate a sparse, ordered stroke plan with Molmo 7B-D.

This deliberately produces sparse control points. Compile the result with the
shared deterministic compiler rather than making a VLM emit dense robot
waypoints directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from wildtrace.trajectory_candidates import compile_stroke_plan, extract_json_plan


PROMPT = """You are planning a single-pen line drawing from the supplied image.
Return JSON only, with no Markdown and no explanation. Use exactly this schema:
{"drawing_id":"<short_id>","strokes":[{"points":[[x,y],[x,y]]}]}

Coordinates are normalized to [0,1]; (0,0) is top-left and y increases downward.
Each stroke is one continuous pen-down trace. A new stroke means lift the pen.
Keep the important visible outlines, preserve stroke order, and use no more than
24 control points per stroke. Do not emit robot joints, 3D coordinates, or pen-up
transfer lines. This is a sparse plan: straight segments between listed points
will be resampled by a deterministic trajectory compiler."""


class MolmoStrokePlanner:
    """Load a Molmo checkpoint once and generate sparse-plan text for images."""

    def __init__(self, model_name: str, *, load_in_4bit: bool, local_files_only: bool) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            local_files_only=local_files_only,
        )
        model_options = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_options)

    def generate_text(self, image: Image.Image, max_new_tokens: int) -> str:
        from transformers import GenerationConfig

        inputs = self._processor.process(images=[image.convert("RGB")], text=PROMPT)
        inputs = {key: value.to(self._model.device).unsqueeze(0) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
                tokenizer=self._processor.tokenizer,
            )
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        return self._processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="RGB line-art PNG/JPEG to plan")
    parser.add_argument("--output", required=True, help="Canonical WildTrace trajectory JSON")
    parser.add_argument("--raw-output", default=None, help="Optional path to store untrusted model text")
    parser.add_argument("--model", default="allenai/Molmo-7B-D-0924")
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--max-segment-length", type=float, default=0.01)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use bitsandbytes 4-bit loading for checkpoints too large for local VRAM.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load only from the local Hugging Face cache; do not make network requests.",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    try:
        planner = MolmoStrokePlanner(
            args.model,
            load_in_4bit=args.load_in_4bit,
            local_files_only=args.local_files_only,
        )
    except ImportError as exc:
        parser.error(f"Molmo 4-bit loading requires bitsandbytes: {exc}")
    generated_text = planner.generate_text(image, args.max_new_tokens)
    if args.raw_output:
        raw_output = Path(args.raw_output)
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        raw_output.write_text(generated_text, encoding="utf-8")

    plan = extract_json_plan(generated_text)
    trajectory = compile_stroke_plan(
        plan,
        drawing_id=Path(args.output).stem,
        max_segment_length=args.max_segment_length,
        max_control_points_per_stroke=24,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(trajectory, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] generated {len(trajectory['strokes'])} strokes: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
