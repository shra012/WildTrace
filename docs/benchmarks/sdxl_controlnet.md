# SDXL + ControlNet outline rectifier

SDXL with scribble ControlNet is an alternative outline generator. It uses the same silhouette contour, species/pose/view prompt, OpenCV gates, retry loop, and trajectory export as FLUX. The production default remains `flux_silhouette_rectifier`.

Historical `r2` numbers were measured before a shared contour bug was fixed. They describe that experiment. They are not measurements of this implementation. Quality and latency for SDXL are unmeasured; see `docs/benchmarks/sdxl_preflight.json`.

## Checkpoints

| Role | Checkpoint |
| --- | --- |
| Generator | `stabilityai/stable-diffusion-xl-base-1.0` |
| ControlNet | `xinsir/controlnet-scribble-sdxl-1.0` |
| VAE | `madebyollin/sdxl-vae-fp16-fix` |

Settings use the `sdxl_` prefix in `configs/models.yaml`, so FLUX GGUF keys are not reused as SDXL checkpoints.

## Generation

- The shared helper builds the contour at 512 px, then the rectifier resizes it to 1024 px.
- Conditioning is white strokes on black, which is what the scribble ControlNet expects.
- Defaults are 30 steps, guidance scale 7.0, and ControlNet scale 1.0.
- The drawing is binarized and resized back to the crop.
- The seed is `sdxl_seed` (or `seed`) plus a CRC of `sample_id:attempt_filename`. A retry writes a new filename, so it gets a new seed. The same sample and filename repeat the same seed.
- The retry loop still edits `strength`. SDXL turns that into a ControlNet scale: nominal strength 0.85 keeps the configured scale, and each 0.05 of strength moves the scale by 0.20 in the opposite direction. Scale stays between 0.35 and 1.5.
- CUDA loads in float16 and uses model CPU offload. MPS loads in float16 on device. CPU loads in float32.

The contour helper used to threshold masks at 128. Masks that reach it are 0/1, so that test erased the subject and produced a blank contour for FLUX, SD1.5, and SDXL. It now keeps every pixel greater than zero, then scales to 255. A new comparison has to rerun FLUX and OmniGen2 with this helper.

## Comparison config

`configs/benchmark_sdxl.yaml` is experiment `sdxl-r1`:

- Frozen 20-subject `benchmarks/evalset_v2.json`.
- At most 3 attempts per subject.
- FLUX, OmniGen2, and SDXL, each judged by hosted `qwen/qwen3-vl-8b-instruct`.
- Final drawings also go to the shared `qwen/qwen3-vl-235b-a22b-instruct` referee.
- Outputs go to `outputs/benchmarks/sdxl-r1`, not the historical `r2` directory.

Select the file with `WILDTRACE_BENCHMARK_CONFIG`. That file is what `scripts/benchmark_models.py` and `scripts/benchmark_setups.py` load, and its bytes are part of the provenance hash. `pipeline --resume` is forwarded to the worker process.

## Run

Install `diffusers`, `accelerate`, `gguf`, and `open-clip-torch`. Restore the 20 crops, masks, and isolated images named in `benchmarks/evalset_v2.json`. Download the three checkpoints above. Export `OPENROUTER_API_KEY`. OmniGen2 still needs `scripts/install_omnigen2.sh` when that setup runs.

```powershell
$env:WILDTRACE_BENCHMARK_CONFIG = "configs/benchmark_sdxl.yaml"
python scripts/preflight_sdxl_benchmark.py
python scripts/benchmark_setups.py pipeline --setup flux-qwen3
python scripts/benchmark_setups.py pipeline --setup omnigen2-qwen3
python scripts/benchmark_setups.py pipeline --setup sdxl-qwen3
python scripts/benchmark_setups.py evaluate
python scripts/benchmark_setups.py referee
python scripts/benchmark_setups.py report
```

Resume a setup with `--resume`. An OOM kill (`exit -9`) also restarts the worker with `--resume`.

To use SDXL outside the comparison:

```powershell
$env:OUTLINE_RECTIFIER_BACKEND = "sdxl_controlnet_rectifier"
```

Planned metrics, all still null until that run: acceptance and first-attempt pass rates, attempts, combined validation score, strong-referee pass rate, BioCLIP recognizability, silhouette IoU, drawing structure, blank-output rate, exportable-trajectory rate, ink coverage, estimated drawing time, generation latency, and memory. Robot tracking RMSE is a separate Isaac Sim execution of the exported trajectories.
