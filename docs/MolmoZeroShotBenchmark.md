# Molmo zero-shot trajectory benchmark

## Decision

**Do not promote MolmoE-1B-0924 to the trajectory-generation pipeline.**
The zero-shot model produced no valid canonical trajectory on the evaluated
subjects, so it cannot be compared on source-geometry or Isaac Sim tracking
quality yet. A score of zero is not assigned to those unavailable metrics.

## Protocol

- Model: `allenai/MolmoE-1B-0924`, 4-bit local inference.
- Input: Open Images masked subject crop, directly supplied to Molmo.
- Output contract: sparse normalized JSON stroke plan, then deterministic
  compilation to WildTrace canonical trajectory JSON.
- Requested set: five images in each configured category (35 total).
- Available/attempted set: 27 crops. Open Images yielded two acceptable Bird
  crops and no acceptable Butterfly crops under the current mask/quality rules.
- No Flux, SVG generation, trajectory teacher, or Isaac Sim run participated
  in this experiment.

## Results

| Model | Subjects attempted / requested | Coverage | Valid / invalid outputs | Valid rate | Mean / p50 / p95 latency (s) | Geometry vs. gold | Isaac tracking | GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MolmoE-1B zero-shot | 27 / 35 | 77.143% | 0 / 27 | 0.000% | 91.422 / 98.719 / 110.278 | Not applicable | Not applicable | Not measured |

| Category | Attempted | Valid trajectories |
| --- | ---: | ---: |
| Cat | 5 | 0 |
| Dog | 5 | 0 |
| Horse | 5 | 0 |
| Bird | 2 | 0 |
| Butterfly | 0 | 0 |
| Fish | 5 | 0 |
| Frog | 5 | 0 |

All 27 failures were either non-JSON responses or degenerate strokes (fewer
than two distinct points). The mean total inference latency was 91.422 s per
subject; this includes model generation after the subject crop was prepared.

## Fair Flux / SANA / Molmo comparison contract

Flux and SANA must run on the frozen subject list at
`outputs/model_benchmarks/shared_subjects_v0.json`. For each model, record:

1. subject coverage, valid/invalid output rate, and mean/p50/p95 latency;
2. source geometry against a shared gold trajectory: symmetric raster distance
   in pixels and pixel IoU;
3. a semantic/judge score using the same validator and threshold;
4. peak GPU memory sampled during generation; and
5. Isaac Sim desired-to-executed nearest-path RMSE and max error only for valid
   trajectories that finish the identical controller run.

The reusable table generator is
`scripts/build_vlm_benchmark_table.py`. It leaves unavailable metrics blank
instead of converting invalid generations into misleading zero-error values.

## Next action

To make Molmo a serious candidate, train or fine-tune it on a much larger set
of image-to-sparse-stroke-plan pairs, use constrained JSON decoding, and rerun
this exact frozen subject set. This 27-subject zero-shot result is sufficient to
reject the current checkpoint/prompt combination, not the general idea of
fine-tuning a VLM for the task.
