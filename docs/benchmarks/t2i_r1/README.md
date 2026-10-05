# Text-to-image comparison `t2i-r1`: SANA vs FLUX-schnell
git `159d5f5-dirty`, config hash `84798a2c43e9`. Config: `configs/benchmark.yaml` `t2i_comparison`; script: `scripts/benchmark_t2i.py`.

Both models run on FAL (`fal-ai/sana`, `fal-ai/flux/schnell`) at 1024×1024, one image per subject, prompted with the production line-drawing prompt for each of the 20 r2 evalset subjects (`benchmarks/evalset_v2.json`), same seed per subject, production binarization. Neither model sees the source photo, so shape and pose fidelity (r2 groups C, D) are not measured. Estimated draw time is not reported: its model calibrates from Isaac Sim run logs that were not available on the machine that ran this. Values are means or rates with bootstrap 95% CI. **Bold** = better and CIs don't overlap.

## Key findings
- **Keep FLUX-schnell; don't switch to SANA as-is.** The strong judge (Qwen3-VL-235B) passed 19/20 FLUX drawings and 11/20 SANA drawings. On the same subjects, SANA never passed where FLUX failed, and FLUX passed 8 that SANA failed (paired sign test p = 0.0078). SANA failed all 3 fish and 2 of 3 frogs.
- **SANA draws too much detail for the robot.** Its drawings have fish scales, fur strokes and hatching; FLUX's are plain outlines. SANA averages 107 connected components against 16.5 for FLUX, and passes the OpenCV prescreen 10% of the time against 45%. Every one of the strong judge's 9 SANA rejections cites excessive interior detail or clutter. The exported trajectory also covers less of SANA's ink (37% vs 51%) and its pen-down path is about 40% longer (1053 mm vs 753 mm).
- **Both are equally recognizable, and neither produces blanks.** BioCLIP names the right animal for 80% of drawings from both models. Neither produced a blank page.
- **SANA is cheaper and faster.** On FAL it costs $0.001 per 1024×1024 image against $0.006 for FLUX-schnell, and returns in 2.9 s against 4.5 s.
- **Caveats.** This is text-to-image only: neither model saw the source photo, so these numbers aren't directly comparable to r2's FLUX row, which was image-to-image on the silhouette with a retry loop and a different judge. n = 20 with one seed per subject. A prompt asking for fewer interior lines might help SANA, but that wasn't tested.
- **The cheaper judge goes easy on cluttered drawings.** qwen3-vl-8b, which r2 recommended as the production judge, passed 6 SANA drawings that the strong judge rejected for clutter, and never rejected one the strong judge passed (κ = 0.35 on SANA). If SANA or another detail-heavy generator is used, the production judge would let these through. On FLUX, κ shows 0.00 only because the 8b judge passed all 20, which leaves nothing to agree or disagree on; that's not real disagreement.

## Results
| metric | better | sana | flux-schnell | winner |
|---|---|---|---|---|
| **B. Drawing structure** |  |  |  |  |
| OpenCV prescreen pass | ↑ | 10% (0%–25%) | 45% (25%–70%) | flux-schnell (CIs overlap) |
| OpenCV score | ↑ | 0.08 (0.02–0.16) | 0.33 (0.21–0.45) | **flux-schnell** |
| Ink ratio | ◎ 0.16 | 0.04 (0.04–0.05) | 0.02 (0.02–0.03) | **sana** |
| Connected components | ↓ | 107 (74–139) | 16.5 (12.7–20.2) | **flux-schnell** |
| Small-contour ratio | ↓ | 0.60 (0.49–0.70) | 0.24 (0.17–0.32) | **flux-schnell** |
| Bottom reach | ↑ | 0.85 (0.82–0.88) | 0.82 (0.80–0.85) | sana (CIs overlap) |
| Solid-fill ratio | ↓ | 0.26 (0.18–0.36) | 0.16 (0.10–0.23) | flux-schnell (CIs overlap) |
| Mean stroke width (px) |  | 3.74 (3.39–4.11) | 4.08 (3.79–4.42) |  |
| Blank output | ↓ | 0% (0%–0%) | 0% (0%–0%) | tie |
| **E. Semantic recognizability** |  |  |  |  |
| BioCLIP category top-1 | ↑ | 80% (60%–95%) | 80% (60%–95%) | tie |
| BioCLIP p(true category) | ↑ | 0.78 (0.60–0.92) | 0.77 (0.60–0.90) | sana (CIs overlap) |
| **G. Robot drawability** |  |  |  |  |
| Trajectory strokes | 1–6 exportable | 5.85 (5.65–6.00) | 5.60 (5.10–6.00) |  |
| Trajectory exportable | ↑ | 100% (100%–100%) | 100% (100%–100%) | tie |
| Ink coverage by trajectory | ↑ | 0.37 (0.32–0.42) | 0.51 (0.47–0.55) | **flux-schnell** |
| Pen-down path (mm) | ↓ | 1053 (883–1260) | 753 (666–857) | **flux-schnell** |
| Pen-up travel (mm) | ↓ | 236 (203–272) | 172 (131–215) | flux-schnell (CIs overlap) |
| **H. Cost and speed** |  |  |  |  |
| Generation time per attempt (s) | ↓ | 2.92 (2.49–3.45) | 4.49 (4.16–4.85) | **sana** |
| Generation cost per image (USD) | ↓ | $0.0010 | $0.0060 | **sana** |
| Generation time p50 / p95 (s) | ↓ | 2.49 / 4.98 | 4.28 / 6.01 |  |
| **F. Judges (OpenRouter)** |  |  |  |  |
| Judge pass: `qwen/qwen3-vl-8b-instruct` | ↑ | 85% (70%–100%) | 100% (100%–100%) | flux-schnell (CIs overlap) |
| Judge latency p50 (s): `qwen3-vl-8b` | ↓ | 1.41 | 1.36 |  |
| Judge pass: `qwen/qwen3-vl-235b-a22b-instruct` (strong) | ↑ | 55% (30%–75%) | 95% (85%–100%) | **flux-schnell** |
| Judge latency p50 (s): `qwen3-vl-235b` | ↓ | 2.16 | 2.08 |  |
| `qwen3-vl-8b` vs strong judge (Cohen's κ) | ↑ | 0.35 | 0.00 |  |

Strong judge (qwen3-vl-235b) paired sign test: sana passes where flux-schnell fails on 0 subjects, the reverse on 8; p = 0.0078.

Cost for this run: generation sana $0.0210, flux-schnell $0.1200; OpenRouter judges $0.0235. Generation errors: 0.

## Strong-judge pass by category (qwen3-vl-235b)
| category | sana | flux-schnell |
|---|---|---|
| Bird (n=3) | 2/3 | 3/3 |
| Cat (n=4) | 3/4 | 4/4 |
| Dog (n=4) | 3/4 | 3/4 |
| Fish (n=3) | 0/3 | 3/3 |
| Frog (n=3) | 1/3 | 3/3 |
| Horse (n=3) | 2/3 | 3/3 |

Contact sheet: `docs/benchmarks/t2i_r1/contact_sheet.png`.
