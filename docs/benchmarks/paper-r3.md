# Setup comparison `paper-r3`
git `d9d1b53-dirty`, config hash `d64ef6cd544d`. Metric definitions: `docs/benchmarks/eval_suite_r2.docx`. Values are means or rates over final drawings, with bootstrap 95% CI.
Excluded categories (no source masks): Butterfly.

## Results
Final drawing per completed subject. Missing outputs from API failures are excluded from image metrics; see subject and final-image counts below. Mean or rate, with bootstrap 95% CI in brackets. **Bold** = better and CIs don't overlap; "(CIs overlap)" = direction only.

| Metric | Better | informative-qwen3 | sd15cn-qwen3 | flux-qwen3 | omnigen2-qwen3 | Better setup |
|---|---|---|---|---|---|---|
| **A. Pipeline outcome** |  |  |  |  |  |  |
| Acceptance rate | ↑ | 0% (0%–0%) | 15% (0%–35%) | 95% (85%–100%) | 30% (10%–50%) |  |
| First-attempt pass rate | ↑ | 0% (0%–0%) | 0% (0%–0%) | 45% (25%–65%) | 5% (0%–15%) |  |
| Attempts used | ↓ | 3.00 (3.00–3.00) | 2.90 (2.75–3.00) | 1.60 (1.35–1.85) | 2.80 (2.55–3.00) |  |
| Combined score | ↑ | 0.00 (0.00–0.00) | 0.14 (0.04–0.28) | 0.79 (0.69–0.86) | 0.44 (0.31–0.58) |  |
| Error rate | ↓ | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) |  |
| **F. Judges** |  |  |  |  |  |  |
| Own judge model |  | `qwen/qwen3-vl-8b-instruct` | `qwen/qwen3-vl-8b-instruct` | `qwen/qwen3-vl-8b-instruct` | `qwen/qwen3-vl-8b-instruct` |  |
| Own-judge pass rate | ↑ | – | 100% (100%–100%) | 95% (85%–100%) | 67% (33%–100%) |  |
| Judge parse-failure rate | ↓ | – | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) |  |
| Own-judge latency p50 (s) | ↓ | – | 2.46 | 1.06 | 2.19 |  |
| Referee pass: `qwen2.5-vl-7b` | ↑ | 100% (100%–100%) | 75% (55%–95%) | 100% (100%–100%) | 100% (100%–100%) |  |
| Referee pass: `qwen3-vl-8b` | ↑ | 10% (0%–25%) | 20% (5%–40%) | 100% (100%–100%) | 45% (25%–65%) |  |
| Referee pass: `qwen3-vl-30b-a3b` | ↑ | 0% (0%–0%) | 20% (5%–40%) | 100% (100%–100%) | 35% (15%–55%) |  |
| Referee pass: `gemma-3-12b` | ↑ | 20% (5%–35%) | 40% (20%–60%) | 100% (100%–100%) | 55% (35%–75%) |  |
| Referee pass: `qwen3-vl-235b` (strong) | ↑ | 0% (0%–0%) | 10% (0%–25%) | 80% (60%–95%) | 40% (20%–60%) |  |
| Own judge vs strong referee (Cohen's κ) | ↑ | 0.76 | 0.76 | 0.76 | 0.76 |  |
| **B. Drawing structure** |  |  |  |  |  |  |
| OpenCV prescreen pass | ↑ | 0% (0%–0%) | 15% (0%–35%) | 95% (85%–100%) | 30% (10%–50%) |  |
| OpenCV score | ↑ | 0.00 (0.00–0.00) | 0.15 (0.06–0.28) | 0.69 (0.58–0.78) | 0.61 (0.48–0.70) |  |
| Ink ratio | ◎ 0.16 | 0.03 (0.02–0.03) | 0.29 (0.20–0.39) | 0.02 (0.02–0.03) | 0.08 (0.03–0.15) |  |
| Connected components | ↓ | 822 (623–1035) | 569 (126–1220) | 7.00 (2.45–14.8) | 18.5 (2.25–49.6) |  |
| Small-contour ratio | ↓ | 0.93 (0.90–0.95) | 0.50 (0.37–0.63) | 0.22 (0.11–0.33) | 0.10 (0.02–0.22) |  |
| Bottom reach | ↑ | 0.95 (0.94–0.97) | 0.98 (0.96–1.00) | 0.88 (0.84–0.92) | 0.51 (0.32–0.71) |  |
| Solid-fill ratio | ↓ | 0.01 (0.00–0.02) | 0.60 (0.44–0.75) | 0.04 (0.01–0.06) | 0.41 (0.22–0.61) |  |
| Mean stroke width (px) |  | 1.53 (1.44–1.62) | 11.6 (6.69–17.4) | 2.75 (2.31–3.19) | 7.55 (3.20–12.7) |  |
| Blank output | ↓ | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 45% (25%–65%) |  |
| **C. Shape fidelity to the source** |  |  |  |  |  |  |
| Silhouette IoU | ↑ | 0.09 (0.07–0.12) | 0.64 (0.49–0.77) | 0.56 (0.40–0.73) | 0.22 (0.10–0.36) |  |
| Chamfer distance (% of diagonal) | ↓ | 1.65 (1.27–2.05) | 18.9 (5.06–34.9) | 4.36 (2.51–6.14) | 51.7 (32.1–71.3) |  |
| Boundary F-score @2% | ↑ | 0.71 (0.65–0.78) | 0.52 (0.34–0.70) | 0.56 (0.38–0.75) | 0.25 (0.13–0.38) |  |
| Aspect-ratio error | ↓ | 0.01 (0.00–0.01) | 0.04 (0.01–0.07) | 0.14 (0.06–0.23) | 0.08 (0.04–0.12) |  |
| Centroid offset (% of diagonal) | ↓ | 6.07 (4.67–7.53) | 3.86 (1.70–6.13) | 2.78 (1.55–4.04) | 4.66 (2.84–6.80) |  |
| **D. Pose / view consistency** |  |  |  |  |  |  |
| Angle-bucket match | ↑ | 0% (0%–0%) | 15% (0%–30%) | 10% (0%–25%) | 10% (0%–25%) |  |
| Facing-direction match | ↑ | 50% (30%–75%) | 55% (35%–75%) | 55% (35%–75%) | 30% (10%–50%) |  |
| |Δ direction score| | ↓ | 0.81 (0.53–1.10) | 0.61 (0.39–0.83) | 0.61 (0.44–0.78) | 0.64 (0.43–0.84) |  |
| |Δ symmetry| | ↓ | 0.32 (0.22–0.43) | 0.36 (0.23–0.49) | 0.22 (0.14–0.29) | 0.29 (0.18–0.41) |  |
| **E. Semantic recognizability** |  |  |  |  |  |  |
| BioCLIP category top-1 | ↑ | 25% (5%–45%) | 25% (10%–45%) | 55% (35%–75%) | 25% (10%–45%) |  |
| BioCLIP p(true category) | ↑ | 0.22 (0.09–0.36) | 0.25 (0.11–0.41) | 0.51 (0.33–0.68) | 0.22 (0.10–0.35) |  |
| Drawing-photo cosine | ↑ | 0.29 (0.25–0.33) | 0.27 (0.23–0.30) | 0.27 (0.23–0.31) | 0.23 (0.19–0.28) |  |
| **G. Robot drawability** |  |  |  |  |  |  |
| Trajectory strokes | ◎ 1–6 | 6.00 (6.00–6.00) | 5.45 (4.95–5.85) | 2.85 (1.95–3.75) | 2.65 (1.55–3.80) |  |
| Trajectory exportable | ↑ | 100% (100%–100%) | 100% (100%–100%) | 100% (100%–100%) | 55% (35%–75%) |  |
| Ink coverage by trajectory | ↑ | 0.18 (0.16–0.21) | 0.23 (0.17–0.30) | 0.76 (0.68–0.85) | 0.24 (0.12–0.37) |  |
| Pen-down path (mm) | ↓ | 452 (356–555) | 977 (798–1190) | 647 (550–765) | 483 (281–697) |  |
| Pen-up travel (mm) | ↓ | 373 (332–412) | 235 (188–283) | 71.8 (34.8–115) | 117 (58.7–180) |  |
| Estimated draw time (s) | ↓ | 418 (418–418) | 373 (331–406) | 156 (81.2–231) | 320 (229–388) |  |
| Within 10-min budget | ↑ | 100% (100%–100%) | 100% (100%–100%) | 100% (100%–100%) | 55% (35%–75%) |  |
| **H. Cost and resources** |  |  |  |  |  |  |
| Time per subject (s) | ↓ | 13.2 (12.5–14.0) | 19.9 (15.8–27.1) | 49.6 (40.8–58.9) | 229 (207–247) |  |
| Generation p50 per attempt (s) | ↓ | 1.6 | 2.9 | 25.1 | 75.3 |  |
| Recorded pipeline wall time, incl. recovery (min) | ↓ | 4.30 | 6.80 | 17.5 | 75.7 |  |
| Peak VRAM, generation process (GB) | ↓ | 0.00 | 3.58 | 11.3 | 8.98 |  |
| Peak system RAM (GB) | ↓ | 13.3 | 7.85 | 24.1 | 22.7 |  |
| Peak swap (GB, of ~6.3) | ↓ | 0.00 | 0.00 | 6.09 | 4.02 |  |
| OOM kills | ↓ | 0 | 0 | 0 | 0 |  |
| API cost (USD) | ↓ | $0 | $0.0007 | $0.0035 | $0.0016 |  |

| setup | subjects | final_images |
|---|---|---|
| informative-qwen3 | 20 | 20 |
| sd15cn-qwen3 | 20 | 20 |
| flux-qwen3 | 20 | 20 |
| omnigen2-qwen3 | 20 | 20 |

**Per category** (A = informative-qwen3, B = sd15cn-qwen3)

| Category | n | Accepted A / B | Strong-referee pass A / B | BioCLIP top-1 A / B | Silhouette IoU A / B |
|---|---|---|---|---|---|
| Bird | 3 | 0% / 0% | 0% / 0% | 33% / 33% | 0.10 / 0.48 |
| Cat | 4 | 0% / 0% | 0% / 0% | 0% / 0% | 0.08 / 0.63 |
| Dog | 4 | 0% / 0% | 0% / 0% | 100% / 50% | 0.07 / 0.42 |
| Fish | 3 | 0% / 33% | 0% / 33% | 0% / 0% | 0.11 / 0.76 |
| Frog | 3 | 0% / 67% | 0% / 33% | 0% / 67% | 0.13 / 0.65 |
| Horse | 3 | 0% / 0% | 0% / 0% | 0% / 0% | 0.10 / 0.95 |

**Head-to-head per subject** (which setup's final drawing was better on each subject)

| Metric | informative-qwen3 wins | sd15cn-qwen3 wins | Ties | Sign-test p |
|---|---|---|---|---|
| Strong-referee score | 1 | 9 | 10 | 0.0215 |
| Silhouette IoU | 1 | 19 | 0 | 0.0 |
| Trajectory ink coverage | 8 | 12 | 0 | 0.5034 |
| BioCLIP p(true category) | 10 | 10 | 0 | 1.0 |
| Solid-fill ratio (lower wins) | 20 | 0 | 0 | 0.0 |
| Chamfer distance (lower wins) | 13 | 7 | 0 | 0.2632 |

Contact sheet: `outputs/benchmarks/paper-r3/contact_sheet.png`.
