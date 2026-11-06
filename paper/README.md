# WildTrace paper

IEEE conference format (`IEEEtran`). Sources: `main.tex`, `refs.bib`, `figures/`.

Rebuild the figures (reads `isaac-sim/results/` and the local executed paths in `outputs/act_eval/`; also
prints the numbers used in the tables):

```bash
~/anaconda3/bin/python3 paper/scripts/make_figures.py
```

Compile with any TeX Live that has `IEEEtran`, or upload `main.tex`, `refs.bib`, `figures/` and `tables/` to Overleaf:

```bash
cd paper && latexmk -pdf main.tex
```

The November rerun writes robot outputs to `outputs/paper_eval/` and the generator
study to `outputs/benchmarks/paper-r3/`. Refresh the robot summary with:

```bash
~/anaconda3/bin/python3 paper/scripts/robot_summary.py
```

Each robot batch reports `expected`, `complete`, and `pending`; `finished` counts
successful drawings. ACT seed spread uses completed batches and their shared
successful drawings. Interim results are not a completed three-seed estimate.

All seven robot batches finished on November 3. Metrics, comparison images and
a row per run are archived in `isaac-sim/results/paper_eval/20261103/`.
The original queue is `outputs/paper_eval/run_robot_track_v5.sh`; do not rerun the
completed evaluation. Resume an interrupted generator queue with:

```bash
nohup python3 -u scripts/run_paper_generator_track.py > outputs/paper_eval/logs/gen_supervisor.log 2>&1 &
```

The generator supervisor preserves completed subjects and retries failed API
rows or missing images. It probes the hosted judge every five minutes during
an outage before loading generators, preserves the 18 GiB RAM gate, then runs
evaluation, resumable judging and reporting. Once all subjects and verdicts are
complete it refreshes Table I and the discussion values and rebuilds the paper.
Completion requires an 8-page PDF and writes `outputs/paper_eval/GENERATOR_COMPLETE.json`.
Follow `outputs/paper_eval/logs/gen_supervisor.log` for progress.

Regenerate the generator table and discussion macros directly with:

```bash
python3 paper/scripts/make_generator_table.py
```

This refuses incomplete studies. `--allow-incomplete` creates an explicitly
interim table with actual denominators; the current draft uses that while the
hosted 8B judge is rate limited. Old r2 generator claims are excluded. Figure 3
contains only the twin and animal drawing; the unknown-consent portrait was
removed pending a physical drawing of one of the licensed paper faces.

Face fidelity is a separate local report, without changing Studio or physical-arm inputs:

```bash
uv run --extra app python scripts/face_fidelity_report.py
```

It needs a local Ollama with `qwen3-vl:8b-instruct`. A separate local server can be selected
with `--ollama-host http://127.0.0.1:11435`. Hosted judging is restricted to the
fixed Open Images test faces with `--backend openrouter`. Results include JSON
checklists, comparison sheets, stroke counts and estimated drawing times under
`outputs/face_fidelity/paper_faces/`. Uncertain glasses verdicts keep glasses off.
All eight local comparisons completed. The report flags two baseline drawings
where the judge claimed glasses despite zero frame strokes, so the checker
remains diagnostic. It is not connected to a dataset quality gate or Studio.

## CCWC 2027 review version

`main.tex`, `refs.bib` and `figures/` hold the anonymised 7-page IEEE CCWC 2027 review copy (authors shown as
Author 1 to Author 5): supervisor edits, paired real-arm ACT vs IK results from issue #52 and the three-row
Fig. 2 with the xArm7 sketches. `WildTrace_CCWC_submission.pdf` is the file uploaded to EDAS. Camera-ready is due
Dec 20: add real names, affiliations and the copyright number, then run IEEE PDF eXpress.

The previous longer draft (noise-injection ablation, tuned-IK sweep, "what a viewer sees", ethics paragraph) is
kept on the `paper-backup-before-ccwc` branch.
