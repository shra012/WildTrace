# WildTrace paper

IEEE conference format (`IEEEtran`). Sources: `main.tex`, `refs.bib`, `figures/`.

Rebuild the figures (reads `isaac-sim/results/` and the local executed paths in `outputs/act_eval/`; also
prints the numbers used in the tables):

```bash
~/anaconda3/bin/python3 paper/scripts/make_figures.py
```

Compile with any TeX Live that has `IEEEtran`, or upload `main.tex`, `refs.bib` and `figures/` to Overleaf:

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

The robot queue is `outputs/paper_eval/run_robot_track_v5.sh` (already running;
do not launch a duplicate). Its concurrency is controlled by
`outputs/paper_eval/CONCURRENCY`. Resume the generator queue with:

```bash
nohup python3 -u scripts/run_paper_generator_track.py > outputs/paper_eval/logs/gen_supervisor.log 2>&1 &
```

The generator supervisor prevents duplicate launches, preserves completed
setups, logs its 18 GiB RAM wait, and runs evaluation, resumable judging, and
reporting after generation. Follow `outputs/paper_eval/logs/robot_track.log` and
`outputs/paper_eval/logs/gen_supervisor.log` for progress.
