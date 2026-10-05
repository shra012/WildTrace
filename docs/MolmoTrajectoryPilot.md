# Molmo trajectory-generation pilot

## Decision

Use **Molmo 7B-D** as a candidate *stroke planner*, not MolmoAct as a direct
xArm controller. The pilot asks Molmo to output a small ordered set of
normalized 2D stroke control points. `scripts/compile_stroke_plan.py`
deterministically validates and resamples them into the JSON trajectory format
already consumed by Isaac Sim.

MolmoAct is not the first comparison model because its native output is a
robot-specific, closed-loop action chunk. Its released training data and zero-
shot guidance target Franka/DROID-style manipulation rather than 2D pen paths.
It would require xArm observations, action normalization, demonstrations, and
task-specific fine-tuning before it could be evaluated fairly.

## Model output contract

The model must return JSON only:

```json
{
  "drawing_id": "frog_molmo_v0",
  "strokes": [
    {"points": [[0.12, 0.34], [0.18, 0.29], [0.24, 0.27]]}
  ]
}
```

- Coordinates are normalized to `[0, 1]`.
- `(0, 0)` is the top-left of the input canvas; y increases downward.
- A separate stroke means lift the pen. Do not add pen-up transfers as points.
- Use at most 64 sparse control points per stroke. The compiler creates dense
  waypoints. Directly generating thousands of waypoints is intentionally not
  supported because sequence length and numeric formatting make it fragile.

## Pilot protocol

1. Freeze a held-out set of input images and their Flux-derived trajectories.
2. Start with the **same final line-art image** as input to Molmo. This isolates
   trajectory planning from image generation quality.
3. Compile each Molmo plan, then run:

   ```bat
   python scripts\compare_trajectory_candidates.py --reference <flux.json> --candidate <molmo.json> --output outputs\comparisons\frog_molmo.json
   ```

4. Run surviving candidates through Isaac Sim with
   `scripts/run_two_stroke_drawing.py` under 9.81 m/s².
5. Compare: stroke count, symmetric raster distance, pixel IoU, simulation
   completion, mean/p95 pen tracking error, contact rate, and draw duration.
6. Only after a zero-shot pilot, create a supervised LoRA data set from
   image + human-corrected stroke-plan pairs. Do not use only the existing Flux
   outputs as labels: that would train Molmo to copy the baseline's errors.

## First commands

Run Molmo in a separate GPU Python environment, not Isaac Sim's bundled Python:

```bat
python scripts\generate_molmo_stroke_plan.py --image isaac-sim\inputs\diagrams\Frog\3f221bad695edcf8.png --output isaac-sim\inputs\trajectories\Molmo\frog.json --raw-output outputs\molmo\frog_raw.txt
```

Then compare its compiled candidate against the existing trajectory:

```bat
python scripts\compare_trajectory_candidates.py --reference isaac-sim\inputs\trajectories\Frog\3f221bad695edcf8.json --candidate isaac-sim\inputs\trajectories\Molmo\frog.json --output outputs\comparisons\frog_molmo.json
```

Model inference and fine-tuning are deliberately separate from Isaac Sim: Molmo
weights are large and should run in a dedicated GPU environment.
