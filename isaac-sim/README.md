# Isaac Sim xArm 7 two-stroke drawing baseline

This project implements and has executed a simulation-only, deterministic inverse-kinematics baseline for a UFACTORY xArm 7 in NVIDIA Isaac Sim 6.0.1. It draws exactly two distinct strokes and explicitly approaches, lowers, draws, lifts, transfers above the paper, lowers, draws, and lifts again. It contains no xArm network client, robot IP setting, voltage/torque command, ROS control plugin, or physical-robot path.

The current default is the purpose-designed `data/puppy_two_stroke.json`: one closed puppy silhouette stroke followed by one continuous facial-detail stroke. The earlier cat-derived sample and synthetic fallback remain available but are not used by the current configuration. Detailed provenance and inspection results are in `DISCOVERY_REPORT.md`.

## Gravity-capable simulation controller

The default scene now runs at Earth gravity (`9.81 m/s^2`). A force-mode PhysX position servo models
the xArm's low-level joint controller beneath Lula IK. Drive settings are authored before physics
initialization, the articulation uses additional solver iterations, and joint targets are slew-limited
from the previous command rather than from the gravity-displaced measured pose.

The complete two-stroke puppy run was validated headlessly at 1 g. It completed both pen-down cycles,
the lift/transfer motion, and the final lift with 0.88 mm mean pen-path tracking error and 0.91 mm
RMSE. GUI mode is enabled by omitting `--headless`. These simulation drive-force ceilings are not
physical xArm torque limits; hardware deployment still requires the vendor controller's payload,
speed, force, and safety configuration.

## Windows Command Prompt quick start

The supplied `python.bat` attempts to copy its Python executable to `kit.exe` when `PYTHONEXE` is unset. Because the Isaac installation is intentionally read-only here, set `PYTHONEXE` first. These commands are for **Command Prompt**, not PowerShell:

```bat
cd /d C:\Users\Omni-User\Desktop\DATA298B\WildTrace\isaac-sim
set "ISAAC_ROOT=C:\Isaac-Sim"
set "PYTHONEXE=%ISAAC_ROOT%\kit\python\python.exe"

call "%ISAAC_ROOT%\python.bat" scripts\generate_xarm7_urdf.py
call "%ISAAC_ROOT%\python.bat" scripts\validate_trajectory.py --config config\xarm7_drawing.yaml
call "%ISAAC_ROOT%\python.bat" -m unittest discover -s tests -v
call "%ISAAC_ROOT%\python.bat" scripts\check_environment.py --config config\xarm7_drawing.yaml --headless
call "%ISAAC_ROOT%\python.bat" scripts\inspect_robot.py --config config\xarm7_drawing.yaml --headless
call "%ISAAC_ROOT%\python.bat" scripts\run_two_stroke_drawing.py --config config\xarm7_drawing.yaml --headless
```

To watch the run, omit `--headless` from the last command:

```bat
call "%ISAAC_ROOT%\python.bat" scripts\run_two_stroke_drawing.py --config config\xarm7_drawing.yaml
```

To record a complete deterministic demonstration:

```bat
call "%ISAAC_ROOT%\python.bat" scripts\record_demonstrations.py --config config\xarm7_drawing.yaml --headless
```

A failed or partial run is never saved as a demonstration.

## Configuration and data

`config/xarm7_drawing.yaml` controls the robot paths, safe home, drive gains, paper rectangle, target orientation, `pen_up_z_m`, `pen_down_z_m`, `approach_height_m`, tolerance, 5 mm maximum Cartesian step, workspace bounds, and timeouts. All internal geometry is SI metres/radians.

The canonical trajectory shape is:

```python
{
    "drawing_id": str,
    "strokes": [
        {"stroke_id": int, "points": [[x, y], ...]}
    ],
}
```

The loader supports JSON, CSV, NPY, NPZ, pickle, and optional Parquet. It preserves point and stroke order, rejects non-finite/empty data, removes consecutive duplicate points, flips image Y when configured, preserves aspect ratio, performs corner-aware smoothing, and resamples without exceeding the configured step.

## State machine and safety

The run advances from a waypoint only when measured pen-tip error is within tolerance. A timeout causes a safe failure; no fixed frame-count waypoint advancement is used.

```text
HOME
APPROACH_STROKE_1
LOWER_PEN_STROKE_1
DRAW_STROKE_1
LIFT_PEN_STROKE_1
MOVE_TO_STROKE_2
LOWER_PEN_STROKE_2
DRAW_STROKE_2
LIFT_PEN_STROKE_2
FINISHED
```

Targets are checked against Cartesian workspace limits before execution. IK results are rejected on failure, non-finite output, or joint-limit violation. Commands are simulation articulation position targets only.

## Outputs from the verified run

- `outputs/desired_path.csv`
- `outputs/executed_path.csv`
- `outputs/two_stroke_comparison.png`
- `outputs/run_metrics.json`
- `outputs/environment_check.json`
- `outputs/demonstration_two_stroke.npz`

The NPZ includes drawing/stroke/waypoint identity, simulation time, joint positions and velocities, pen-tip position and rotation, target and five-point lookahead, pen state, bounded Cartesian delta action, error, state, feature names, units, and normalization metadata.

## Behavioral cloning

The policy is prepared but should not be treated as useful from the single demonstration now present. It is an MLP with `[256, 256, 128]` ReLU layers and a bounded 3D Cartesian-delta output. Training uses normalized features, complete-drawing-ID train/validation splitting, MSE, deterministic seeds, checkpointing, early stopping, CPU/CUDA selection, and TensorBoard.

Record at least two complete drawing IDs before training. The training script deliberately refuses a one-ID dataset:

```bat
call "%ISAAC_ROOT%\python.bat" -m pip install --target .vendor -r requirements-bc.txt
call "%ISAAC_ROOT%\python.bat" scripts\train_behavior_cloning.py --config config\xarm7_drawing.yaml --device auto
call "%ISAAC_ROOT%\python.bat" scripts\evaluate_policy.py --config config\xarm7_drawing.yaml --demo-glob "outputs\demonstration_*.npz" --device cpu
```

Evaluation is offline imitation evaluation only; it does not claim closed-loop Isaac or physical-robot performance.

## V0 trajectory-conditioned PPO baseline

`scripts/train_v0_trajectory_ppo.py` trains a seven-action residual joint-velocity policy while the
validated 1-g IK and position controller remain underneath it. The observation contains
joint state, current target error, five lookahead targets, and pen state. This V0 intentionally uses
reference targets; it is the measurable control baseline before the image-only policy.

The training curriculum uses target prefixes of 100, 250, 500, and then the complete path. A stage
advances only after deterministic evaluation completes it. The tolerance tightens from 3 mm to 2 mm.
Each checkpoint contains policy and optimizer state, global update number, curriculum stage, and the
best evaluation score. `v0_trajectory_ppo_1g.pt` is the latest checkpoint and
`v0_trajectory_ppo_best_1g.pt` is the best deterministic checkpoint. The trainer rejects a resume
when checkpoint gravity metadata differs from the current task.

From Command Prompt in this directory, start a fresh 1-g checkpoint with a short test before
committing to the configured 300 updates:

```bat
set "ISAAC_ROOT=C:\Isaac-Sim"
set "PYTHONEXE=%ISAAC_ROOT%\kit\python\python.exe"
set "PYTHONPATH="
call "%ISAAC_ROOT%\python.bat" scripts\train_v0_trajectory_ppo.py --config config\xarm7_drawing.yaml --headless --device cpu --updates 10
call "%ISAAC_ROOT%\python.bat" scripts\train_v0_trajectory_ppo.py --config config\xarm7_drawing.yaml --headless --device cpu --resume
```

Do not reuse the older zero-gravity checkpoint: gravity changes the transition dynamics and therefore
the RL task. Every new 1-g checkpoint supports a complete resume. To verify the camera, lighting,
robot, and paper before waiting for a complete rollout, run this short GUI-only scene check:

```bat
call "%ISAAC_ROOT%\python.bat" scripts\train_v0_trajectory_ppo.py --config config\xarm7_drawing.yaml --scene-check --hold-seconds 30
```

Then view and compare the learned residual against IK with zero RL action (again, omit
`--headless`):

```bat
call "%ISAAC_ROOT%\python.bat" scripts\train_v0_trajectory_ppo.py --config config\xarm7_drawing.yaml --device cpu --evaluate --compare-zero-action --checkpoint outputs\v0_trajectory_ppo_best_1g.pt --hold-seconds 30
```

Evaluation prints episode return, target completion, mean and 95th-percentile tracking error, pen
contact rate, action magnitude, and action smoothness. It does not update the policy.

## Goal-conditioned PPO drawing prototype

The branch `feature/rl-gravity-sim2real` also contains a first reinforcement-learning prototype in
`scripts/train_rl_drawing.py`. It uses the configured gravity, keeps the existing Cartesian IK/safety layer, and
trains PPO to output a bounded Cartesian delta plus a pen command. The policy observation contains a
raster of the target drawing, a raster of the simulated drawing so far, pen-tip position/velocity,
and pen state. Waypoints and waypoint indices are not passed to the policy; the input trajectory is
used only to construct the target raster for the episode.

This is a single-environment baseline intended for debugging, not yet a production sim-to-real policy.
Start with a small training run by temporarily setting `rl.ppo_updates` to `10` in the YAML. Run it
with the Isaac Sim Python, after the xArm USD has been generated and environment-checked:

```bat
set "ISAAC_ROOT=C:\Isaac-Sim"
set "PYTHONEXE=%ISAAC_ROOT%\kit\python\python.exe"
call "%ISAAC_ROOT%\python.bat" scripts\generate_xarm7_urdf.py
call "%ISAAC_ROOT%\python.bat" scripts\check_environment.py --config config\xarm7_drawing.yaml --headless
call "%ISAAC_ROOT%\python.bat" scripts\train_rl_drawing.py --config config\xarm7_drawing.yaml --headless --device cpu
```

The checkpoint is written to `outputs/rl_drawing_policy_1g.pt`. Only after the short run starts and the
robot remains stable under gravity should `rl.ppo_updates` be increased. The prototype currently uses
the target raster generated from the configured drawing input; camera rendering and parallel Isaac
environments are the next steps before real-robot transfer.

## Test result

The launcher-only unit run executes 51 tests: 47 pass and four skip (three behavior-cloning tests
need a fully initialized PyTorch extension, and one optional prior sample is absent). The suite now
also verifies that reusable IK actions are not mutated and that the command-rate limiter continues
toward its target while measured joints are displaced by load.
