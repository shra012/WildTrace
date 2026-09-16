# Isaac Sim xArm 7 two-stroke drawing baseline

This project implements and has executed a simulation-only, deterministic inverse-kinematics baseline for a UFACTORY xArm 7 in NVIDIA Isaac Sim 6.0.1. It draws exactly two distinct strokes and explicitly approaches, lowers, draws, lifts, transfers above the paper, lowers, draws, and lifts again. It contains no xArm network client, robot IP setting, voltage/torque command, ROS control plugin, or physical-robot path.

The current default is the purpose-designed `data/puppy_two_stroke.json`: one closed puppy silhouette stroke followed by one continuous facial-detail stroke. The earlier cat-derived sample and synthetic fallback remain available but are not used by the current configuration. Detailed provenance and inspection results are in `DISCOVERY_REPORT.md`.

## Important stage-1 limitation

`safety.gravity_m_s2` is intentionally `0.0` for this first kinematic baseline. With normal gravity, the imported official effort caps did not hold the extended configured home pose. The implementation keeps those caps rather than silently increasing them. Before enabling gravity, add and validate an appropriate gravity-compensation/dynamics controller.

The puppy run was successfully exercised headlessly. GUI mode is enabled by omitting `--headless`. Desired paths are authored as USD curves and live executed traces use Isaac debug draw; the saved puppy comparison plot was inspected.

## Windows Command Prompt quick start

The supplied `python.bat` attempts to copy its Python executable to `kit.exe` when `PYTHONEXE` is unset. Because the Isaac installation is intentionally read-only here, set `PYTHONEXE` first. These commands are for **Command Prompt**, not PowerShell:

```bat
cd /d C:\wildtrace\isaac_xarm7_drawing
set "ISAAC_ROOT=C:\Users\Aswin\isaac-sim-standalone-6.0.1-windows-x86_64"
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

## Test result

Fourteen non-Isaac unit tests pass in the available PyTorch-enabled packaged environment. Under the supplied bare Isaac 6.0.1 Python, the 11 robotics/data tests pass and the three BC tests skip unless PyTorch is installed into `.vendor`. The tests cover the existing and fallback formats, exactly-two-stroke selection, invalid/duplicate data, aspect-preserving workspace mapping, corner-preserving resampling, pen-up insertion, reached-driven and timeout transitions, robot frames/joints, metrics, policy tensor dimensions/action bounds, complete-ID split, and checkpoint save/load.
