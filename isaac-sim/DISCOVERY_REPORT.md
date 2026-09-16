# Discovery report

Inspection date: 2026-09-06 (America/Los_Angeles)

## Scope and exclusions

`C:\isaacsim` was inspected recursively as a read-only source. It is primarily a complete Isaac Sim 5.1.0-rc installation, not only a prior WildTrace project. The scan excluded Git internals, caches, virtual environments, checkpoints, generated images, and simulator-distribution content when classifying project artifacts. `C:\wildtrace\_samples` does not exist; the sample directory actually present is `C:\isaacsim\wildtrace_samples`.

No file under `C:\isaacsim` or the Isaac Sim 6.0.1 installation was modified.

## Prior custom work found

- Drawing scripts: `franka_circle_rmpflow.py`, `franka_square_rmpflow.py`, `franka_triangle_rmpflow.py`, `franka_wildtrace_rmpflow.py`, and shared `franka_rmpflow_common.py`.
- Trajectories: `bird.json`, `cat.json`, `cat_full.json`, and `fish.json` under `C:\isaacsim\wildtrace_samples`.
- Existing trajectory processing: `_wildtrace_strokes_xy`, `_canvas_to_draw_plane`, `_pen_lift`, and `generate_wildtrace_trajectory` in `franka_rmpflow_common.py`. It retains file order, flips image Y, fits the data with one scale factor, resamples polylines, and inserts lifted transfers.
- The old execution path is a Franka Panda RMPflow demo for Isaac Sim 5.1. It is useful evidence for the data convention but is not assumed compatible with xArm 7, Lula IK, or Isaac Sim 6.0.1.

No project-specific SVG files, CSV/NPY/NPZ/pickle/Parquet trajectories, Gold export directory, metadata CSV, MuJoCo environment, PPO script/configuration, xArm URDF/Xacro/MJCF/USD, or project robot asset was found in the custom prior files. Files with similar names inside the bundled Isaac distribution were not treated as prior project data.

## Trajectory sample inspection

| File | Drawing ID | Strokes | Points | X range | Y range | Pen field |
|---|---:|---:|---:|---:|---:|---|
| `bird.json` | `demo_bird` | 1 | 16 | 0.10–0.90 | 0.26–0.72 | `down` |
| `cat.json` | `demo_cat` | 1 | 18 | 0.10–0.90 | 0.12–0.90 | `down` |
| `cat_full.json` | `demo_cat_full` | 14 | 93 | 0.04–1.00 | 0.02–0.93 | `down` |
| `fish.json` | `demo_fish` | 1 | 12 | 0.08–0.80 | 0.28–0.72 | `down` |

Findings:

- Coordinates are dimensionless `normalized_canvas` values, derived by width/height normalization of a 256×256 canvas.
- X increases to image right; Y increases downward. Mapping to the horizontal Isaac drawing plane therefore flips Y.
- Point order is the JSON list order and must be preserved.
- Drawing identity is `sample_id`; this implementation standardizes it to `drawing_id`.
- Strokes are separate entries in the ordered `strokes` list and have integer `stroke_id` values.
- Every inspected stroke has `pen_state: "down"`. Pen-up motion is implicit between stroke entries; there are no point-level pen-up samples.
- `cat_full.json` is the only inspected sample with two or more strokes. Its first two strokes contain 49 and 7 source points and were copied into `data/sample_two_stroke.json`. The original remains untouched.
- The configured mapper fits both selected strokes into a 0.16 m × 0.12 m rectangle with a uniform scale of 0.1318681319, flips image Y, preserves aspect ratio, and resamples them to 133 and 7 points with a 5 mm maximum Cartesian segment.

The supplied synthetic fallback is included as `data/fallback_two_stroke.json`, but it was not used for the initial verified run. After the initial cat-derived test, `data/puppy_two_stroke.json` was added as a clearer exactly-two-stroke visual target and is now the configured default.

## xArm 7 description and imported articulation

No local xArm asset was found, so the description came from the official UFACTORY `xArm-Developer/xarm_ros2` repository, `humble` branch, commit `62936f7ea1846a85f7350de2c4c18f39e6d19715`. Only the description source is used; no driver, ROS control, SDK, IP address, or hardware interface is loaded.

The official description identifies `link_eef` as the fixed final flange alias attached to `link7`. The simulation-only wrapper adds `pen_body` and a separate fixed `pen_tip` frame 0.12 m from that flange. The imported Isaac data is metre-scale and Z-up.

Imported articulation report:

- Reference prim: `/World/xarm7`
- Articulation root: `/World/xarm7/Geometry/world/link_base`
- URDF root link: `world`
- Final flange: `link_eef`
- IK frame: `pen_tip`
- Gripper joints: none
- Controllable arm joints: `joint1`, `joint2`, `joint3`, `joint4`, `joint5`, `joint6`, `joint7`

| Joint | Lower (rad) | Upper (rad) | Velocity (rad/s) | URDF effort |
|---|---:|---:|---:|---:|
| joint1 | -6.283185307 | 6.283185307 | 3.14 | 50 |
| joint2 | -2.059000000 | 2.094400000 | 3.14 | 50 |
| joint3 | -6.283185307 | 6.283185307 | 3.14 | 30 |
| joint4 | -0.191980000 | 3.927000000 | 3.14 | 30 |
| joint5 | -6.283185307 | 6.283185307 | 3.14 | 30 |
| joint6 | -1.692970000 | 3.141592654 | 3.14 | 20 |
| joint7 | -6.283185307 | 6.283185307 | 3.14 | 20 |

`XS13` alone does not uniquely identify the full UFACTORY serial/model variant, so no 1300/1305-specific calibration was guessed. The official description's default offline xArm 7 mesh, inertial, and kinematic parameters are used.

## Isaac Sim 6.0.1 API verification

The supplied installation reports `6.0.1-rc.7+release.42383.32955d8d.gl`. `SimulationApp` was started before runtime imports. The installed, working APIs were verified as:

- `isaacsim.asset.importer.urdf.URDFImporter` and `URDFImporterConfig`
- `isaacsim.robot_motion.motion_generation.LulaKinematicsSolver`
- `isaacsim.robot_motion.motion_generation.ArticulationKinematicsSolver`
- `isaacsim.core.prims.SingleArticulation`

The legacy URDF command API was not used. `check_environment.py` imported and validated `assets/xarm7/xarm7.usd`; the report is in `outputs/environment_check.json`.

## Verified baseline result

The deterministic baseline was actually executed headlessly in Isaac Sim 6.0.1. Both the initial cat-derived trajectory and the later purpose-designed puppy completed all states through `FINISHED`, used exactly two strokes, and saved the requested output files. The run used configurable zero gravity for this first kinematic stage because the official URDF effort caps did not hold the extended home pose under gravity with the imported drive model. The code does not raise those effort caps. Gravity compensation/dynamics tuning remains a later step.

The current puppy run covered 8,761 simulation steps and 148.4915 simulated seconds. Waypoint reach RMSE was 0.002854 m; executed-to-desired nearest-path RMSE was 0.004040 m. See `outputs/run_metrics.json` for all metrics.
