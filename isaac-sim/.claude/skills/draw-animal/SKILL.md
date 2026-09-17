---
name: draw-animal
description: This skill should be used when the user asks to draw an animal with the xArm7 robot in Isaac Sim, e.g. "draw a dog", "I want a cat drawn", "can the arm draw a horse", or any "draw <animal>" / "sketch <animal>" request against the WildTrace isaac-sim project. Checks WildTrace's curated gold trajectories for the requested category and either draws one in Isaac Sim or reports plainly that the category isn't available -- never generates a new trajectory on the fly.
---

# Draw Animal

Turns "I want a `<animal>`" into a robot-drawn sketch, end to end: provision a
trajectory from WildTrace's existing curated gold set, drive Isaac Sim
through it, and report back tracking-accuracy metrics plus a desired-vs-
executed comparison plot.

**Scope discipline**: this only ever serves trajectories that already exist
under `outputs/gold/trajectories/<Category>/` in the WildTrace repo. It never
runs WildTrace's fetch/curate/generate/validate pipeline, never bypasses
gold's per-angle-bucket selection, and never invents a trajectory. If the
requested category has no gold trajectories, say so plainly and stop --
that's a normal, expected outcome, not an error to work around.

## Prerequisites

Isaac Sim must already be connected via the `mcp__isaac-sim__*` MCP tools,
with the scene already built (robot, physics scene, camera, Action Graph
running `scripts/mcp_drawing_controller.py`). Scene state does **not**
persist across an Isaac Sim restart -- if `mcp__isaac-sim__get_scene_info`
errors with "Could not connect" or comes back with an unexpectedly empty
stage, rebuild the scene first via this exact 4-step sequence before
continuing:

1. `mcp__isaac-sim__create_physics_scene(gravity=[0,0,0])`
2. `mcp__isaac-sim__reload_script(file_path="\\wsl.localhost\Ubuntu-24.04\home\shravan\Workspace\WildTrace\isaac-sim\scripts\setup_scene.py")`
   -- config-driven, reads `config/xarm7_drawing.yaml`
3. `mcp__isaac-sim__create_camera(prim_path="/OmniverseKit_Persp", resolution=[1024,1024])`
4. `mcp__isaac-sim__create_action_graph(graph_path="/World/DrawingGraph", script_file="\\wsl.localhost\Ubuntu-24.04\home\shravan\Workspace\WildTrace\isaac-sim\scripts\mcp_drawing_controller.py")`

Then confirm with `mcp__isaac-sim__get_robot_info(prim_path="/World/xarm7")`.

## Steps

1. **Provision the trajectory.** Run, via `wsl.exe -d Ubuntu-24.04 -- bash -lc "..."` (never a bare path argument -- it gets MSYS-mangled from Git-Bash):

   ```bash
   cd ~/Workspace/WildTrace/isaac-sim && python3 scripts/provision_and_draw.py --category <Animal>
   ```

   Pass `--allow-repeat` only if the user explicitly wants a repeat draw even
   though every sample in that category has already been drawn this project.

   `wsl.exe`'s own shell exit code is unreliable through Git-Bash -- always
   parse the script's JSON stdout and check the `status` field, never the
   shell exit code.

   - `{"status": "not_available", ...}` -> tell the user plainly that
     WildTrace doesn't have curated gold trajectories for that category yet,
     and stop here. Do not fall back to generating one.
   - `{"status": "ok", "category", "sample_id", "prefix", "stroke_count",
     "point_count", "already_drawn_before", ...}` -> continue to step 2. This
     call already wrote `outputs/mcp_sessions/job.json` with a Windows-UNC
     `trajectory_path` that the running controller reads at its next INIT.

2. **Drive the draw.** The controller clears any previous drawing at INIT and
   re-reads `job.json`, so a fresh stop/play cycle is enough to pick up the
   new job:

   - `mcp__isaac-sim__stop_simulation()`
   - `mcp__isaac-sim__play_simulation()`

3. **Wait for completion**, not a fixed delay. Poll from WSL for the report
   file this run will produce (`outputs/mcp_sessions/<prefix>_run_metrics.json`,
   where `<prefix>` is exactly what step 1 returned):

   ```bash
   until wsl.exe -d Ubuntu-24.04 -- bash -lc "test -f ~/Workspace/WildTrace/isaac-sim/outputs/mcp_sessions/<prefix>_run_metrics.json"; do sleep 3; done
   ```

   **Never call `mcp__isaac-sim__capture_image` before this file exists.**
   Its no-frame-available fallback silently pauses the running simulation
   with no auto-resume -- if that happens, the fix is to call
   `mcp__isaac-sim__play_simulation()` again, not to keep calling
   `capture_image`.

4. **Report results.** Read `<prefix>_run_metrics.json` and surface at least
   `status`, `stroke_count`, and `desired_to_executed_nearest_path_error.mean_error_m`
   (convert to mm) to the user. Then generate the comparison plot:

   ```bash
   cd ~/Workspace/WildTrace/isaac-sim && python3 scripts/plot_drawing_comparison.py \
     --trajectory ~/Workspace/WildTrace/outputs/gold/trajectories/<Category>/<sample_id>.json \
     --executed-csv outputs/mcp_sessions/<prefix>_executed_path.csv \
     --output outputs/mcp_sessions/<prefix>_comparison.png \
     --title "<Animal> (<sample_id>)"
   ```

   Use the WSL-native `~/Workspace/...` path here, not the Windows-UNC
   `trajectory_path` step 1 returned (that form is for Isaac Sim's `job.json`,
   not a bash argument) and not a path relative to the isaac-sim project root
   (the gold trajectory lives one level up, under WildTrace itself, at
   `../outputs/gold/trajectories/...` relative to `isaac-sim/`).

   Optionally call `mcp__isaac-sim__capture_image` now (report file already
   exists, so the live-frame path is used, not the pausing fallback) for an
   in-sim screenshot alongside the plot.

## Known-good reference numbers

Mean tracking error has consistently landed in the 2-3mm range across Cat,
Dog, Bird, Fish, and Horse gold trajectories with the current
`config/xarm7_drawing.yaml` tuning (`smoothing_strength: 0.25`,
`max_cartesian_step_m: 0.002`). A result far outside that band on an
otherwise-normal run is worth flagging to the user rather than reporting
silently as success.
