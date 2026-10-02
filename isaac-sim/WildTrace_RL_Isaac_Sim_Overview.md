# WildTrace Robotic Drawing: Isaac Sim and PPO Overview

## 1. Project goal

The goal is to make the xArm7 draw images in simulation and eventually transfer the learned behavior
to the physical robot. The intended final architecture is:

```text
Target image or high-level drawing goal
                 |
                 v
       Reinforcement-learning policy
    (high-level Cartesian drawing decisions)
                 |
                 v
        IK, joint limits, workspace limits,
        rate limits, and safety controller
                 |
                 v
             xArm7 robot
```

RL should not bypass inverse kinematics or the robot safety layer. The policy should decide how to
draw, while deterministic control remains responsible for converting safe Cartesian or residual
commands into robot motion.

Development is currently on the Git branch:

```text
feature/rl-gravity-sim2real
```

## 2. What was already in the repository

The repository already had a deterministic drawing pipeline:

```text
Trajectory file
  -> coordinate mapping onto the paper
  -> approach / pen-down / draw / lift state machine
  -> Lula inverse kinematics
  -> joint position commands
  -> xArm7 in Isaac Sim
```

This deterministic controller is important because it provides:

- A known working robot and paper scene.
- Correct xArm joint names and limits.
- A tested end-effector frame and pen orientation.
- Workspace and joint-step safety limits.
- A reference against which the RL policy can be measured.

## 3. Problems found and resolved

### Isaac Sim path and Python environment

Isaac Sim is installed at:

```text
C:\Isaac-Sim
```

Commands must use Isaac Sim's launcher rather than an unrelated system Python:

```bat
set "ISAAC_ROOT=C:\Isaac-Sim"
set "PYTHONPATH="
call "%ISAAC_ROOT%\python.bat" ...
```

Importing PyTorch before Isaac Sim initializes can produce `ModuleNotFoundError`. The training script
starts `SimulationApp` first, allowing Isaac to configure its packaged Python dependencies.

### Robot drawing in the air

The paper position, pen-down height, mapped coordinates, end-effector frame, IK orientation, and
position-drive settings were aligned with the repository's working deterministic trajectory drawer.
A reach test now checks that the pen can reach both an approach point and the paper before training.

The original zero-gravity reach test achieved approximately:

```text
Approach error:  1.29 mm
Pen-down error:  1.87 mm
```

### Gravity

The early acceleration-drive setup failed because the imported URDF drive limits saturated and the
joint-target slew limiter was incorrectly centered on the measured, gravity-displaced position. That
made the commanded setpoint follow the falling arm instead of restoring the intended IK pose.

The repaired controller now uses:

```yaml
safety:
  gravity_m_s2: 9.81

robot:
  drive_type: force
  drive_stiffness: 75000.0
  drive_damping: 650.0
  solver_position_iterations: 16
  solver_velocity_iterations: 4
```

The full deterministic two-stroke drawing now completes at Earth gravity. Its measured pen-path error
was 0.88 mm mean and 0.91 mm RMSE. This validates the simulation control layer; it does not mean the
simulation force limits are calibrated physical motor torques. Real-hardware deployment must retain
the xArm vendor controller and validate payload, speed, force, and emergency-stop settings.

## 4. The V0 PPO baseline

The V0 trainer is:

```text
scripts/train_v0_trajectory_ppo.py
```

V0 is intentionally trajectory-conditioned. It proves that PPO, the robot controller, contact
tracking, checkpointing, and evaluation work before removing waypoints in the later image-only policy.

### Observation: 33 values

| Observation component | Values |
|---|---:|
| Seven joint positions | 7 |
| Seven joint velocities | 7 |
| Current target-position error | 3 |
| Five lookahead target positions | 15 |
| Pen-up / pen-down state | 1 |
| Total | 33 |

### Action: 7 residual joint velocities

The neural network produces one bounded correction for each xArm joint. The final command is:

```text
safe joint command = nominal IK target + small RL residual
```

With the present configuration, the maximum residual adjustment per control decision is approximately:

```text
0.15 rad/s x 0.01667 s x 4 substeps = 0.01 rad, or about 0.57 degrees
```

IK still performs most of the motion. RL learns small corrections that may improve tracking,
smoothness, and pen contact without being allowed to command arbitrary unsafe motion.

### Reward

The reward is approximately:

```text
reward =
    10.0 * progress toward the active target
  -  0.5 * distance from the active target
  -  0.002 * action magnitude
  -  0.01 * action change / lack of smoothness
  -  pen-contact error penalty
  +  0.5 when a target is reached
  +  5.0 when the active path is completed
```

The policy is rewarded for reaching the reference path and maintaining contact, while large or
rapidly changing joint corrections are discouraged.

## 5. How PPO training works

The policy is an actor-critic neural network:

- The **actor** selects seven residual joint actions.
- The **critic** predicts the expected future reward from the current observation.

For each PPO update:

1. The policy collects 512 simulator control steps.
2. Training samples slightly random actions so that the policy explores alternatives.
3. Isaac Sim applies IK plus the sampled RL correction.
4. The environment calculates tracking, contact, and smoothness rewards.
5. Generalized Advantage Estimation determines which actions performed better or worse than expected.
6. PPO trains for four epochs with minibatches of 128 samples.
7. PPO clipping limits how much the policy may change in a single update.

Evaluation is different from training: evaluation uses the actor's deterministic mean action and does
not add exploration noise or update the network.

## 6. Initial training results

Two 300-update sessions completed successfully. Before the metrics upgrade, output values included:

```text
mean_reward: approximately 0.08 to 0.25 per control step
printed target position: approximately 0 to 301 of 957
tracking distance near active targets: commonly 1.4 to 5.4 mm
```

The old `target=N/957` value was only the target index at the exact end of a 512-step rollout. It was
not maximum progress or an episode average. Periodic `target=0` values were episode resets caused by
the alignment of 512-step rollouts with the old 800-step episode limit.

No clear improvement could be established from those old logs. This led to the training-quality
upgrade described below.

## 7. Training-quality upgrades

### Feasible episode duration

The environment had 957 targets but only 800 episode steps. Because at most one target can be accepted
per step, full completion was mathematically impossible. The limit is now:

```yaml
max_episode_steps: 3000
```

### Curriculum

Training now progresses through:

```text
Stage 1: 100 targets at 3.0 mm tolerance
Stage 2: 250 targets at 3.0 mm tolerance
Stage 3: 500 targets at 2.5 mm tolerance
Stage 4: all targets at 2.0 mm tolerance
```

`-1` in the YAML means all available targets, so the final stage adapts if the input trajectory changes.
Every 10 PPO updates, a deterministic evaluation runs. The next curriculum stage is enabled only when
the current stage succeeds.

### Useful episode metrics

The trainer now reports:

- Full-path success rate.
- Mean targets reached per episode.
- Mean episode return.
- Mean tracking error in millimeters.
- 95th-percentile tracking error in millimeters.
- Pen-contact rate during pen-down motion.
- RMS policy-action magnitude.
- RMS action change as a smoothness measurement.
- PPO policy/value losses, approximate KL divergence, and clipping fraction.

### Latest and best checkpoints

```text
outputs/v0_trajectory_ppo_1g.pt
```

This is the latest checkpoint and should be used with `--resume`.

```text
outputs/v0_trajectory_ppo_best_1g.pt
```

This is replaced only when deterministic evaluation obtains a better curriculum-aware score. It is
the preferred checkpoint for demonstrations and comparison against IK-only control.

New checkpoints contain:

- Actor and critic weights.
- Adam optimizer state.
- Global update count.
- Curriculum stage.
- Best evaluation score.
- Environment/control metadata.

The older zero-gravity checkpoint must not be resumed for 1-g training. The trainer now verifies the
saved gravity metadata and rejects mismatched dynamics. All 1-g resumes restore the complete state.

### IK-only comparison

Evaluation can compare:

```text
IK-ONLY: nominal IK with a zero RL residual
POLICY:  nominal IK plus the learned residual
```

The RL policy is useful only if it improves completion, tracking error, contact rate, or smoothness
relative to IK-only control.

## 8. Commands to train and evaluate

Run all commands from Windows Command Prompt.

### Enter the project

```bat
cd /d C:\Users\Omni-User\Desktop\DATA298B\WildTrace\isaac-sim
set "ISAAC_ROOT=C:\Isaac-Sim"
set "PYTHONPATH="
```

### Start a fresh 10-update 1-g curriculum test

```bat
set "PYTHONEXE=%ISAAC_ROOT%\kit\python\python.exe"
call "%ISAAC_ROOT%\python.bat" scripts\train_v0_trajectory_ppo.py --config config\xarm7_drawing.yaml --headless --device cpu --updates 10
```

After this fresh save, future resumes include optimizer and global-update state.

### Run the configured 300-update session

```bat
call "%ISAAC_ROOT%\python.bat" scripts\train_v0_trajectory_ppo.py --config config\xarm7_drawing.yaml --headless --device cpu --resume
```

### Watch the best model and compare it with IK-only control

Run this after an evaluation has created the best checkpoint:

```bat
call "%ISAAC_ROOT%\python.bat" scripts\train_v0_trajectory_ppo.py --config config\xarm7_drawing.yaml --device cpu --evaluate --compare-zero-action --checkpoint outputs\v0_trajectory_ppo_best_1g.pt --hold-seconds 30
```

Green debug lines show the desired reference path. Pink/magenta lines show the path actually executed
while the pen is considered in contact.

## 9. How to interpret the upgraded output

| Metric | Meaning |
|---|---|
| `global_update` | Update count preserved across new-format resumes |
| `stage` | Active curriculum stage |
| `step_reward` | Mean reward over the latest 512 simulator decisions |
| `success` | Percentage of completed evaluation/training episodes |
| `targets_mean` | Average target progress for completed episodes |
| `return_mean` | Average sum of rewards over a complete episode |
| `mean_error_mm` | Average pen-to-active-target error |
| `p95_error_mm` | Error below which 95% of measurements fall |
| `contact` | Fraction of pen-down steps satisfying the contact-height tolerance |
| `action_rms` | Typical magnitude of the RL correction |
| `smooth_rms` | Typical change between consecutive RL corrections; lower is smoother |
| `kl` | Approximate size of the PPO policy update |
| `clip_fraction` | Fraction of updates constrained by PPO clipping |

`policy_loss` is not a supervised-learning accuracy value and does not need to steadily approach zero.
Actual drawing quality should be judged using deterministic episode success, tracking error, contact,
and the IK-only comparison.

## 10. Expected training time on this computer

The two original 300-update runs took:

```text
Run 1: 1171.814 seconds = 19 minutes 31.8 seconds
Run 2: 1165.548 seconds = 19 minutes 25.5 seconds
Average:                  about 19 minutes 29 seconds
```

The upgraded trainer performs deterministic evaluation every 10 updates, so it will take longer:

| Operation | Expected wall-clock time |
|---|---:|
| Isaac Sim startup | 15-25 seconds |
| One-update smoke test, including startup | 30-45 seconds |
| 10-update curriculum test | About 1-3 minutes |
| 300 updates, old trainer | Measured at about 19.5 minutes |
| 300 updates, upgraded trainer, headless | Approximately 25-40 minutes |
| 300 updates with the GUI rendering | Often 45-90 minutes |
| Full policy plus IK-only visual evaluation | Approximately 1-3 minutes |

The upgraded 300-update estimate depends on whether deterministic evaluations complete early or use
the full 3000-step timeout. The policy network is small, so Isaac physics, Lula IK, resets, and
evaluation dominate runtime. Selecting `--device cuda` may not greatly reduce the total duration.

## 11. Current limitations

- V0 still receives trajectory waypoints; it is not trajectory-free.
- Physics runs at Earth gravity, but actuator and contact parameters are not yet calibrated from the physical xArm.
- Only one Isaac environment is collected at a time, so PPO samples are slow and correlated.
- Contact is inferred from pen height, not yet from a calibrated physical force/contact sensor.
- No mass, friction, actuator-delay, camera, or calibration domain randomization is active yet.
- The policy has not been validated on the physical xArm.

## 12. Recommended next phases

1. Complete V0 curriculum training and compare the best policy against IK-only control.
2. Require consistent full-path success, low tracking error, and stable contact before changing tasks.
3. Calibrate the validated 1-g simulation controller against measured physical-arm motion and payload behavior.
4. Add domain randomization for paper pose, joint friction, actuator delay, mass, and observation noise.
5. Move the environment into Isaac Lab with many parallel environments for faster PPO training.
6. Train the image-conditioned policy to make high-level Cartesian drawing decisions without exposing
   waypoint indices.
7. Keep IK, joint limits, velocity limits, workspace constraints, watchdogs, and emergency-stop logic
   under the learned policy during sim-to-real deployment.
8. Begin physical tests at reduced speed and force, starting above the paper before allowing contact.

## 13. Files added or modified for this work

```text
config/xarm7_drawing.yaml
scripts/run_two_stroke_drawing.py
scripts/train_rl_drawing.py
scripts/train_v0_trajectory_ppo.py
src/articulation_control.py
src/ik_controller.py
src/rl_policy.py
tests/test_path_geometry.py
README.md
```

The implementation passed Python compilation, 51 launcher-only unit tests (47 passed and four
expected skips), a 1-g reach/contact preflight, a complete two-stroke 1-g drawing, and a one-update
1-g PPO smoke test. Zero-gravity checkpoints are intentionally kept separate.
