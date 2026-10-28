# Trained ACT and SmolVLA policies

LeRobot `pretrained_model` folders (weights + normalization stats), tracked with Git LFS.
Run `git lfs pull` after cloning to fetch the weights.

| Folder | Training data | Use |
|---|---|---|
| `act_kinematic/` | `kinematic_v3` labels (8,488 episodes), 40k steps | Results in #39: 48/48 gold drawings at 1 g, 0.325 mm path RMSE |
| `act_kinematic_v2/` | `kinematic_v4` labels with the 5 mm projection cap (hairpin fix), 40k steps | 48/48 gold drawings at 1 g, 0.239 mm path RMSE, but rounds spike tips by up to 2.6 mm |
| `act_kinematic_v4/` | `kinematic_v6` labels with the 1.5 mm projection cap (spike-tip fix), 40k steps | **Recommended (sim geometry).** 48/48, 0.256 mm path RMSE, worst point 0.89 mm mean (IK 0.99), none over 1.5 mm |
| `act_real_kinematic_v2/` | Lab-arm twin (`config/xarm7_drawing_real.yaml`), latest labeller, paper height jitter +-15 mm, gold excluded, 40k steps | **Recommended for the lab arm.** Stiffened twin (`9ba7336`), 48 gold drawings: 48/48, 0.195 mm path RMSE, jerk 390 (IK 44/48, 0.283 mm, jerk 752) |
| `smolvla_stage1/` | SmolVLA (`train_smolvla.py --state full --direct-state --vlm-layers 8 --image-size 256 --chunk-size 20`), `wildtrace_real_kin_v2` (lab-arm twin), one of each drawing's three recordings, 50k steps from the first SmolVLA run | Lab arm, for comparison with ACT (#45). First-action error 0.49 on held-out drawings (ACT 0.28). Serve with `act_policy_server.py --policy smolvla` or `run_xarm7_act_hardware.py --policy smolvla` |

Serve one to the Isaac runner (see the ACT section of `isaac-sim/README.md`):

```bash
isaac-sim/.venv-act/bin/python isaac-sim/scripts/act_policy_server.py --checkpoint isaac-sim/models/act_kinematic_v4
python3 isaac-sim/scripts/isaac_batch.py eval --controller act --tag act_kin
```

Evaluation metrics behind these numbers are in `isaac-sim/results/act_eval/` (per-drawing `run_metrics.json`,
batch logs and the comparison reports). Datasets and per-run plots stay local; regenerate them with
`scripts/label_kinematic_ik.py`, `scripts/build_lerobot_dataset.py` and `scripts/isaac_batch.py`.

These models are trained for the default sim geometry (`config/xarm7_drawing.yaml`: paper at z = 200 mm, 120 mm stylus).
They do not transfer to the lab arm (`config/xarm7_drawing_real.yaml`); use `act_real_kinematic_v2` there (see #39).
