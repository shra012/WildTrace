# Trained ACT policies

LeRobot `pretrained_model` folders (weights + normalization stats), tracked with Git LFS.
Run `git lfs pull` after cloning to fetch the weights.

| Folder | Training data | Use |
|---|---|---|
| `act_kinematic/` | `kinematic_v3` labels (8,488 episodes), 40k steps | Results in #39: 48/48 gold drawings at 1 g, 0.325 mm path RMSE |
| `act_kinematic_v2/` | `kinematic_v4` labels with the 5 mm projection cap (hairpin fix), 40k steps | 48/48 gold drawings at 1 g, 0.239 mm path RMSE, but rounds spike tips by up to 2.6 mm |
| `act_kinematic_v4/` | `kinematic_v6` labels with the 1.5 mm projection cap (spike-tip fix), 40k steps | **Recommended (sim geometry).** 48/48, 0.256 mm path RMSE, worst point 0.89 mm mean (IK 0.99), none over 1.5 mm |

Serve one to the Isaac runner (see the ACT section of `isaac-sim/README.md`):

```bash
isaac-sim/.venv-act/bin/python isaac-sim/scripts/act_policy_server.py --checkpoint isaac-sim/models/act_kinematic_v4
python3 isaac-sim/scripts/isaac_batch.py eval --controller act --tag act_kin
```

Evaluation metrics behind these numbers are in `isaac-sim/results/act_eval/` (per-drawing `run_metrics.json`,
batch logs and the comparison reports). Datasets and per-run plots stay local; regenerate them with
`scripts/label_kinematic_ik.py`, `scripts/build_lerobot_dataset.py` and `scripts/isaac_batch.py`.

These models are trained for the default sim geometry (`config/xarm7_drawing.yaml`: paper at z = 200 mm, 120 mm stylus).
They do not transfer to the lab arm (`config/xarm7_drawing_real.yaml`); see #39 for the sim twin and retraining.
