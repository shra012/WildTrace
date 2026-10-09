# Trained ACT policies

LeRobot `pretrained_model` folders (weights + normalization stats), tracked with Git LFS.
Run `git lfs pull` after cloning to fetch the weights.

| Folder | Training data | Use |
|---|---|---|
| `act_kinematic/` | `kinematic_v3` labels (8,488 episodes), 40k steps | Results in #39: 48/48 gold drawings at 1 g, 0.325 mm path RMSE |
| `act_kinematic_v2/` | `kinematic_v4` labels with the 5 mm projection cap (hairpin fix), 40k steps | Evaluation in progress (#39) |

Serve one to the Isaac runner (see the ACT section of `isaac-sim/README.md`):

```bash
isaac-sim/.venv-act/bin/python isaac-sim/scripts/act_policy_server.py --checkpoint isaac-sim/models/act_kinematic
python3 isaac-sim/scripts/isaac_batch.py eval --controller act --tag act_kin
```

Evaluation metrics behind these numbers are in `isaac-sim/results/act_eval/` (per-drawing `run_metrics.json`,
batch logs and the comparison reports). Datasets and per-run plots stay local; regenerate them with
`scripts/label_kinematic_ik.py`, `scripts/build_lerobot_dataset.py` and `scripts/isaac_batch.py`.
