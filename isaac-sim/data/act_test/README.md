# ACT held-out test set

The 48 gold trajectories (`<Category>/<sample_id>.json`, listed in `test_split.json`). They are the
fixed test set for every ACT-vs-IK comparison in #39 and must never be trained on.

- `scripts/label_kinematic_ik.py`, `scripts/build_lerobot_dataset.py` and `isaac_batch.py record`
  exclude these sample ids by default (`--exclude-dir`).
- `isaac_batch.py eval` draws exactly these by default (`--trajectories`).

Committed so every machine uses the same split: a local `outputs/gold/trajectories` can differ between
checkouts (it is a pipeline output and gitignored).
