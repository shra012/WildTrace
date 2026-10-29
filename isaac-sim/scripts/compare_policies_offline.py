"""ACT vs SmolVLA on the same held-out frames: first-action error in radians.

    isaac-sim/.venv-act/bin/python isaac-sim/scripts/compare_policies_offline.py

Draws 1,024 frames (seed 17) from wildtrace_real_kin_v2_val and compares each
policy's first predicted joint step with the label: relative error (mean
|pred - true| over mean |true|, 1.0 = no better than predicting zero), mean
error in mrad and the median cosine between predicted and true steps.
SmolVLA samples its chunk from noise; modes are noise:samples[:flow_steps], e.g.
random:1 (fresh noise, one sample), random:4 (mean of four), zero:1:5 (zero
noise, 5 flow-matching steps). Arguments pick the modes (default: act random:1
random:4 zero:1); --frames=N, --checkpoint=DIR and --dataset=NAME override the defaults.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import vla_features  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

N = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--frames=")), 1024))
SEED = 17
CHECKPOINT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--checkpoint=")), None)
name = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--dataset=")), "wildtrace_real_kin_v2_val")
ds = LeRobotDataset(f"wildtrace/{name}", root=ROOT.parent / "outputs/lerobot" / name)
idx = np.random.default_rng(SEED).choice(len(ds), N, replace=False)
frames = [ds[int(i)] for i in idx]
true = torch.stack([f["action"] for f in frames]).numpy()  # (N, 7) one control step, rad

def act_preds():
    from lerobot.policies.act.modeling_act import ACTPolicy
    ck = ROOT / "models/act_real_kinematic_v2"
    cfg = PreTrainedConfig.from_pretrained(ck); cfg.device = "cuda"; cfg.n_action_steps = 1; cfg.temporal_ensemble_coeff = None
    pol = ACTPolicy.from_pretrained(ck, config=cfg).cuda().eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(ck))
    out = []
    with torch.no_grad():
        for f in frames:
            pol.reset()
            b = pre({"observation.state": f["observation.state"], "observation.environment_state": f["observation.environment_state"]})
            out.append(post(pol.select_action(b)).reshape(-1).cpu().numpy())
    return np.stack(out)

def vla_preds(noise="random", samples=1, flow_steps=0, checkpoint=None):
    sys.path.insert(0, str(ROOT / "scripts"))
    from act_policy_server import SmolVlaRunner
    ck = Path(checkpoint) if checkpoint else ROOT / "outputs/smolvla/real_kinematic_v2/checkpoints/last/pretrained_model"
    runner = SmolVlaRunner(ck, "cuda", 1, noise, samples, flow_steps)
    # Camera checkpoints draw the paper from the frame's episode so far.
    history, starts = vla_features.load_history(ds.root) if runner.camera else (None, None)
    out, started = [], time.perf_counter()
    for f in frames:
        current = int(f["index"])
        prior = history[starts[current] : current + 1] if runner.camera else None
        out.append(runner.chunk(f["observation.state"].numpy(), f["observation.environment_state"].numpy(),
                                prior)[0].numpy())
    print(json.dumps({"ms_per_chunk": round((time.perf_counter() - started) / len(frames) * 1e3, 1)}), flush=True)
    return np.stack(out)


def report(name, pred):
    err = np.abs(pred - true)
    rel = err.mean() / np.abs(true).mean()
    cos = np.sum(pred * true, 1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-12)
    print(json.dumps({"policy": name, "first_action_rel_error": round(float(rel), 3), "mean_abs_err_mrad": round(float(err.mean() * 1e3), 4),
                      "median_direction_cosine": round(float(np.median(cos)), 3)}), flush=True)

print(json.dumps({"frames": N, "mean_abs_true_mrad": round(float(np.abs(true).mean() * 1e3), 4)}))
modes = [a for a in sys.argv[1:] if not a.startswith("--")] or ["act", "random:1", "random:4", "zero:1"]
for mode in modes:
    if mode == "act":
        report("act_real_kinematic_v2", act_preds())
    else:
        noise, samples, *flow = mode.split(":")
        steps = int(flow[0]) if flow else 0
        report(f"smolvla ({noise} noise, {samples} sample(s), {steps or 10} flow steps)",
               vla_preds(noise, int(samples), steps, CHECKPOINT))
