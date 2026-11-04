"""Build Table I from paper-r3; refuse an incomplete study unless explicitly requested."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GENERATORS = {"informative-qwen3": "Informative drawings", "sd15cn-qwen3": "SD1.5 + ControlNet",
              "flux-qwen3": "FLUX.1-schnell", "omnigen2-qwen3": "OmniGen2"}
JUDGES = {"qwen2.5-vl-7b": "Qwen2.5-VL-7B (local)", "qwen3-vl-8b": "Qwen3-VL-8B",
          "qwen3-vl-30b-a3b": "Qwen3-VL-30B-A3B", "gemma-3-12b": "Gemma-3-12B"}
STRONG = "qwen3-vl-235b"


def table(summary, *, allow_incomplete=False):
    setups = summary["setups"]
    n_images = {key: setups[key]["final"]["clip_top1"]["n"] for key in GENERATORS}
    agreement = {}
    for judge in JUDGES:
        agreement[judge] = next(v for key, v in summary["referee_agreement"].items()
                                if set(key.split(" vs ")) == {judge, STRONG})
    complete = all(n == 20 and setups[key]["referee"][STRONG]["n"] == 20 for key, n in n_images.items())
    complete = complete and all(v["n"] == 80 for v in agreement.values())
    if not complete and not allow_incomplete:
        raise ValueError(f"Study is incomplete: final images {n_images}; judge pairs {[v['n'] for v in agreement.values()]}")
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Generator and judge study (paper-r3, 20 subjects per generator). $n$: evaluated images or paired verdicts. Pass: Qwen3-VL-235B; export: trajectory constraints.}",
             r"\label{tab:judge}", r"\setlength{\tabcolsep}{2.6pt}", r"\begin{tabular}{lrrrrr}",
             r"\toprule", r"Generator & $n$ & Pass & BioCLIP & Blank & Export \\", r"\midrule"]
    for key, label in GENERATORS.items():
        data = setups[key]
        final = data["final"]
        passed = data["referee"][STRONG]
        passes = round(passed["value"] * passed["n"])
        pct = lambda name: f"{100 * final[name]['value']:.0f}\\%"
        lines.append(f"{label} & {n_images[key]} & {passes}/{passed['n']} & {pct('clip_top1')} & {pct('blank')} & {pct('trajectory_ok')} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{3pt}", r"\begin{tabular}{lrrr}", r"\toprule",
              r"Judge & $n$ & Agreement & Cohen's $\kappa$ \\", r"\midrule"]
    for key, label in JUDGES.items():
        a = agreement[key]
        lines.append(f"{label} & {a['n']} & {100 * a['agreement']:.1f}\\% & {a['kappa']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if not complete:
        lines += [r"\par\smallskip\footnotesize\textbf{Interim:} hosted judge failures interrupted some generator subjects and referee calls; missing verdicts are not model rejections."]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def metrics(summary):
    values = {}
    complete = True
    for key, prefix in (("flux-qwen3", "Flux"), ("omnigen2-qwen3", "Omni")):
        data = summary["setups"][key]["final"]
        n = data["silhouette_iou"]["n"]
        values.update({prefix + "ImageCount": str(n), prefix + "MeanIoU": f"{data['silhouette_iou']['value']:.3f}",
                       prefix + "BlankCount": str(round(data["blank"]["value"] * n)),
                       prefix + "DirectionMatch": f"{100 * data['direction_match']['value']:.0f}\\%"})
    for key in GENERATORS:
        data = summary["setups"][key]
        complete = complete and data["final"]["clip_top1"]["n"] == 20 and data["referee"][STRONG]["n"] == 20
    complete = complete and all(summary["referee_agreement"][key]["n"] == 80
                                for key in summary["referee_agreement"] if STRONG in key)
    return "\n".join([r"\newif\ifGeneratorStudyComplete", r"\GeneratorStudyComplete" + ("true" if complete else "false")]
                      + [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=REPO / "outputs/benchmarks/paper-r3/summary.json")
    p.add_argument("--out", type=Path, default=REPO / "paper/tables/generator_judges.tex")
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args()
    summary = json.loads(args.summary.read_text())
    tex = table(summary, allow_incomplete=args.allow_incomplete)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    args.out.with_name("generator_metrics.tex").write_text(metrics(summary))
    print(args.out)


if __name__ == "__main__":
    main()
