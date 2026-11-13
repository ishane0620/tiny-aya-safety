"""Analyse the layer-choice control: is layer 18 better than other single-matrix adapters?

    python scripts/analyze_layer_control.py [answers_dir]     # default: results/layer_control

Reads answers_L{layer}_s{seed}.jsonl from answers_dir, plus the original run's answers
(original model, published layer-18 adapter, broad adapter) from data/results/pipeline5_answers.jsonl.
Writes results/layer_control/{metrics.json, LAYER_CONTROL_TABLES.md, layer_control.png}.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tiny_aya_safety import metrics  # noqa: E402

SELECTED = 18
LAYER_ROLE = {18: "selected (activation analysis)", 15: "runner-up", 9: "control (never top-5)",
              30: "control (never top-5)"}


def pct(x):
    from decimal import ROUND_HALF_UP, Decimal
    return str((Decimal(str(round(x, 10))) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) + "%"


def load(answers_dir: Path):
    orig = metrics.load_answers(REPO / "data/results/pipeline5_answers.jsonl")
    orig["model"] = orig["model"].replace({"restricted": "L18_published"})
    arms = [metrics.load_answers(p) for p in sorted(answers_dir.glob("answers_L*_s*.jsonl"))]
    if not arms:
        raise SystemExit(f"no answers_L*_s*.jsonl files in {answers_dir}")
    df = pd.concat([orig, *arms], ignore_index=True)
    names = sorted({m for a in arms for m in a["model"].unique()},
                   key=lambda m: (int(m.split("_")[0][1:]) != SELECTED, int(m.split("_")[0][1:]), m))
    return df, names


def layer_of(name):
    return int(re.match(r"L(\d+)", name).group(1))


def analyse(df, arm_names, answers_dir: Path):
    order = ["baseline", "L18_published", *arm_names, "broad"]
    head = metrics.headline_table(df, models=order)
    out = {"arms": arm_names, "headline": head.to_dict(), "vs_original": {}, "vs_layer18": {}, "train": {}}
    for name in ["L18_published", *arm_names]:
        out["vs_original"][name] = {"harmful_unsafe": metrics.paired(df, name, "harmful", "unsafe"),
                                    "twin_refused": metrics.paired(df, name, "twin", "refused")}
    for name in arm_names:
        layer, seed = layer_of(name), name.split("_s")[1]
        ref = f"L{SELECTED}_s{seed}"
        if layer == SELECTED or ref not in arm_names:
            continue
        out["vs_layer18"][name] = {
            "reference": ref,
            # base = layer 18: to_bad counts prompts the control gets worse on than layer 18.
            "twin_refused": metrics.paired(df, name, "twin", "refused", base=ref),
            "harmful_unsafe": metrics.paired(df, name, "harmful", "unsafe", base=ref),
            "twin_refused_seed_bootstrap": metrics.seed_cluster_bootstrap(df, name, "twin", ["answer"], "refused", base=ref),
        }
    for name in arm_names:
        log = answers_dir / "adapters" / name / "train_log.json"
        if log.exists():
            t = json.loads(log.read_text())
            out["train"][name] = {k: t.get(k) for k in ("trainable_params", "loss_first_quarter", "loss_last_quarter",
                                                         "train_seconds", "peak_gpu_gb", "max_abs_lora_B")}
    # Per-layer summary across seeds
    summary = {}
    for layer in sorted({layer_of(n) for n in arm_names}, key=lambda l: (l != SELECTED, l)):
        names = [n for n in arm_names if layer_of(n) == layer]
        vals = {k: [head[n][k] for n in names] for k in ("unsafe_rate_decline", "needless_refusal_rate_twins",
                                                          "refusal_rate_answer_carefully", "mean_harmless_answer_chars")}
        summary[layer] = {"role": LAYER_ROLE.get(layer, "other"), "seeds": names,
                          **{k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v)} for k, v in vals.items()}}
    out["per_layer"] = summary
    return out, head, order


def tables(out, head, order):
    label = {"baseline": "Original model", "L18_published": "Layer 18, published adapter (original run)", "broad": "Broad LoRA (all layers), reference"}
    L = ["# Layer-choice control", "",
         "Same one-matrix LoRA (`layers.N.mlp.down_proj`, 208,896 trainable parameters), same 195 training pairs and "
         "settings, trained on different layers with two seeds each; evaluated on the same 228 validation prompts "
         "(Qwen3Guard judgments, analysis split).", "",
         "## Every arm", "",
         "| Model | Unsafe, decline prompts | Needless refusals, twins | Refusals, answer-carefully | Harmless answer length | Train loss (first → last quarter) |",
         "|---|---:|---:|---:|---:|---:|"]
    for m in order:
        h = head[m]
        t = out["train"].get(m)
        loss = f"{t['loss_first_quarter']:.2f} → {t['loss_last_quarter']:.2f}" if t else "–"
        name = label[m] if m in label else \
            f"Layer {layer_of(m)} ({LAYER_ROLE.get(layer_of(m), 'other')}), seed {m.split('_s')[1]}"
        L.append(f"| {name} | {pct(h['unsafe_rate_decline'])} ({h['unsafe_decline']}) | "
                 f"{pct(h['needless_refusal_rate_twins'])} ({h['needless_refusal_twins']}) | "
                 f"{pct(h['refusal_rate_answer_carefully'])} ({h['refusal_answer_carefully']}) | "
                 f"{h['mean_harmless_answer_chars']:,.0f} | {loss} |")
    L += ["", "## Per layer (mean across seeds, [min, max])", "",
          "| Layer | Role | Unsafe, decline | Needless refusals, twins | Refusals, answer-carefully | Answer length |",
          "|---:|---|---:|---:|---:|---:|"]
    for layer, s in out["per_layer"].items():
        f = lambda k, p=True: (f"{pct(s[k]['mean'])} [{pct(s[k]['min'])}, {pct(s[k]['max'])}]" if p else
                               f"{s[k]['mean']:,.0f} [{s[k]['min']:,.0f}, {s[k]['max']:,.0f}]")
        L.append(f"| {layer} | {s['role']} | {f('unsafe_rate_decline')} | {f('needless_refusal_rate_twins')} | "
                 f"{f('refusal_rate_answer_carefully')} | {f('mean_harmless_answer_chars', False)} |")
    L += ["", "## Each arm vs the original model (same prompts, exact McNemar)", "",
          "| Arm | harmful: fixed / broken | p | twins: newly refused / now answered | p |", "|---|---:|---:|---:|---:|"]
    for m, v in out["vs_original"].items():
        h, t = v["harmful_unsafe"], v["twin_refused"]
        L.append(f"| {m} | {h['to_good']} / {h['to_bad']} | {h['p_exact_mcnemar']:.2g} | {t['to_bad']} / {t['to_good']} | {t['p_exact_mcnemar']:.2g} |")
    if out["vs_layer18"]:
        L += ["", "## Each other layer vs layer 18 with the same seed (same prompts)", "",
              "Twins: prompts the other layer refuses but layer 18 answers, and the reverse. Positive change = the other "
              "layer refuses more harmless prompts than layer 18.", "",
              "| Arm | vs | twins: refuses where L18 answers / answers where L18 refuses | p | change in twin refusal rate [seed-bootstrap 95% CI] | harmful: unsafe where L18 safe / safe where L18 unsafe |",
              "|---|---|---:|---:|---:|---:|"]
        for m, v in out["vs_layer18"].items():
            t, h, b = v["twin_refused"], v["harmful_unsafe"], v["twin_refused_seed_bootstrap"]
            L.append(f"| {m} | {v['reference']} | {t['to_bad']} / {t['to_good']} | {t['p_exact_mcnemar']:.2g} | "
                     f"{100 * b['diff']:+.1f} pts [{100 * b['ci95_low']:+.1f}, {100 * b['ci95_high']:+.1f}] | {h['to_bad']} / {h['to_good']} |")
    return "\n".join(L) + "\n"


def figure(head, order, path):
    colors = {18: "#1D9E75", 15: "#5DCAA5", 9: "#D85A30", 30: "#BA7517"}
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for m in order:
        x, y = 100 * head[m]["needless_refusal_rate_twins"], 100 * head[m]["unsafe_rate_decline"]
        if m == "baseline":
            ax.scatter(x, y, s=120, color="#888780", zorder=3); ax.annotate("Original", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
        elif m == "broad":
            ax.scatter(x, y, s=120, color="#7F77DD", zorder=3); ax.annotate("Broad (all layers)", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9)
        elif m == "L18_published":
            ax.scatter(x, y, s=120, color=colors[18], marker="*", zorder=4)
        else:
            layer = layer_of(m)
            ax.scatter(x, y, s=80, color=colors.get(layer, "#444"), marker="o" if m.endswith("s0") else "s", zorder=3,
                       label=f"layer {layer}" if m.endswith("s0") else None)
    ax.set_xlabel("Needless refusals on harmless twins (%)  → less helpful")
    ax.set_ylabel("Unsafe answers on decline prompts (%)  → less safe")
    ax.set_title("Which layer? One-matrix LoRA on different layers\n(● seed 0, ■ seed 1, ★ published layer-18 adapter)", fontsize=10)
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    answers_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/layer_control"
    out_dir = REPO / "results/layer_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, names = load(answers_dir)
    out, head, order = analyse(df, names, answers_dir)
    metrics.to_json(out, out_dir / "metrics.json")
    (out_dir / "LAYER_CONTROL_TABLES.md").write_text(tables(out, head, order), encoding="utf-8")
    figure(head, order, out_dir / "layer_control.png")
    print(f"analysed {len(names)} arms -> {out_dir}")


if __name__ == "__main__":
    main()
