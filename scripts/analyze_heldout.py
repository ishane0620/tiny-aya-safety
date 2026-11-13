"""Analyse the heldout (final test) answers. See docs/HELDOUT_PLAN.md.

    python scripts/analyze_heldout.py [results/heldout]

Reads <dir>/answers_<model>.jsonl for whichever models are complete and writes
<dir>/HELDOUT_TABLES.md, <dir>/metrics.json and <dir>/heldout.png. CPU only.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tiny_aya_safety import metrics  # noqa: E402

ORDER = ["baseline", "restricted", "broad", "L9_s0", "L9_s1", "L18_s0", "L18_s1"]
LABEL = {"baseline": "Original", "restricted": "Restricted L18 (published)", "broad": "Broad (all layers)",
         "L9_s0": "L9 seed 0", "L9_s1": "L9 seed 1", "L18_s0": "L18 seed 0", "L18_s1": "L18 seed 1"}
SETS = [("pure", "Pure English/Japanese"), ("mixed", "Code-mixed"), ("all", "All heldout")]


def r1(x):
    from decimal import ROUND_HALF_UP, Decimal
    return str((Decimal(str(round(x, 10))) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def pct(n, d):
    return f"{r1(n / d)}% ({n}/{d})" if d else "–"


def summary_row(g):
    dec = g[(g.kind == "harmful") & (g.expected == "decline")]
    care = g[(g.kind == "harmful") & (g.expected == "answer_carefully")]
    tw = g[g.kind == "twin"]
    judged = dec[dec.unsafe.notna()]
    return {
        "unsafe_decline": (int(judged.unsafe.sum()), len(judged)),
        "unsafe_any_harmful": (int(g[(g.kind == "harmful") & g.unsafe.notna()].unsafe.sum()),
                               int(g[(g.kind == "harmful") & g.unsafe.notna()].shape[0])),
        "needless_refusals": (int(tw.refused.eq(1).sum()), int(tw.refused.notna().sum())),
        "care_refusals": (int(care.refused.eq(1).sum()), int(care.refused.notna().sum())),
        "twin_chars": float(tw.answer.str.len().mean()) if len(tw) else float("nan"),
        "controversial": int((g.label == "Controversial").sum()),
    }


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "heldout"
    frames = [metrics.load_answers(p) for m in ORDER if (p := d / f"answers_{m}.jsonl").exists()]
    if not frames:
        raise SystemExit(f"no complete answers_<model>.jsonl in {d}")
    df = pd.concat(frames, ignore_index=True)
    models = [m for m in ORDER if m in set(df.model)]
    out = {"models": models, "n_prompts": int(df[df.model == models[0]].shape[0]), "sets": {}, "paired": {}, "seed_level": {}}
    lines = ["# Heldout (final test) results", "",
             f"Qwen3Guard-Gen-0.6B judgments; {out['n_prompts']} heldout prompts per model "
             "(representative stratified sample, see docs/HELDOUT_PLAN.md). Not human-reviewed.", ""]

    for key, title in SETS:
        sub = df if key == "all" else df[df["set"] == key]
        rows = {m: summary_row(sub[sub.model == m]) for m in models}
        out["sets"][key] = rows
        lines += [f"## {title}", "",
                  "| Model | Unsafe, decline | Unsafe, all harmful | Needless refusals, twins | Refusals, answer-carefully | Twin answer length |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m in models:
            r = rows[m]
            lines.append(f"| {LABEL[m]} | {pct(*r['unsafe_decline'])} | {pct(*r['unsafe_any_harmful'])} | "
                         f"{pct(*r['needless_refusals'])} | {pct(*r['care_refusals'])} | {r['twin_chars']:,.0f} |")
        lines.append("")

    lines += ["## Same prompt, two models (exact McNemar)", "",
              "| Comparison | Set | Harmful: fixed / broken | p | Twins: newly refused / now answered | p |",
              "|---|---|---:|---:|---:|---:|"]
    comps = [(m, "baseline") for m in models if m != "baseline"]
    comps += [(a, b) for a, b in [("broad", "restricted"), ("L9_s0", "L18_s0"), ("L9_s1", "L18_s1")]
              if a in models and b in models]
    for arm, base in comps:
        for key in ("pure", "mixed", "all"):
            sub = df if key == "all" else df[df["set"] == key]
            h = metrics.paired(sub, arm, "harmful", "unsafe", base=base)
            t = metrics.paired(sub, arm, "twin", "refused", base=base)
            out["paired"][f"{arm}_vs_{base}_{key}"] = {"harmful_unsafe": h, "twin_refused": t}
            lines.append(f"| {LABEL[arm]} vs {LABEL[base]} | {key} | {h['to_good']} / {h['to_bad']} | "
                         f"{h['p_exact_mcnemar']:.2g} | {t['to_bad']} / {t['to_good']} | {t['p_exact_mcnemar']:.2g} |")

    lines += ["", "## Seed-level robustness (all heldout prompts; 95% CI resampling source seeds)", "",
              "| Comparison | Metric | prompts / seeds | change | 95% CI | seeds down / up |", "|---|---|---:|---:|---:|---:|"]
    for arm, base in comps:
        for kind, exp, metric, mname in [("harmful", ["decline"], "unsafe", "unsafe, decline"),
                                         ("twin", ["answer"], "refused", "needless refusals")]:
            b = metrics.seed_cluster_bootstrap(df, arm, kind, exp, metric, base=base)
            s = metrics.seed_sign_test(df, arm, kind, exp, metric, base=base)
            out["seed_level"][f"{arm}_vs_{base}_{metric}"] = {"bootstrap": b, "sign": s}
            sgn = lambda x: ("+" if x >= 0 else "") + r1(x)
            lines.append(f"| {LABEL[arm]} vs {LABEL[base]} | {mname} | {b['n_prompts']} / {b['n_seeds']} | "
                         f"{sgn(b['diff'])} pts | [{sgn(b['ci95_low'])}, {sgn(b['ci95_high'])}] | {s['seeds_down']} / {s['seeds_up']} |")

    pure_h = df[(df["set"] == "pure") & (df.kind == "harmful") & df.unsafe.notna()]
    out["culture_gap_pure"] = {m: metrics.mantel_haenszel(pure_h[pure_h.model == m]) for m in models}
    lines += ["", "## US vs neutral framing, pure prompts (Mantel–Haenszel by language)", "",
              "| Model | odds ratio | p | n |", "|---|---:|---:|---:|"]
    for m, r in out["culture_gap_pure"].items():
        orr = "undefined" if r["or_mh"] != r["or_mh"] else f"{r['or_mh']:.2f}"
        lines.append(f"| {LABEL[m]} | {orr} | {r['p_mh']:.3g} | {r['n_rows']} |")

    lines += ["", "## By jailbreak technique (all heldout; harmful unsafe / twins refused)", "",
              "| Model | " + " | ".join(f"{f} unsafe" for f in ("control", "rp", "fc", "rs")) + " | " +
              " | ".join(f"{f} refused" for f in ("control", "rp", "fc", "rs")) + " |", "|---|" + "---:|" * 8]
    for m in models:
        g = df[df.model == m]
        cells = []
        for kind, col in (("harmful", "unsafe"), ("twin", "refused")):
            for f in ("control", "rp", "fc", "rs"):
                x = g[(g.kind == kind) & (g.framing == f)][col].dropna()
                cells.append(f"{int(x.sum())}/{len(x)}")
        lines.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")

    (d / "HELDOUT_TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics.to_json(out, d / "metrics.json")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    colors = {"baseline": "#888780", "restricted": "#1D9E75", "broad": "#7F77DD", "L9_s0": "#D85A30",
              "L9_s1": "#D85A30", "L18_s0": "#5DCAA5", "L18_s1": "#5DCAA5"}
    for ax, (key, title) in zip(axes, SETS[:2]):
        for m in models:
            r = out["sets"][key][m]
            x = 100 * r["needless_refusals"][0] / max(r["needless_refusals"][1], 1)
            y = 100 * r["unsafe_decline"][0] / max(r["unsafe_decline"][1], 1)
            ax.scatter(x, y, s=110, color=colors[m], zorder=3, marker="o" if "_s" not in m else "^")
            ax.annotate(LABEL[m], (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
        ax.set_title(f"Heldout, {title.lower()}", fontsize=11)
        ax.set_xlabel("Needless refusals on harmless twins (%)")
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Unsafe answers on decline prompts (%)")
    fig.tight_layout()
    fig.savefig(d / "heldout.png", dpi=160)
    print(f"wrote {d}/HELDOUT_TABLES.md, metrics.json, heldout.png for {models}")


if __name__ == "__main__":
    main()
