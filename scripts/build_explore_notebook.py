"""Build notebooks/06_explore_results.ipynb (adapted from ez-penta's post_train_2/view_evals).

    python scripts/build_explore_notebook.py
"""
import json
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8], "metadata": {},
            "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


SETUP = (REPO / "scripts" / "migrate_notebooks.py").read_text(encoding="utf-8").split("SETUP = '''")[1].split("'''")[0]

cells = [
    md("""
# 06 · Explore the results: reproduce the metrics, read the answers, try your own prompt

**Start here.** This notebook is the quickest way to check the project's claims.

1. **Sections 1–3 need no GPU and no Hugging Face token.** They load the saved answers of all three
   models (original Tiny Aya Water, restricted LoRA, broad LoRA) on the 228 validation prompts with
   their Qwen3Guard verdicts, and recompute every reported number from them.
2. **Section 4** lets you read any prompt and answer, filtered by language, culture, jailbreak
   technique and prompt type.
3. **Section 5 needs a GPU** (Colab: Runtime → Change runtime type → GPU) and a Hugging Face token
   whose account has accepted the [Tiny Aya Water terms](https://huggingface.co/CohereLabs/tiny-aya-water).
   It loads the base model plus our published adapter and answers your own prompt with the
   evaluation's exact settings, then shows Qwen3Guard's verdict.

To regenerate all 228 answers instead of using the saved ones, run `05_evaluate_models.ipynb`.

> Content warning: the validation prompts are jailbreak-style requests about violence, drugs,
> self-harm, privacy and similar topics, and a few saved answers were judged unsafe. They are shown
> verbatim because they are the evidence for the reported numbers.
"""),
    md("## 1. Setup"),
    code(SETUP),
    code("""
import pandas as pd
from IPython.display import HTML, display

from tiny_aya_safety import metrics

answers = metrics.load_answers(data_path("pipeline5_answers.jsonl"))
FRAMING_NAMES = {"control": "direct (no jailbreak)", "rp": "role-play", "fc": "fiction",
                 "rs": "refusal suppression"}
answers["technique"] = answers["framing"].map(FRAMING_NAMES)
print(answers.groupby("model")["label"].value_counts().unstack(fill_value=0))
"""),
    md("""
## 2. Reproduce the headline metrics

Recomputed here from the saved answers, not copied. Unsafe rates exclude answers Qwen3Guard
labelled `Controversial` (it gives no safe/unsafe verdict for them); counts show the denominators.
"""),
    code("""
head = metrics.headline_table(answers)
head
"""),
    code("""
for arm in ["restricted", "broad"]:
    for kind, metric, names in [("harmful", "unsafe", ("safe", "unsafe")), ("twin", "refused", ("answered", "refused"))]:
        p = metrics.paired(answers, arm, kind, metric)
        print(f"{arm} vs original, {kind} prompts ({metric}), n={p['n_paired']}:")
        display(pd.DataFrame([[p["both_0"], p["to_bad"]], [p["to_good"], p["both_1"]]],
                             index=[f"original: {names[0]}", f"original: {names[1]}"],
                             columns=[f"{arm}: {names[0]}", f"{arm}: {names[1]}"]))
        print(f"  exact McNemar p = {p['p_exact_mcnemar']:.2g}\\n")
"""),
    md("""
### Seed-level robustness

Several validation prompts are rewrites of the same source request ("seed"), so they are not
independent. These numbers treat each seed as one unit.
"""),
    code("""
rows = []
for arm in ["restricted", "broad"]:
    for kind, exp, metric, title in [("harmful", ["decline"], "unsafe", "unsafe rate, decline prompts"),
                                     ("twin", ["answer"], "refused", "refusal rate, harmless twins")]:
        b = metrics.seed_cluster_bootstrap(answers, arm, kind, exp, metric)
        s = metrics.seed_sign_test(answers, arm, kind, exp, metric)
        rows.append({"model": arm, "metric": title, "seeds": b["n_seeds"], "change (pts)": round(100 * b["diff"], 1),
                     "95% CI": f"[{100 * b['ci95_low']:+.1f}, {100 * b['ci95_high']:+.1f}]",
                     "seeds down/up": f"{s['seeds_down']}/{s['seeds_up']}", "sign-test p": f"{s['p_sign']:.2g}"})
pd.DataFrame(rows)
"""),
    md("## 3. Breakdowns by language, culture and jailbreak technique"),
    code("""
def rate(df, by, metric):
    g = df.dropna(subset=[metric]).groupby([by, "model"])[metric].agg(["sum", "count"])
    return (g["sum"].astype(int).astype(str) + "/" + g["count"].astype(str)).unstack("model")[list(metrics.MODELS)]

harmful, twins = answers[answers["kind"] == "harmful"], answers[answers["kind"] == "twin"]
for by in ["language", "culture", "technique"]:
    print(f"Harmful prompts, unsafe answers by {by}:"); display(rate(harmful, by, "unsafe"))
    print(f"Harmless twins, refusals by {by}:"); display(rate(twins, by, "refused"))
"""),
    code("""
# The culture gap: US-framed vs neutral-framed versions of the same request (Mantel-Haenszel, by language).
pd.DataFrame(metrics.culture_gap(answers)).T
"""),
    md("""
## 4. Browse prompts and answers

Text is shown escaped, as plain text. Use `browse(...)` with any of: `model`, `language` (en/ja),
`culture` (us/neutral), `framing` (control/rp/fc/rs), `kind` (harmful/twin), `expected`
(decline/answer_carefully/answer), `label` (Safe/Unsafe/Controversial), `refused` (0/1).
"""),
    code("""
STYLE = ("<style>.ev td{white-space:pre-wrap;vertical-align:top;text-align:left;max-width:460px;font-size:12px}"
         ".ev th{text-align:left}</style>")
COLS = ["model", "language", "culture", "technique", "kind", "expected", "prompt", "answer", "label", "refused"]


def browse(max_rows=50, **filters):
    df = answers
    for k, v in filters.items():
        df = df[df[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
    print(f"{len(df)} rows" + (f" (showing {max_rows})" if len(df) > max_rows else ""))
    html = df[COLS].head(max_rows).to_html(index=False, escape=True, na_rep="-").replace("<table", '<table class="ev"', 1)
    display(HTML(STYLE + html))
"""),
    code("""# Every answer judged unsafe, for all three models
browse(label="Unsafe")"""),
    code("""# Harmless prompts the restricted model refused
browse(model="restricted", kind="twin", refused=1)"""),
    code("""
# One request answered by all three models, side by side
pid = answers[(answers["kind"] == "harmful") & (answers["label"] == "Unsafe") & (answers["model"] == "baseline")]["id"].iloc[0]
browse(id=pid)
"""),
    md("""
## 5. Try your own prompt (GPU)

Pick a model, paste a prompt between the quotes and run the cell; then run the judge cell. The first
run downloads Tiny Aya Water (6.7 GB) and the adapter, which takes a few minutes; the model stays
loaded afterwards. Add your Hugging Face token as a Colab secret named `HF_TOKEN` first.
"""),
    code("""
MODEL = "restricted"   # "baseline", "restricted" or "broad"
prompt = ""

import os
import subprocess
import sys
if Path("/content").exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "torchao"], check=True)
    if not os.environ.get("HF_TOKEN"):
        try:
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        except Exception:
            print("No HF_TOKEN secret found; the gated base model download will fail.")
from tiny_aya_safety import generation

if not prompt.strip():
    my_answer = None
    print('Paste your prompt between the quotes in prompt = "" and run this cell again.')
else:
    _loaded = globals().setdefault("_loaded_models", {})
    if MODEL not in _loaded:
        for other in list(_loaded):          # keep one Tiny Aya copy on the GPU at a time
            del _loaded[other]
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        _loaded[MODEL] = generation.load_model(MODEL)
    tok, mdl = _loaded[MODEL]
    my_prompt = prompt
    my_answer = generation.generate_answers(mdl, tok, [my_prompt], batch_size=1, verbose=False)[0]
    print(f"[{MODEL}]\\n")
    print(my_answer)
"""),
    code("""
if globals().get("my_answer") is None:
    print("No answer to judge yet: run the prompt cell above first.")
else:
    if "_guard" not in globals():
        _guard = generation.load_guard()
    v = generation.judge_answers(_guard[1], _guard[0], [my_prompt], [my_answer], batch_size=1)[0]
    print("Safety: ", v["label"])
    print("Refusal:", {1: "refusal", 0: "not a refusal"}.get(v["refused"], "unclear"))
    print("\\nRaw Qwen3Guard output:\\n" + v["guard_output"])
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO / "notebooks" / "06_explore_results.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)
