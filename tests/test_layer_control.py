"""The layer-control analysis runs and computes the right numbers (synthetic arms, no GPU)."""
import importlib.util
import json

import pandas as pd

from tiny_aya_safety import metrics
from tiny_aya_safety.config import REPO

spec = importlib.util.spec_from_file_location("alc", REPO / "scripts/analyze_layer_control.py")
alc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alc)


def test_analysis_on_synthetic_arms(tmp_path):
    # Fake arms from real saved answers: "layer 18" = the published restricted adapter's answers,
    # "layer 9" = the broad adapter's, so every expected number is already known.
    src = pd.read_json(REPO / "data/results/pipeline5_answers.jsonl", lines=True)
    for model, fake in [("restricted", "L18"), ("broad", "L9")]:
        for seed in (0, 1):
            rows = src[src["model"] == model].assign(model=f"{fake}_s{seed}")
            rows.to_json(tmp_path / f"answers_{fake}_s{seed}.jsonl", orient="records", lines=True, force_ascii=False)

    df, names = alc.load(tmp_path)
    assert names == ["L18_s0", "L18_s1", "L9_s0", "L9_s1"]
    out, head, order = alc.analyse(df, names, tmp_path)
    assert head["L18_s0"]["needless_refusal_twins"] == "16/80" and head["L9_s1"]["needless_refusal_twins"] == "29/80"
    assert head["L18_published"]["unsafe_decline"] == "1/117"
    v = out["vs_layer18"]["L9_s0"]
    assert v["reference"] == "L18_s0"
    # Same numbers as the committed restricted-vs-broad comparison.
    committed = json.loads((REPO / "results/metrics.json").read_text())["paired"]["broad_vs_restricted"]["twin_refused"]
    assert (v["twin_refused"]["to_bad"], v["twin_refused"]["to_good"]) == (committed["to_bad"], committed["to_good"])
    assert out["per_layer"][18]["needless_refusal_rate_twins"]["mean"] == 0.2
    md = alc.tables(out, head, order)
    assert "Layer 9" in md and "16/80" in md
    alc.figure(head, order, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").stat().st_size > 10_000
    metrics.to_json(out, tmp_path / "m.json")
    json.loads((tmp_path / "m.json").read_text())
