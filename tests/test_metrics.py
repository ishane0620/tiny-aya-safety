"""The reported numbers must be recomputable from the saved answers."""
import json
import math
from pathlib import Path

import pytest

from tiny_aya_safety import metrics
from tiny_aya_safety.config import REPO

ANSWERS = REPO / "data/results/pipeline5_answers.jsonl"
PIPELINE5 = json.loads((REPO / "data/results/pipeline5_results.json").read_text())


@pytest.fixture(scope="module")
def df():
    return metrics.load_answers(ANSWERS)


def test_answer_file_shape(df):
    assert len(df) == 3 * 228
    assert sorted(df["model"].unique()) == ["baseline", "broad", "restricted"]
    assert df.groupby("model")["id"].nunique().eq(228).all()


def test_headline_counts(df):
    h = metrics.headline_table(df)
    assert h["baseline"]["unsafe_decline"] == "12/115"
    assert h["restricted"]["unsafe_decline"] == "1/117"
    assert h["broad"]["unsafe_decline"] == "1/118"
    assert [h[m]["needless_refusal_twins"] for m in metrics.MODELS] == ["8/80", "16/80", "29/80"]
    assert [h[m]["refusal_answer_carefully"] for m in metrics.MODELS] == ["9/30", "14/30", "13/30"]
    assert round(h["baseline"]["mean_harmless_answer_chars"]) == 1072


def test_matches_original_pipeline5_mcnemar(df):
    for arm, orig in PIPELINE5["mcnemar_vs_baseline_harmful"].items():
        p = metrics.paired(df, arm, "harmful", "unsafe")
        assert (p["n_paired"], p["to_good"], p["to_bad"]) == (orig["n_paired"], orig["unsafe_to_safe"], orig["safe_to_unsafe"])
        assert math.isclose(p["p_exact_mcnemar"], orig["p_exact_mcnemar"])


def test_matches_original_pipeline5_mantel_haenszel(df):
    gap = metrics.culture_gap(df)
    for model, orig in PIPELINE5["mantel_haenszel_harmful"].items():
        assert gap[model]["n_rows"] == orig["n_rows"]
        for k in ("or_mh", "p_mh"):
            a, b = gap[model][k], orig[k]
            assert (math.isnan(a) and (b is None or math.isnan(b))) or math.isclose(a, b, rel_tol=1e-9)


def test_committed_results_are_current(df):
    """results/metrics.json must equal a fresh recomputation (rerun scripts/compute_metrics.py)."""
    fresh = json.loads(json.dumps(metrics.compute_all(df), default=float))
    committed = json.loads((REPO / "results/metrics.json").read_text())
    committed.pop("source", None)
    assert fresh == committed


def test_mantel_haenszel_textbook_case():
    import pandas as pd
    rows = []
    for stratum, (a, b, c, d) in {"s1": (10, 20, 5, 25), "s2": (8, 12, 4, 16)}.items():
        for exp, out, n in [("us", 1, a), ("us", 0, b), ("neutral", 1, c), ("neutral", 0, d)]:
            rows += [{"culture": exp, "unsafe": out, "language": stratum}] * n
    r = metrics.mantel_haenszel(pd.DataFrame(rows))
    expected_or = (10 * 25 / 60 + 8 * 16 / 40) / (20 * 5 / 60 + 12 * 4 / 40)
    assert math.isclose(r["or_mh"], expected_or)
