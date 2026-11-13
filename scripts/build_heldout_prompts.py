"""Build the heldout (final test) prompt set: data/prompts/heldout_test_prompts.json.

    python scripts/build_heldout_prompts.py

Representative stratified sample of all eligible heldout prompts (see docs/HELDOUT_PLAN.md):
every clearly harmful (decline) prompt, plus samples of borderline prompts and harmless twins,
for pure English/Japanese and code-mixed prompts. Deterministic (seed 42).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tiny_aya_safety.config import data_path  # noqa: E402

RNG_SEED = 42
PLAN = {  # (set, kind, expected): number of prompts (None = all)
    ("pure", "harmful", "decline"): None,
    ("pure", "harmful", "answer_carefully"): 100,
    ("pure", "twin", "answer"): 200,
    ("mixed", "harmful", "decline"): None,
    ("mixed", "harmful", "answer_carefully"): 60,
    ("mixed", "twin", "answer"): 150,
}


def stratified_sample(pool: pd.DataFrame, n: int, strata: list[str], rng) -> pd.DataFrame:
    """Proportional allocation across strata (largest remainders), then within each stratum
    pick prompts round-robin over seeds so no seed dominates."""
    if n is None or n >= len(pool):
        return pool
    sizes = pool.groupby(strata).size()
    exact = sizes / sizes.sum() * n
    alloc = np.floor(exact).astype(int)
    for key in (exact - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[key] += 1
    picked = []
    for key, k in alloc.items():
        if k == 0:
            continue
        g = pool.set_index(strata).loc[[key] if not isinstance(key, tuple) else [key]].reset_index()
        g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        g["_rank"] = g.groupby("seed_id").cumcount()          # seed's 1st, 2nd, ... prompt
        picked.append(g.sort_values(["_rank"], kind="stable").head(k).drop(columns="_rank"))
    return pd.concat(picked)


def main():
    df = pd.read_parquet(data_path("observations.parquet"))
    pool = df[(df["split"] == "heldout") & (df["analysis_eligible"] == True)
              & (df["generation_status"] == "completed") & (df["scoring_status"] == "completed")].copy()
    pool["set"] = np.where(pool["prompt_language"] == "mix", "mixed", "pure")
    rng = np.random.default_rng(RNG_SEED)
    parts = []
    for (s, kind, expected), n in PLAN.items():
        sub = pool[(pool["set"] == s) & (pool["kind"] == kind) & (pool["expected"] == expected)]
        lang_col = "mix_direction" if s == "mixed" else "prompt_language"
        parts.append(stratified_sample(sub, n, [lang_col, "framing", "culture"], rng))
    test = pd.concat(parts).sort_values(["set", "kind", "expected", "prompt_id"]).reset_index(drop=True)

    # ---- checks: heldout only, and no prompt text seen anywhere else in the project
    assert set(test["split"]) == {"heldout"}
    seen = set()
    for name in ("pipeline3_seed_prompts_v2.json", "pipeline3_helpful_seed_prompts_v2.json",
                 "pipeline5_validation_prompts_v2.json"):
        seen |= {r.get("instruction") or r.get("prompt") for r in json.loads(data_path(name).read_text())}
    p2 = json.loads(data_path("pipeline2_prompts_v2.json").read_text())
    seen |= {p for lang in p2.values() for v in lang.values() if isinstance(v, list) for p in v}
    seen |= {json.loads(l)["instruction"] for l in data_path("pipeline3_safety_pairs_v2.jsonl").read_text().splitlines()}
    shared = set(test["prompt"]) & seen
    assert not shared, f"{len(shared)} heldout prompt texts also appear in earlier steps"
    other_seeds = set(df.loc[df["split"] != "heldout", "seed_id"])
    assert not set(test["seed_id"]) & other_seeds

    rows = [{
        "prompt": r.prompt, "id": r.prompt_id, "seed_id": r.seed_id, "language": r.prompt_language,
        "culture": r.culture, "framing": r.framing, "kind": r.kind, "expected": r.expected,
        "category": r.category, "mix_direction": r.mix_direction if r.set == "mixed" else None,
        "output_language": r.output_language, "set": r.set, "pair_id": i, "pair_bucket": f"heldout_{r.set}",
        "gguf_baseline_label": r.safety,
    } for i, r in enumerate(test.itertuples())]
    out = REPO / "data" / "prompts" / "heldout_test_prompts.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}: {len(rows)} prompts, {test['seed_id'].nunique()} seeds")
    print(test.groupby(["set", "kind", "expected"]).size().to_string())
    print("languages:", test["prompt_language"].value_counts().to_dict(), "| framings:", test["framing"].value_counts().to_dict())


if __name__ == "__main__":
    main()
