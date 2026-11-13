"""Independent check that training, layer selection and validation use disjoint source seeds.

The builder scripts assert this themselves; this module re-derives it from the published
artifacts and Dataset2's `observations` table, in two ways:

- by prompt ID (the rows each step actually used), which is the authoritative check;
- by exact prompt text, which also catches identical wording generated for two different seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def split_report(observations: pd.DataFrame, data_dir: str | Path) -> dict:
    """observations needs columns prompt_id, prompt, seed_id, split."""
    d = Path(data_dir)
    by_id = observations.set_index("prompt_id")

    val = _read_json(d / "prompts/pipeline5_validation_prompts_v2.json")
    seed_prompts = _read_json(d / "prompts/pipeline3_seed_prompts_v2.json") + \
        _read_json(d / "prompts/pipeline3_helpful_seed_prompts_v2.json")
    instruction_to_id = {x["instruction"]: x["id"] for x in seed_prompts}
    pairs = [json.loads(l) for l in (d / "training/pipeline3_safety_pairs_v2.jsonl").read_text().splitlines() if l]
    p2 = _read_json(d / "prompts/pipeline2_prompts_v2.json")
    p2_prompts = [p for lang in p2.values() for v in lang.values() if isinstance(v, list) for p in v]

    train_ids = [instruction_to_id.get(p["instruction"]) for p in pairs]
    val_ids = [r["id"] for r in val]
    # Pipeline 2 stored prompt text only; its builder drew from tune rows, so resolve text within tune.
    tune_text_to_seed = observations[observations["split"] == "tune"].groupby("prompt")["seed_id"].first()

    train_seeds = set(by_id.loc[[i for i in train_ids if i], "seed_id"])
    val_seeds = set(by_id.loc[val_ids, "seed_id"])
    p2_seeds = {tune_text_to_seed[p] for p in p2_prompts if p in tune_text_to_seed.index}

    train_text = {p["instruction"] for p in pairs}
    val_text = {r["prompt"] for r in val}
    shared_p2_val_text = sorted(set(p2_prompts) & val_text)
    return {
        "by_id": {
            "training_pairs": {"n": len(pairs), "unmapped": sum(i is None for i in train_ids),
                               "splits": by_id.loc[[i for i in train_ids if i], "split"].value_counts().to_dict(),
                               "n_seeds": len(train_seeds)},
            "layer_selection": {"n": len(p2_prompts), "resolved_in_tune": sum(p in tune_text_to_seed.index for p in p2_prompts),
                                "n_seeds": len(p2_seeds)},
            "validation": {"n": len(val), "splits": by_id.loc[val_ids, "split"].value_counts().to_dict(),
                           "n_seeds": len(val_seeds)},
            "seed_overlap_training_validation": len(train_seeds & val_seeds),
            "seed_overlap_layer_selection_validation": len(p2_seeds & val_seeds),
            "heldout_rows_used": int((by_id.loc[[i for i in train_ids if i] + val_ids, "split"] == "heldout").sum()),
        },
        "by_exact_text": {
            "training_and_validation": len(train_text & val_text),
            "layer_selection_and_validation": len(shared_p2_val_text),
            "shared_layer_selection_validation_texts": shared_p2_val_text,
        },
    }
