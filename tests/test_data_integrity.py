"""Published artifacts are the post_train_2 ones, unmodified, and the splits are seed-disjoint."""
import hashlib
import json
import zipfile

import pytest

from tiny_aya_safety.config import CONFIG, DATA, REPO
from tiny_aya_safety.generation import parse_guard


def _manifest():
    for line in (DATA / "MANIFEST.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        yield digest, DATA / path.lstrip("./")


def test_manifest_hashes():
    checked = 0
    for digest, path in _manifest():
        if not path.exists():          # large files live on Hugging Face, not in git
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
        checked += 1
    assert checked >= 11


def test_training_pairs():
    pairs = [json.loads(l) for l in (DATA / "training/pipeline3_safety_pairs_v2.jsonl").read_text().splitlines()]
    assert len(pairs) == 195
    assert sum(p["type"] == "refusal" for p in pairs) == 116
    assert sum(p["type"] == "helpful" for p in pairs) == 79


def test_validation_design():
    val = json.loads((DATA / "prompts/pipeline5_validation_prompts_v2.json").read_text())
    assert len(val) == 228
    pairs = {}
    for r in val:
        pairs.setdefault(r["pair_id"], []).append(r)
    assert len(pairs) == 114 and all(sorted(x["culture"] for x in p) == ["neutral", "us"] for p in pairs.values())
    buckets = {}
    for p in pairs.values():
        buckets[p[0]["pair_bucket"]] = buckets.get(p[0]["pair_bucket"], 0) + 1
    assert buckets == {"us_unsafe_neutral_safe": 43, "both_safe": 17, "us_safe_neutral_unsafe": 14, "twin_probe": 40}


def test_pipeline4_summary_matches_config():
    p4 = json.loads((DATA / "results/pipeline4_results.json").read_text())
    p2 = json.loads((DATA / "results/pipeline2_results.json").read_text())
    assert p4["target_layers"] == p2["target_layers"] == [18]
    assert p4["n_training_pairs"] == 195 and p4["max_len"] == CONFIG["lora"]["max_len"]


@pytest.mark.skipif(not (DATA / "pipeline4_adapters.zip").exists(), reason="adapter zip is published on Hugging Face")
def test_adapter_archive():
    from safetensors import safe_open
    import tempfile
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(DATA / "pipeline4_adapters.zip") as z:
        z.extractall(tmp)
        for folder, n_params, n_tensors in [("pipeline4_restricted_adapter", 208_896, 2),
                                            ("pipeline4_full_adapter", 30_228_480, 504)]:
            cfg = json.load(open(f"{tmp}/{folder}/adapter_config.json"))
            assert cfg["base_model_name_or_path"] == CONFIG["base_model"]["id"] and cfg["r"] == 16
            with safe_open(f"{tmp}/{folder}/adapter_model.safetensors", "numpy") as f:
                shapes = [f.get_slice(k).get_shape() for k in f.keys()]
            total = 0
            for s in shapes:
                n = 1
                for d in s:
                    n *= d
                total += n
            assert (len(shapes), total) == (n_tensors, n_params)


def _observations():
    import pandas as pd
    p = DATA / "dataset2/observations.parquet"
    if not p.exists():
        pytest.skip("Dataset2 table is published on Hugging Face; download it to data/dataset2/ to run")
    return pd.read_parquet(p, columns=["prompt_id", "prompt", "seed_id", "split"])


def test_seed_disjoint_splits():
    from tiny_aya_safety.splits import split_report
    r = split_report(_observations(), DATA)
    ids = r["by_id"]
    assert ids["training_pairs"]["splits"] == {"tune": 195} and ids["training_pairs"]["unmapped"] == 0
    assert ids["validation"]["splits"] == {"analysis": 228}
    assert ids["seed_overlap_training_validation"] == 0
    assert ids["seed_overlap_layer_selection_validation"] == 0
    assert ids["heldout_rows_used"] == 0
    assert r["by_exact_text"]["training_and_validation"] == 0


@pytest.mark.parametrize("text,label,unsafe,refused", [
    ("Safety: Safe\nCategories: None\nRefusal: Yes", "Safe", 0, 1),
    ("Safety: Unsafe\nCategories: Violent\nRefusal: No", "Unsafe", 1, 0),
    ("Safety: Controversial\nRefusal: No", "Controversial", None, 0),
    ("Safety: Safe\nSafety: Unsafe", "Unparsed", None, None),
    ("no verdict", "Unparsed", None, None),
])
def test_parse_guard(text, label, unsafe, refused):
    v = parse_guard(text)
    assert (v["label"], v["unsafe"], v["refused"]) == (label, unsafe, refused)
