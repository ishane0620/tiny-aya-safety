"""Experiment settings and file locations, shared by every notebook and script.

Nothing here hard-codes a Colab path. Files are resolved in this order:
1. the repository's own `data/` folder (small artifacts are committed there);
2. the project's public Hugging Face dataset repos (large Dataset2 tables, logs);
Outputs go to `$TAS_OUTPUT_DIR`, else `/content/outputs` on Colab, else `<repo>/outputs`.
Set `HF_NAMESPACE` to point at a different Hugging Face account or organization.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG = json.loads((REPO / "configs" / "experiment.json").read_text(encoding="utf-8"))
DATA = REPO / "data"

# Which HF dataset repo holds each file that is too large for git.
_HF_DATASET_FILES = {
    "observations.parquet": "dataset2_repo",
    "matched_observations.parquet": "dataset2_repo",
}


def hf_namespace() -> str:
    ns = os.environ.get("HF_NAMESPACE") or CONFIG["hf"]["namespace"]
    if ns == "HF_NAMESPACE_NOT_SET":
        raise RuntimeError("Set CONFIG['hf']['namespace'] in configs/experiment.json or the HF_NAMESPACE env var.")
    return ns


def hf_repo(key: str) -> str:
    """Full repo id, e.g. hf_repo('restricted_adapter') -> '<namespace>/tiny-aya-water-safety-lora-layer18'."""
    return f"{hf_namespace()}/{CONFIG['hf'][key]}"


def data_path(name: str) -> Path:
    """Local path of a project file, downloading it from Hugging Face if it is not in the repo."""
    for sub in ("", "prompts", "training", "results", "dataset2"):
        p = DATA / sub / name
        if p.exists():
            return p
    if name in _HF_DATASET_FILES:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(hf_repo(_HF_DATASET_FILES[name]), name, repo_type="dataset"))
    raise FileNotFoundError(f"{name} is not in {DATA} and is not a known Hugging Face dataset file")


def output_dir() -> Path:
    if os.environ.get("TAS_OUTPUT_DIR"):
        out = Path(os.environ["TAS_OUTPUT_DIR"])
    elif Path("/content").exists():
        out = Path("/content/outputs")
    else:
        out = REPO / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def artifact(name: str) -> Path:
    """An intermediate result (e.g. pipeline2_results.json): the copy an earlier notebook wrote
    in this environment if there is one, otherwise the published copy from the original run."""
    fresh = output_dir() / name
    return fresh if fresh.exists() else data_path(name)


def adapter_source(which: str) -> str:
    """Where to load an adapter from: a freshly trained local folder if pipeline 4 ran in this
    environment, otherwise the published Hugging Face repo. which = 'restricted' or 'broad'."""
    local = output_dir() / {"restricted": "pipeline4_restricted_adapter", "broad": "pipeline4_full_adapter"}[which]
    if (local / "adapter_config.json").exists():
        return str(local)
    return hf_repo(f"{which}_adapter")
