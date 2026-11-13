"""Publish the adapters, datasets and collections to Hugging Face, then record the real URLs.

    python hf/publish.py --namespace <your-hf-username>            # dry run: stage + check only
    python hf/publish.py --namespace <your-hf-username> --upload   # create repos, upload, collections

Requires `hf auth login` with a write token for that namespace. The adapter zip and the Dataset2
parquet files must exist under data/ (they are not in git). Every staged file is scanned for
secrets before upload. After upload the script checks that each repo is public and un-gated,
writes hf/published.json, sets the namespace in configs/experiment.json, and fills the links in
README.md and docs.
"""
import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "hf"))
from cards import dataset_card, model_card  # noqa: E402

CONFIG_PATH = REPO / "configs" / "experiment.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
NAMES = CONFIG["hf"]
STAGE = REPO / "hf" / "_staging"
SECRET = re.compile(rb"hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}")
DATA = REPO / "data"


def stage(ns):
    shutil.rmtree(STAGE, ignore_errors=True)
    repos = {}
    # --- adapters
    with zipfile.ZipFile(DATA / "pipeline4_adapters.zip") as z:
        for which, folder in [("restricted", "pipeline4_restricted_adapter"), ("broad", "pipeline4_full_adapter")]:
            d = STAGE / NAMES[f"{which}_adapter"]
            d.mkdir(parents=True)
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                (d / f).write_bytes(z.read(f"{folder}/{f}"))
            (d / "README.md").write_text(model_card(which, ns, NAMES), encoding="utf-8")
            shutil.copy(DATA / "results/pipeline4_results.json", d / "training_summary.json")
            repos[NAMES[f"{which}_adapter"]] = ("model", d)
    # --- datasets
    plan = {
        "dataset2_repo": [DATA / "dataset2/observations.parquet", DATA / "dataset2/matched_observations.parquet"],
        "training_repo": [DATA / "training/pipeline3_safety_pairs_v2.jsonl", DATA / "training/pipeline3_guard_dropped_v2.jsonl",
                          DATA / "prompts/pipeline3_seed_prompts_v2.json", DATA / "prompts/pipeline3_helpful_seed_prompts_v2.json"],
        "eval_repo": [DATA / "results/pipeline5_answers.jsonl", DATA / "prompts/pipeline5_validation_prompts_v2.json",
                      REPO / "results/metrics.json", REPO / "results/RESULTS_TABLES.md",
                      DATA / "results/pipeline1_baseline_mh_stats.json", DATA / "results/pipeline2_results.json",
                      DATA / "prompts/pipeline2_prompts_v2.json", DATA / "results/pipeline4_results.json",
                      DATA / "results/pipeline5_results.json", DATA / "MANIFEST.sha256"],
    }
    for key, files in plan.items():
        d = STAGE / NAMES[key]
        d.mkdir(parents=True)
        for f in files:
            shutil.copy(f, d / f.name)
        if key == "eval_repo":
            shutil.copytree(REPO / "results/figures", d / "figures")
            shutil.copytree(REPO / "notebooks/executed", d / "executed_notebooks")
        (d / "README.md").write_text(dataset_card(key, ns, NAMES), encoding="utf-8")
        repos[NAMES[key]] = ("dataset", d)
    # --- static Space: browse the saved answers (no model runs there)
    d = STAGE / NAMES["space"]
    (d / "data").mkdir(parents=True)
    shutil.copy(REPO / "demo/index.html", d / "index.html")
    shutil.copy(REPO / "demo/README.md", d / "README.md")
    shutil.copy(DATA / "results/pipeline5_answers.jsonl", d / "data/pipeline5_answers.jsonl")
    shutil.copy(REPO / "results/metrics.json", d / "data/metrics.json")
    shutil.copytree(REPO / "results/figures", d / "figures")
    repos[NAMES["space"]] = ("space", d)
    return repos


def scan(repos):
    for name, (_, d) in repos.items():
        for f in d.rglob("*"):
            if f.is_file() and SECRET.search(f.read_bytes()):
                raise SystemExit(f"possible secret in {f}; not uploading")
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"  staged {name:42s} {len(list(d.rglob('*.*'))):3d} files  {size / 1e6:8.1f} MB")


def upload(ns, repos):
    from huggingface_hub import HfApi
    api = HfApi()
    urls = {}
    for name, (kind, d) in repos.items():
        rid = f"{ns}/{name}"
        api.create_repo(rid, repo_type=kind, private=False, exist_ok=True,
                        **({"space_sdk": "static"} if kind == "space" else {}))
        api.upload_folder(repo_id=rid, repo_type=kind, folder_path=str(d),
                          commit_message="Publish post_train_2 artifacts (tiny-aya-safety)")
        info = {"model": api.model_info, "dataset": api.dataset_info, "space": api.space_info}[kind](rid)
        assert not info.private and not getattr(info, "gated", False), f"{rid} is not public and un-gated"
        urls[name] = f"https://huggingface.co/{ {'model': '', 'dataset': 'datasets/', 'space': 'spaces/'}[kind] }{rid}"
        print("  published", urls[name])
    cols = {}
    for kind, title, desc in [
        ("model", "Tiny Aya Water safety LoRA adapters",
         "Layer-restricted vs broad LoRA safety fine-tuning of CohereLabs/tiny-aya-water (LEXSI hackathon, Track 2)"),
        ("dataset", "Tiny Aya safety datasets",
         "Dataset2 JA/EN jailbreak prompts, verified training pairs, evaluation answers, metrics and run logs"),
    ]:
        existing = [c for c in api.list_collections(owner=ns, limit=100) if c.title == title]
        col = existing[0] if existing else api.create_collection(title=title, namespace=ns, description=desc, private=False)
        for name, (k, _) in repos.items():
            if k == kind or (kind == "model" and k == "space"):
                api.add_collection_item(col.slug, item_id=f"{ns}/{name}", item_type=k, exists_ok=True)
        cols[kind] = f"https://huggingface.co/collections/{col.slug}"
        print("  collection", cols[kind])
    return urls, cols


def record(ns, urls, cols):
    out = {"namespace": ns, "repos": urls, "collections": cols}
    (REPO / "hf" / "published.json").write_text(json.dumps(out, indent=2) + "\n")
    CONFIG["hf"]["namespace"] = ns
    CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False) + "\n")
    block = "\n".join([
        "<!-- HF_LINKS_START -->",
        f"- **Model collection:** {cols['model']}",
        f"  - restricted adapter (layer 18): {urls[NAMES['restricted_adapter']]}",
        f"  - broad adapter (all layers): {urls[NAMES['broad_adapter']]}",
        f"- **Dataset collection:** {cols['dataset']}",
        f"  - Dataset2 prompts + baseline judgments: {urls[NAMES['dataset2_repo']]}",
        f"  - verified training pairs: {urls[NAMES['training_repo']]}",
        f"  - evaluation answers, metrics, logs: {urls[NAMES['eval_repo']]}",
        f"- **Demo (static Space, saved outputs):** {urls[NAMES['space']]}",
        "<!-- HF_LINKS_END -->"])
    for path in [REPO / "README.md", REPO / "SUBMISSION.md", *(REPO / "docs").glob("*.md")]:
        if not path.exists():
            continue
        s = path.read_text(encoding="utf-8").replace("<HF_NAMESPACE>", ns)
        s = re.sub(r"<!-- HF_LINKS_START -->.*?<!-- HF_LINKS_END -->", block, s, flags=re.S)
        path.write_text(s, encoding="utf-8")
    print("updated hf/published.json, configs/experiment.json, README.md, SUBMISSION.md, docs/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    if args.upload:
        from huggingface_hub import HfApi
        me = HfApi().whoami()
        allowed = {me["name"], *(o["name"] for o in me.get("orgs", []))}
        if args.namespace not in allowed:
            raise SystemExit(f"logged in as {me['name']}; cannot publish to {args.namespace}")
    repos = stage(args.namespace)
    scan(repos)
    if not args.upload:
        print("dry run: nothing uploaded (add --upload)")
        return
    urls, cols = upload(args.namespace, repos)
    record(args.namespace, urls, cols)


if __name__ == "__main__":
    main()
