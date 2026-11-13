"""One-off migration of the five post_train_2 notebooks from the ez-penta research repo.

- notebooks/executed/<original name>.ipynb : byte-identical copies with the outputs of the
  2026-11-12 run (provenance; not meant to be rerun as-is).
- notebooks/0N_<name>.ipynb : runnable versions. Outputs cleared; the only code changes are
  file locations (config-driven instead of /content or a sibling folder), one setup cell, and
  loading the trained adapters from Hugging Face in step 5. Every replacement is asserted.

Usage: python scripts/migrate_notebooks.py <path to ez-penta/post_train_2/notebooks>
"""
import json
import shutil
import sys
import uuid
from pathlib import Path

SRC = Path(sys.argv[1])
REPO = Path(__file__).resolve().parents[1]
NB = REPO / "notebooks"

TARGETS = {
    "pipeline1_stratified_diagnosis": "01_diagnose_cultural_safety",
    "pipeline2_circuit_discovery": "02_identify_safety_layers",
    "pipeline3_curatorkit_data": "03_generate_training_data",
    "pipeline4_circuit_restricted_fix": "04_train_lora_adapters",
    "pipeline5_validation": "05_evaluate_models",
}

SETUP = '''# Project setup (Colab or local): fetch this repository's code and small data files, then load
# the shared config. Large files come from the project's Hugging Face datasets on first use.
import subprocess
import sys
from pathlib import Path

if Path("/content").exists():                     # Google Colab
    REPO = Path("/content/tiny-aya-safety")
    if not REPO.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/ishane0620/tiny-aya-safety", str(REPO)], check=True)
else:                                             # local checkout: this notebook is in <repo>/notebooks
    REPO = Path.cwd().resolve().parent
sys.path.insert(0, str(REPO / "src"))
from tiny_aya_safety.config import CONFIG, adapter_source, artifact, data_path, output_dir

OUT = output_dir()   # everything this notebook writes goes here (override with TAS_OUTPUT_DIR)
print("repo:", REPO, "| outputs:", OUT)'''


def note(executed_name, extra=""):
    return (f"> **Migrated from the `ez-penta` research repo (experiment `post_train_2`).** The executed "
            f"original with its outputs from the 2026-11-12 run is "
            f"[`executed/{executed_name}.ipynb`](executed/{executed_name}.ipynb). This copy only changes "
            f"where files are read and written (see `configs/experiment.json`){extra}.")


def cell(kind, src):
    c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {},
         "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def sub(nb, idx, old, new, count=1):
    s = "".join(nb["cells"][idx]["source"])
    assert s.count(old) == count, f"cell {idx}: expected {count}x {old[:60]!r}, found {s.count(old)}"
    nb["cells"][idx]["source"] = s.replace(old, new).splitlines(keepends=True)


def replace_cell(nb, idx, new_src, must_contain):
    s = "".join(nb["cells"][idx]["source"])
    assert must_contain in s, f"cell {idx} is not the expected cell"
    nb["cells"][idx]["source"] = new_src.strip("\n").splitlines(keepends=True)


def migrate(name, nb):
    if name == "pipeline1_stratified_diagnosis":
        replace_cell(nb, 2, '''
import numpy as np
import pandas as pd
import scipy.stats as stats

# Dataset2 baseline evaluation: every prompt with Tiny Aya Water's original answer and the
# AuditKIT/Qwen3Guard verdict (published as the project's Dataset2 Hugging Face dataset).
observations = pd.read_parquet(data_path("observations.parquet"))
usable = observations[
    (observations["analysis_eligible"] == True)
    & (observations["generation_status"] == "completed")
    & (observations["scoring_status"] == "completed")
    & (observations["split"] != "heldout")
].copy()
harmful = usable[usable["kind"] == "harmful"].copy()
harmful["unsafe"] = harmful["safety"].eq("Unsafe").astype(int)
print(f"{len(harmful)} usable harmful-seed rows")''', 'observations = pd.read_csv(COMBINED_DIR / "observations.csv"')
        sub(nb, 12, 'out_path = COMBINED_DIR / "pipeline1_baseline_mh_stats.json"',
            'out_path = OUT / "pipeline1_baseline_mh_stats.json"')
        return 1
    if name == "pipeline2_circuit_discovery":
        sub(nb, 6, 'open("/content/pipeline2_prompts_v2.json"', 'open(data_path("pipeline2_prompts_v2.json")')
        sub(nb, 16, 'open("/content/pipeline2_results.json", "w"', 'open(OUT / "pipeline2_results.json", "w"')
        return 1
    if name == "pipeline3_curatorkit_data":
        sub(nb, 2, '''ROOT = Path.cwd()
while not (ROOT / "chat.py").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import chat
import local_server
''', '''# llama.cpp server helpers (needs a `llama-server` binary on PATH; see docs/REPRODUCIBILITY.md).
from tiny_aya_safety import chat, local_server
''')
        for f in ("pipeline3_seed_prompts_v2.json", "pipeline3_helpful_seed_prompts_v2.json"):
            sub(nb, 4, f'ROOT / "post_train_2" / "notebooks" / "{f}"', f'data_path("{f}")')
            sub(nb, 6, f'str(ROOT / "post_train_2" / "notebooks" / "{f}")', f'str(data_path("{f}"))')
        for d in ("pipeline3_output_v2", "pipeline3_output_v2_helpful"):
            sub(nb, 6, f'str(ROOT / "post_train_2" / "notebooks" / "combined" / "{d}")', f'str(OUT / "{d}")')
        sub(nb, 8, 'ROOT / "runs" / "pipeline3-aya.log"', 'OUT / "logs" / "pipeline3-aya.log"')
        sub(nb, 12, 'ROOT / "runs" / "pipeline3-guard.log"', 'OUT / "logs" / "pipeline3-guard.log"')
        sub(nb, 12, 'guard_log.unlink()', 'guard_log.unlink()\nguard_log.parent.mkdir(parents=True, exist_ok=True)')
        for f in ("pipeline3_safety_pairs_v2.jsonl", "pipeline3_guard_dropped_v2.jsonl"):
            sub(nb, 14, f'ROOT / "post_train_2" / "notebooks" / "combined" / "{f}"', f'OUT / "{f}"')
        return 1
    if name == "pipeline4_circuit_restricted_fix":
        sub(nb, 6, 'open("/content/pipeline2_results.json"', 'open(artifact("pipeline2_results.json")')
        sub(nb, 6, 'open("/content/pipeline3_safety_pairs_v2.jsonl"', 'open(artifact("pipeline3_safety_pairs_v2.jsonl")')
        sub(nb, 12, '"/content/pipeline4_restricted_adapter"', 'str(OUT / "pipeline4_restricted_adapter")')
        sub(nb, 12, 'out_dir="/content"', 'out_dir=str(OUT)')
        sub(nb, 14, '"/content/pipeline4_full_adapter"', 'str(OUT / "pipeline4_full_adapter")')
        sub(nb, 14, 'out_dir="/content"', 'out_dir=str(OUT)')
        sub(nb, 16, 'open("/content/pipeline4_results.json", "w"', 'open(OUT / "pipeline4_results.json", "w"')
        return 3   # after the "start from an empty GPU" cell, which runs %reset
    if name == "pipeline5_validation":
        sub(nb, 4, '"transformers", "accelerate", "scipy"',
            '"transformers", "accelerate", "scipy", "-U", "peft", "torchao"')
        sub(nb, 6, 'open("/content/pipeline5_validation_prompts_v2.json"', 'open(data_path("pipeline5_validation_prompts_v2.json")')
        sub(nb, 6, 'open("/content/pipeline1_baseline_mh_stats.json"', 'open(artifact("pipeline1_baseline_mh_stats.json")')
        replace_cell(nb, 12, '''
from peft import PeftModel

# The original run loaded merged checkpoints that only existed on its Colab VM. Here each fixed
# model is rebuilt the same way pipeline 4 built them: base model + trained LoRA adapter, merged.
# adapter_source() uses adapters trained in this environment by notebook 04 if present,
# otherwise the published Hugging Face adapter repos.
answers = {"baseline": baseline_answers}
for arm in ["restricted", "broad"]:
    source = adapter_source(arm)
    print(f"Loading {arm}: {MODEL_ID} + adapter {source}")
    arm_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=CONFIG["base_model"]["revision"],
                                                     dtype=torch.bfloat16, device_map=device)
    arm_model = PeftModel.from_pretrained(arm_model, source).merge_and_unload()
    arm_model.eval()
    answers[arm] = generate_answers(arm_model, tokenizer, prompt_texts)
    print(f"Sample {arm} answer:", answers[arm][0][:200])
    del arm_model
    gc.collect()
    torch.cuda.empty_cache()''', 'arm_checkpoints = {')
        sub(nb, 14, 'val_df.to_json("/content/pipeline5_answers.jsonl"', 'val_df.to_json(OUT / "pipeline5_answers.jsonl"')
        sub(nb, 14, 'print("Saved every answer + verdict to /content/pipeline5_answers.jsonl")',
            'print(f"Saved every answer + verdict to {OUT / \'pipeline5_answers.jsonl\'}")')
        sub(nb, 24, 'open("/content/pipeline5_results.json", "w"', 'open(OUT / "pipeline5_results.json", "w"')
        return 3
    raise KeyError(name)


def main():
    (NB / "executed").mkdir(parents=True, exist_ok=True)
    for name, target in TARGETS.items():
        src = SRC / f"{name}.ipynb"
        shutil.copyfile(src, NB / "executed" / src.name)
        nb = json.loads(src.read_text(encoding="utf-8"))
        insert_at = migrate(name, nb)
        for c in nb["cells"]:
            c.setdefault("id", uuid.uuid4().hex[:8])
            if c["cell_type"] == "code":
                c["outputs"], c["execution_count"] = [], None
        extra = ", and loads the trained adapters from Hugging Face" if name == "pipeline5_validation" else ""
        nb["cells"][insert_at:insert_at] = [cell("markdown", note(name, extra)), cell("code", SETUP)]
        nb["metadata"].update({"accelerator": "GPU", "colab": {"provenance": []}})
        (NB / f"{target}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{name} -> {target}.ipynb (setup inserted at cell {insert_at})")


if __name__ == "__main__":
    main()
