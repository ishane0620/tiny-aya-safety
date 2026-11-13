# Reproducibility

Three levels, from cheapest to most expensive. Levels A and B need nothing from the team's private
Google Drive: data comes from this repository and the project's public Hugging Face datasets, the
base model from Cohere Labs, the adapters from the project's public model repos.

## Accounts
- **Hugging Face:** a token (read access is enough) from an account that has accepted the terms
  on [CohereLabs/tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) (the base model
  is gated, auto-approved, CC BY-NC 4.0). On Colab, add it as a secret named `HF_TOKEN`. Not needed
  for level A.
- **GPU:** levels B and C. Developed on Colab G4 (RTX PRO 6000) and T4.

## A. Check the reported numbers (CPU, ~1 minute)

Colab: open [notebook 06](https://colab.research.google.com/github/ishane0620/tiny-aya-safety/blob/main/notebooks/06_explore_results.ipynb)
and run sections 1–4. Locally:

```bash
git clone https://github.com/ishane0620/tiny-aya-safety && cd tiny-aya-safety
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/compute_metrics.py   # rewrites results/ from data/results/pipeline5_answers.jsonl
git diff --stat results/                      # should show no changes
.venv/bin/python -m pytest -q                 # 35 tests
```

To also run the seed-split test and notebook 01 locally, download the Dataset2 table first:

```bash
.venv/bin/python -c "from huggingface_hub import hf_hub_download as d; import shutil; [shutil.copy(d('<HF_NAMESPACE>/tiny-aya-safety-dataset2', f, repo_type='dataset'), 'data/dataset2/') for f in ['observations.parquet', 'matched_observations.parquet']]"
```

Expected headline (from `results/RESULTS_TABLES.md`): unsafe answers on decline prompts 12/115,
1/117, 1/118; needless refusals 8/80, 16/80, 29/80 (original, restricted, broad).

## B. Regenerate the answers with the published adapters (GPU, ~10 minutes on G4)

Open [notebook 05](https://colab.research.google.com/github/ishane0620/tiny-aya-safety/blob/main/notebooks/05_evaluate_models.ipynb)
on a GPU runtime and run all. It answers the 228 validation prompts with the original model and
with each adapter merged into it (pulled from Hugging Face), judges every answer with Qwen3Guard,
and writes `outputs/pipeline5_answers.jsonl`. Then:

```bash
python scripts/compute_metrics.py outputs/pipeline5_answers.jsonl
```

Greedy decoding is deterministic on the same hardware and software; a different GPU type or
library version can change a few tokens and, occasionally, a verdict. Notebook 06's section 5 runs
the same models on your own prompts.

## C. Retrain everything (GPU; notebook 03 also needs llama.cpp)

| Notebook | Input | Output (in `outputs/`) | Hardware |
|---|---|---|---|
| 01 diagnose | Dataset2 table (HF) | `pipeline1_baseline_mh_stats.json` | CPU |
| 02 find layer | `data/prompts/pipeline2_prompts_v2.json` | `pipeline2_results.json` | GPU (T4 ok) |
| 03 make data | `data/prompts/pipeline3_*seed_prompts_v2.json` | `pipeline3_safety_pairs_v2.jsonl` | `llama-server` binary; CPU works but is slow |
| 04 train | outputs of 02 and 03 (or the published copies) | `pipeline4_{restricted,full}_adapter/` | GPU, < 10 GB |
| 05 evaluate | adapters from 04 if present, else from HF | `pipeline5_answers.jsonl`, `pipeline5_results.json` | GPU |

Each notebook reads a fresh output of the previous step when one exists in `outputs/`, otherwise
the published copy from the original run, so steps can be rerun individually. Set
`TAS_OUTPUT_DIR` to change the output folder and `HF_NAMESPACE` to point at another Hugging Face
account.

**Notebook 03 setup.** CuratorKIT talks to Tiny Aya and Qwen3Guard GGUF models through llama.cpp's
OpenAI-compatible server (`src/tiny_aya_safety/local_server.py` starts and stops it). Install
llama.cpp so that `llama-server` is on `PATH`: `brew install llama.cpp` on macOS, or build it from
https://github.com/ggml-org/llama.cpp on Linux/Colab. Generation uses temperature 0.2, so a rerun
produces different text than the published 195 pairs.

**Running several notebooks in one Colab session.** Restart the runtime between notebooks
(Runtime → Restart session). Notebooks 04 and 05 start with a cell that refuses to continue if an
earlier notebook left a model on the GPU; on a 16 GB T4 two copies of Tiny Aya do not fit.

## Versions

| Package | Version used | Note |
|---|---|---|
| transformers | 5.22.2 | Tiny Aya Water is a `cohere2` model |
| peft | 0.21.2 | the adapters were saved with it |
| torch | 2.15.0 (local), Colab default | |
| torchao | ≥ 0.16 | current peft refuses to import next to Colab's preinstalled 0.10 |
| safetune | 0.1.16 | |
| curatorkit | 0.1.24 | |
| auditkit | 1.1.3 | Dataset2 baseline judging |
| Tiny Aya Water | revision `41a99216d12b3ef4881df762934186e5e4d2bbc1` | |
| Qwen3Guard-Gen-0.6B | revision `fada3b2f655b89601929198343c94cd2f64d93cc` | |

`requirements.txt` covers level A; `requirements-gpu.txt` covers B and C.

## What was actually checked for this release

See [`SUBMISSION.md`](../SUBMISSION.md#verification-record) for the exact list of checks that were
run, where, and which ones were not.
