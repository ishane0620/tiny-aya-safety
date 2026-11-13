# Data from the original post_train_2 run

Copied unchanged from the research repository (`Rynhalt/ez-penta`, `post_train_2/notebooks/`).
`MANIFEST.sha256` lists the SHA-256 of every file as copied, including the large ones that are not in git.

| Folder | Files | Produced by |
|---|---|---|
| `prompts/` | `pipeline2_prompts_v2.json` (layer-selection prompts), `pipeline3_seed_prompts_v2.json` and `pipeline3_helpful_seed_prompts_v2.json` (training seeds), `pipeline5_validation_prompts_v2.json` (228 validation prompts) | `scripts/build_pipeline2_prompts.py`, `scripts/build_sampling_pools.py` |
| `training/` | `pipeline3_safety_pairs_v2.jsonl` (195 verified pairs), `pipeline3_guard_dropped_v2.jsonl` (37 rejected) | notebook 03 |
| `results/` | `pipeline1_baseline_mh_stats.json`, `pipeline2_results.json`, `pipeline4_results.json`, `pipeline5_results.json`, `pipeline5_answers.jsonl` (684 answers + verdicts) | notebooks 01, 02, 04, 05 |

Not in git (published on Hugging Face; download into these paths to work locally):

| Path | Hugging Face |
|---|---|
| `dataset2/observations.parquet`, `dataset2/matched_observations.parquet` | `tiny-aya-safety-dataset2` (the manifest hashes the original `.csv` versions; the parquet files are converted from them without changes) |
| `pipeline4_adapters.zip` | the two adapter model repos |
