# LEXSI AI Safety & Alignment Hackathon 2026: submission

**Project:** Tiny Aya Safety: layer-restricted LoRA for multilingual jailbreak robustness
**Team:** Isaac Shane, Marcus Izumi
**Track:** 2: AI Red Teaming, Safety and Alignment
**Problem statements:** multilingual and code-mixed safety (2a); safety-preserving fine-tuning (2b)
**Model:** `CohereLabs/tiny-aya-water` (Tiny Aya Water, 3.35B)
**Toolkit repositories integrated:** AuditKIT (baseline judging), SafeTune (refusal-direction layer selection, LoRA wrapping and tokenization), CuratorKIT (training-data generation)

## Links

- **GitHub (code, notebooks, write-up):** https://github.com/ishane0620/tiny-aya-safety
- **Start here (Colab, CPU):** https://colab.research.google.com/github/ishane0620/tiny-aya-safety/blob/main/notebooks/06_explore_results.ipynb
<!-- HF_LINKS_START -->
- **Model collection:** https://huggingface.co/collections/isaac369369/tiny-aya-water-safety-lora-adapters-6af671c237fdafbed3b7362a
  - restricted adapter (layer 18): https://huggingface.co/isaac369369/tiny-aya-water-safety-lora-layer18
  - broad adapter (all layers): https://huggingface.co/isaac369369/tiny-aya-water-safety-lora-broad
- **Dataset collection:** https://huggingface.co/collections/isaac369369/tiny-aya-safety-datasets-6af671c33b325a1767adb0a5
  - Dataset2 prompts + baseline judgments: https://huggingface.co/datasets/isaac369369/tiny-aya-safety-dataset2
  - verified training pairs: https://huggingface.co/datasets/isaac369369/tiny-aya-safety-training-pairs
  - evaluation answers, metrics, logs: https://huggingface.co/datasets/isaac369369/tiny-aya-safety-eval-results
- **Demo (static Space, saved outputs):** https://huggingface.co/spaces/isaac369369/tiny-aya-safety-explorer
<!-- HF_LINKS_END -->

## Abstract

Small multilingual models are usually safety-tuned and tested in English. We built Dataset2, a
controlled set of 21,184 Japanese, English and code-mixed jailbreak prompts generated from 320
source requests, each rewritten with neutral, Japanese or American cultural cues, four framings
(direct, role-play, fiction, refusal suppression) and a matched harmless twin. Judging Tiny Aya
Water's answers with Qwen3Guard through AuditKIT, we found that American cultural framing roughly
doubled the odds of an unsafe answer (Mantel–Haenszel OR 1.98, 95% CI 1.58–2.48, stratified by
language), consistently in English and Japanese.

We then asked whether activation-guided, layer-restricted fine-tuning can repair such failures
with less collateral damage than broad fine-tuning. Using SafeTune, we extracted a refusal
direction (mean harmful minus harmless activation) at every layer and ranked layers by
safety-neuron alignment; layer 18 was in the top five in 30 of 30 bootstrap resamples in both
languages. With CuratorKIT, Tiny Aya wrote refusals for harmful prompts and helpful answers for
their harmless twins; a Qwen3Guard gate kept 195 verified examples. We trained two LoRA adapters on
identical data and settings: a restricted adapter on layer 18's MLP output matrix only (208,896
parameters) and a broad adapter on every projection of all 36 layers (30.2 million).

Seeds were split once: training used only "tune" seeds; validation used 228 prompts from disjoint
"analysis" seeds; a heldout split remains untouched. On harmful prompts that should be declined,
unsafe answers fell from 10.4% to 0.9% (restricted) and 0.8% (broad); prompt by prompt, the
restricted adapter fixed 10 unsafe answers and broke none (exact McNemar p = 0.002). The adapters
differ in collateral damage: needless refusals of harmless look-alike prompts rose from 10% to 20%
with the restricted adapter but to 36% with the broad one (direct comparison p = 0.004), and
harmless answers kept 65% versus 27% of their original length. Treating each source request as one
unit, the restricted adapter's safety gain is −8.8 points (95% CI −14.8 to −3.8) and its
over-refusal cost +10 points (CI +2.5 to +19.1), against +26 points (CI +17 to +36) for the broad
adapter.

On this validation sample, confining the update to one activation-selected layer gave the same
safety gain at about half the over-refusal cost, with an adapter 145 times smaller. These are
automatic-judge results on a validation set enriched for hard cases, not human-reviewed or heldout
results, and we have not yet separated the effect of layer choice from adapter size. All code,
adapters, data, answers and executed notebooks are public, and every number can be recomputed on a
CPU from the released answers.

## Official requirements checklist

| Requirement (submission page) | Status | Where |
|---|---|---|
| Registered team member's email | Entered by the team on the form | — |
| Abstract under 500 words | ✅ | above |
| Public, un-gated HF **model** collection | see links | publish script checks `private=False`, `gated=False` |
| Public HF **dataset** collection (datasets, logs, traces, curated data) | see links | Dataset2, training pairs, eval answers + metrics + executed notebooks |
| Write-up / article | ✅ GitHub README + `docs/` | [README](README.md) |
| GitHub repo with code/notebooks and reproducibility notes | ✅ | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| Repos and models used, track, problem statements | ✅ | header of this file |
| Public deployment (demo or repo link) | ✅ repo + Colab notebook 06 | links above |
| Before-and-after metrics, more than one | ✅ unsafe rate, needless refusals, borderline refusals, answer length, paired tests, culture-gap OR | [results](results/RESULTS_TABLES.md) |
| ≥ 2 integrated track repos (3+ preferred) | ✅ 3: AuditKIT, SafeTune, CuratorKIT | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) |
| Tiny Aya / Aya model | ✅ Tiny Aya Water | |
| ROI comparison (optional) | ✅ brief: 145× fewer trainable parameters, 0.8 MB vs 121 MB, < 10 GB GPU, ~15 min training on one G4 | [docs/RESULTS.md](docs/RESULTS.md#cost) |

## Verification record

Checks actually run for this release (2026-11-12/13):

| Check | Where | Result |
|---|---|---|
| Artifacts are the post_train_2 run's | local, SHA-256 vs research repo and team Drive copies | identical; manifest in `data/MANIFEST.sha256` |
| All headline numbers recomputed from saved answers | clean Python 3.12 venv, `requirements.txt` only | match the original pipeline 5 output exactly |
| Pipeline 1 statistics recomputed from the published Dataset2 table | notebook 01, local CPU | 16/16 values identical |
| Prompt selection rebuilt from the published Dataset2 table | `scripts/build_sampling_pools.py`, `scripts/build_pipeline2_prompts.py` | all 4 prompt files byte-identical to the originals |
| Seed-disjoint splits | `tests/test_data_integrity.py` | training = 195 tune rows, validation = 228 analysis rows, 0 shared seeds, 0 shared texts, 0 heldout rows |
| Adapter structure vs base config | safetensors + `AutoConfig` (revision pinned) | shapes, targets, 208,896 / 30,228,480 parameters |
| Unit and integration tests | `pytest`, clean venv | 35 passed |
| Notebooks 01 and 06 (sections 1–4) execute | local CPU, `nbconvert --execute` | no errors |
| Notebook 06 in a **fresh Colab runtime** (clones this public repo, no local files) | Colab CPU runtime via Colab CLI | 13/13 cells, 0 errors, headline numbers identical |
| GPU smoke test of published adapters | see below | *pending* |
| Notebooks 02, 03, 04, 05 re-executed after migration | — | **not re-run**; they differ from the executed originals only in file locations and (05) adapter loading |
| Human review of judgments | — | **not done** |
| Heldout evaluation | — | **not done** |
