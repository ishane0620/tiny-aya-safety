# Heldout (final test) evaluation: plan

Written 2026-11-13, **before** any model saw a heldout prompt. Nothing in the project (data, layer
choice, adapters, settings) was tuned on the heldout split; this evaluation must not change any of
them afterwards either. Results go to [`HELDOUT_FINDINGS.md`](HELDOUT_FINDINGS.md).

## Why
All earlier numbers come from the `analysis` split, which was also used to compare adapters and
layers. The 60 `heldout` seeds (20% of Dataset2, split by seed and category before any modelling)
have never been used. They give an unbiased final estimate.

## Prompts: a representative sample, not the enriched validation design
The validation set was enriched with hard cases (every "US unsafe / neutral safe" pair). The heldout
split has only 21 such pairs, too few to copy that design, and a final test should estimate typical
behaviour anyway. So the test set is a stratified sample of **all eligible heldout prompts**
(`analysis_eligible`, generation and scoring completed in the Dataset2 baseline run; 2,022 prompts,
58 seeds):

| Set | Harmful, decline | Harmful, answer carefully | Harmless twins |
|---|---:|---:|---:|
| Pure English/Japanese | **all 335** | 100 of 230 | 200 of 641 |
| Code-mixed | **all 221** | 60 of 164 | 150 of 431 |

Samples are drawn with a fixed random seed (42), allocated in proportion across language (or mix
direction) × framing × culture, and spread across seeds (each seed's first prompt before any seed's
second). Builder: `scripts/build_heldout_prompts.py`; it asserts heldout-only rows, and no prompt
text shared with training, layer selection or validation. (Overlap with the mixed-language
evaluation, whose prompt file lives in the research repo, was checked separately after the run:
0 texts, 0 seeds.)

## Models (priority order; the run is resumable, so later arms are optional if time runs out)
1. `baseline`: Tiny Aya Water @ `41a99216d12b`
2. `restricted`: + published layer-18 adapter (`isaac369369/tiny-aya-water-safety-lora-layer18`)
3. `broad`: + published all-layer adapter (`isaac369369/tiny-aya-water-safety-lora-broad`)
4. `L9_s0`, `L9_s1`: layer-9 one-matrix adapters from the layer-control study (comparable safety,
   fewer needless refusals than layer 18 on validation)
5. `L18_s0`, `L18_s1`: the seeded layer-18 retrains, for a same-seed comparison with layer 9

Layers 15 and 30 are not evaluated: on validation, 15 over-refused more with shorter answers, and 30
did not remove unsafe answers.

## Settings (identical to pipeline 5)
Greedy decoding, up to 512 new tokens, chat template, left padding, prompts cut at
1,024 tokens, bf16; adapters merged into the base weights. Judge: Qwen3Guard-Gen-0.6B @
`fada3b2f655b`, bf16, greedy, 64 tokens, no truncation.

**Deviation (decided after a speed check, before any result was looked at):** generation batch size
32 instead of pipeline 5's 8. At batch 8 the A100 needed 2.8 s per prompt (~6 h for all models);
batch size only changes how prompts are padded together, and the same setting is used for every
model here. The 128 answers already generated at batch 8 were discarded so the run is uniform.

## Metrics (fixed now)
Per model and per set (pure, mixed, both):
1. Unsafe rate on decline prompts (Unsafe ÷ Safe+Unsafe).
2. Needless-refusal rate on harmless twins.
3. Refusal rate on answer-carefully prompts.
4. Mean answer length on harmless twins.

Tests: exact McNemar vs the original model, and broad vs restricted, on the same prompts; seed-level
bootstrap 95% CIs (resampling source seeds) for the changes in 1 and 2; Mantel–Haenszel US-vs-neutral
odds ratio (pure set, by language) for the original model.

**Primary questions:**
1. Does the restricted adapter's safety gain hold on unseen seeds?
2. Does it still over-refuse less than the broad adapter?
3. Does layer 9 match layer 18 on unseen seeds?

## Checkpointing
The runner writes every judged batch to `<out>/partial_<model>.jsonl` as it goes and resumes from
those files. A model's final `answers_<model>.jsonl` appears only when all its prompts are done.
`progress.json` records counts, timing and ETA. A local poller downloads progress and all
partial/final files every few minutes, so a lost Colab runtime loses at most one batch.
