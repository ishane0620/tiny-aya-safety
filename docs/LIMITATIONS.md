# Limitations and open questions

What is directly reproducible from the published files, and what still needs work.

## Directly reproducible now
- Every reported rate, count, test and figure, recomputed from the saved answers and verdicts
  (`scripts/compute_metrics.py`, notebook 06, `pytest`).
- Pipeline 1's culture-effect statistics, recomputed from the published Dataset2 table (notebook 01
  reproduces all 16 values of the original run exactly).
- The seed-level train/validation separation, re-derived from the published files by prompt ID and
  by exact text (`tests/test_data_integrity.py`).
- The adapters' structure: targets, ranks, parameter counts and compatibility with the base model's
  configuration.

## Not yet established
1. **Human validation.** Every safety and refusal label comes from one automatic judge,
   Qwen3Guard-Gen-0.6B. No human has reviewed the 684 validation answers. Judge errors apply to all
   three models alike, which protects the comparisons more than the absolute rates.
2. **Heldout performance (now measured).** The final test on 1,066 heldout prompts
   (`docs/HELDOUT_FINDINGS.md`) confirms the validation findings' direction but shows a larger
   over-refusal cost (27% of harmless prompts refused by the restricted adapter, 41% by the broad).
   It samples the borderline prompts and twins rather than using all of them, and used generation
   batch size 32 instead of 8.
3. **Selection of the validation set.** It deliberately includes every analysis-split pair where the
   GGUF baseline was unsafe for the US version but safe for the neutral one. It over-represents hard
   cases and is not a random sample of Dataset2; rates do not describe typical traffic.
4. **Small numbers.** The safety improvement rests on about 11 baseline unsafe answers among 145
   judged harmful prompts, from 41 source seeds. Confidence intervals are wide.
5. **Helpfulness is measured crudely.** Refusal rates and answer length do not measure whether
   answers are correct or useful, and some harmless twins are themselves wrapped in jailbreak
   framing, where a refusal is arguably reasonable.
6. **It is the size, not the layer (tested).** The follow-up control trained the same one-matrix
   adapter on layers 9, 15, 18 and 30. Middle layers performed similarly, and layer 9 (never ranked
   by the activation analysis) did as well as layer 18. The advantage over broad fine-tuning comes
   from restricting the update to one middle-layer matrix; the activation analysis did not pick a
   uniquely good layer. See docs/RESULTS.md. On the heldout test, layer 9 refused fewer harmless
   prompts than layer 18 (same seeds) but was less safe on code-mixed prompts, so the middle layers
   are different trade-offs rather than equivalent (docs/HELDOUT_FINDINGS.md).
7. **Correlational layer choice.** Layer 18 was selected from activation statistics (refusal
   direction and neuron weights). No ablation, patching or CircuitKIT analysis tested whether it is
   causally necessary. The projection test that US framing lowers the refusal signal was not
   significant on its own (p = 0.12–0.21).
8. **Seed variation.** The published adapters were each trained once without a fixed seed. Two
   seeded retrains of the layer-18 adapter gave 1.7–3.4% unsafe and 22.5–25.0% needless refusals,
   so the published layer-18 numbers (0.9%, 20.0%) are a favourable draw. The broad adapter was not
   retrained.
9. **Different inference stacks.** The Dataset2 baseline labels come from a GGUF Q4_K_M model with
   a GGUF Q8_0 judge; validation uses bf16 models for both. Only the bf16 runs are compared with each
   other, but the bucket labels used to *select* validation prompts come from the GGUF run.
10. **Scope.** Japanese and English (plus mixed) only, single-turn prompts, one model size.

## Suggested next experiments (not done)
1. Human review of a stratified sample of answers, especially the refusals of harmless twins.
2. More seeds for the broad adapter, and a sweep of adapter rank, to map how capacity (rather than
   location) drives the safety vs over-refusal trade-off.
3. Causal tests (ablation or patching) of the refusal direction, since activation ranking did not
   identify a uniquely effective layer.
