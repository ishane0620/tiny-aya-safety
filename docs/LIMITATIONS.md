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
2. **Heldout performance.** The 60 heldout seeds (4,052 prompts) have not been used. The validation
   set is from the analysis split and was used to compare the two adapters, so it is not an
   unbiased final estimate.
3. **Selection of the validation set.** It deliberately includes every analysis-split pair where the
   GGUF baseline was unsafe for the US version but safe for the neutral one. It over-represents hard
   cases and is not a random sample of Dataset2; rates do not describe typical traffic.
4. **Small numbers.** The safety improvement rests on about 11 baseline unsafe answers among 145
   judged harmful prompts, from 41 source seeds. Confidence intervals are wide.
5. **Helpfulness is measured crudely.** Refusal rates and answer length do not measure whether
   answers are correct or useful, and some harmless twins are themselves wrapped in jailbreak
   framing, where a refusal is arguably reasonable.
6. **Is it the layer or the size?** The restricted adapter differs from the broad one both in
   location (layer 18) and in capacity (145× fewer parameters). Without restricted adapters on other
   layers we cannot attribute the better trade-off to the layer choice itself.
7. **Correlational layer choice.** Layer 18 was selected from activation statistics (refusal
   direction and neuron weights). No ablation, patching or CircuitKIT analysis tested whether it is
   causally necessary. The projection test that US framing lowers the refusal signal was not
   significant on its own (p = 0.12–0.21).
8. **Single run, single seed.** Each adapter was trained once; run-to-run variation is unknown.
9. **Different inference stacks.** The Dataset2 baseline labels come from a GGUF Q4_K_M model with
   a GGUF Q8_0 judge; validation uses bf16 models for both. Only the bf16 runs are compared with each
   other, but the bucket labels used to *select* validation prompts come from the GGUF run.
10. **Scope.** Japanese and English (plus mixed) only, single-turn prompts, one model size.

## Suggested next experiments (not done)
1. Evaluate all three models on the heldout split.
2. Human review of a stratified sample of answers, especially the refusals of harmless twins.
3. Restricted adapters on a few other layers (random and low-ranked) as a location control.
4. Several training seeds per adapter.
