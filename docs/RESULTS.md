# Results and interpretation

All numbers below are recomputed by `scripts/compute_metrics.py` from
`data/results/pipeline5_answers.jsonl` (every answer and verdict of the original run); the full
tables are in [`results/RESULTS_TABLES.md`](../results/RESULTS_TABLES.md) and the machine-readable
version in [`results/metrics.json`](../results/metrics.json). They are **Qwen3Guard judgments on the
validation split**, not human-validated measurements and not heldout results.

## What we found

**1. Both adapters remove nearly all unsafe answers.** On harmful prompts that should be declined,
unsafe answers fell from 12/115 (10.4%) to 1/117 (0.9%) with the restricted adapter and 1/118 (0.8%)
with the broad one. Prompt by prompt, the restricted adapter turned 10 unsafe answers safe and made
no safe answer unsafe (exact McNemar p = 0.002); the broad adapter, 11 and 0 (p = 0.001). Each
adapter left exactly one unsafe answer, on a different prompt (restricted: English; broad:
Japanese; both direct, US-framed requests), and both of those prompts were unsafe at baseline too.

**2. The difference is collateral damage.** On harmless twins, needless refusals went from 8/80
(10%) to 16/80 (20%) with the restricted adapter and 29/80 (36%) with the broad one. Directly
compared, the broad adapter refuses 16 harmless prompts the restricted one answers, against 3 the
other way round (p = 0.004). Answers to harmless prompts shrank from 1,072 characters on average to
700 (restricted) and 292 (broad).

**3. The effect survives seed-level analysis.** Treating each source request as one unit:

| | change vs original | 95% CI | seeds improved / worsened |
|---|---:|---:|---:|
| Restricted, unsafe rate (decline prompts, 41 seeds) | −8.8 pts | −14.8 to −3.8 | 9 / 0 |
| Broad, unsafe rate | −9.6 pts | −14.7 to −4.9 | 11 / 0 |
| Restricted, needless refusals (33 seeds) | +10.0 pts | +2.5 to +19.1 | 1 / 8 |
| Broad, needless refusals | +26.3 pts | +17.1 to +36.3 | 0 / 16 |

**4. The cultural gap disappears, because unsafe answers disappear.** In the validation sample the
original model's US-framed prompts were more often unsafe than neutral ones (Mantel–Haenszel OR 5.7,
p = 0.035). After either fix there are too few unsafe answers left for an odds ratio. This is not
evidence that the culture-specific mechanism was repaired, only that the failures it produced are gone.

**5. Borderline requests became more cautious.** Refusals on `answer_carefully` prompts (where a
careful answer is preferred) rose from 30% to 47% (restricted) and 43% (broad), even though no
borderline prompts were in the training data.

## How to read the trade-off

Safety does not separate the two adapters; helpfulness does. Changing 0.7% as many parameters, in one
layer chosen by activation analysis, bought the same reduction in unsafe answers with half the
increase in needless refusals and much longer helpful answers. This supports, on this sample, the
idea that a targeted intervention generalizes "refuse harmful requests" with less spill-over into
"refuse everything that looks like a harmful request".

It does not show that layer 18 is special, and the follow-up control below shows it is not: the
advantage comes from restricting the update to one middle-layer matrix.

## Follow-up: layer-choice control (2026-11-13)

Plan and decision rule, written before running: [`LAYER_CONTROL_PLAN.md`](LAYER_CONTROL_PLAN.md).
The same one-matrix adapter (`layers.N.mlp.down_proj`, 208,896 parameters for every N), the same 195
pairs and settings, trained on four layers with two fixed seeds each and evaluated on the same 228
prompts (Colab G4; ~3 minutes per arm). Full tables:
[`results/layer_control/LAYER_CONTROL_TABLES.md`](../results/layer_control/LAYER_CONTROL_TABLES.md).

| Layer | Role | Unsafe, decline (seed 0 / 1) | Needless refusals (seed 0 / 1) | Answer length (mean) | Unsafe answers fixed vs original (seed 0 / 1, of 12) |
|---:|---|---:|---:|---:|---:|
| 18 | top-ranked by activation analysis | 1.7% / 3.4% | 25.0% / 22.5% | 670 | 10 / 8 |
| 15 | runner-up | 1.7% / 2.5% | 28.8% / 28.8% | 360 | 11 / 10 |
| 9 | never in the top 5 | 4.2% / 1.7% | 17.5% / 18.8% | 392 | 10 / 9 |
| 30 | never in the top 5 | 7.8% / 6.0% | 13.8% / 12.5% | 1,025 | 5 / 5 |
| 18 (published, no fixed seed) | | 0.9% | 20.0% | 700 | 10 |
| broad (all layers) | reference | 0.8% | 36.3% | 292 | 11 |

**Reading against the pre-registered rule.** Layer 18 did *not* have fewer needless refusals than the
controls at equal safety: layer 9 had fewer (prompt-level comparisons with the same seed, p = 0.11
and 0.51; seed-bootstrap differences −7.5 and −3.8 points, CIs touching zero) at similar safety.
Layer 30 refused least but was a weak fix (5 of 12 unsafe answers fixed, not significant against the
original model), sitting further along the same trade-off rather than beating it. The outcome is
the "same performance" row of the plan: **the benefit over broad fine-tuning comes from the small,
single-matrix update in a middle layer, not from selecting layer 18 by activation analysis.**

Two further observations:
- All middle-layer adapters kept over-refusal below the broad adapter's, so the main comparison
  (one matrix vs all layers) holds up across layers and seeds.
- The published layer-18 adapter, trained once without a fixed seed, was the most favourable of
  three layer-18 trainings. Its headline numbers (0.9% unsafe, 20% needless refusals) should be read
  as one draw; the two seeded retrains averaged 2.5% and 23.8%.

## Breakdown highlights

- **By jailbreak technique (twins refused, original → restricted → broad):** fiction 0 → 0 → 8 of
  18; refusal suppression 2 → 5 → 6 of 14; role-play 3 → 4 → 4 of 12; direct 3 → 7 → 11 of 36. The broad
  adapter's extra refusals are concentrated in fiction-framed harmless prompts.
- **By language (twins refused):** English 3 → 6 → 20 of 44; Japanese 5 → 10 → 9 of 36. The restricted
  adapter's over-refusal falls more on Japanese; the broad adapter's on English.
- **Remaining unsafe answers:** restricted 1 (English, direct framing), broad 1 (Japanese, direct).

## Cost

Both adapters trained in about 15 minutes in total on one Colab G4 GPU (RTX PRO 6000) with under
10 GB of GPU memory each, so either fits a 16 GB T4. The restricted adapter is 0.8 MB on disk against
121 MB for the broad one, and can be merged into the base weights for zero inference overhead.
