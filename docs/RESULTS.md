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
| Broad, needless refusals | +26.2 pts | +17.1 to +36.2 | 0 / 16 |

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

It does not show that layer 18 is special. We did not train restricted adapters on other layers, so
some of the advantage may come simply from the much smaller adapter (fewer parameters, less capacity
to overfit 195 examples) rather than from the chosen location. A control with a randomly chosen
layer is the most important missing experiment.

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
