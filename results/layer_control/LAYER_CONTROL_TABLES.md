# Layer-choice control

Same one-matrix LoRA (`layers.N.mlp.down_proj`, 208,896 trainable parameters), same 195 training pairs and settings, trained on different layers with two seeds each; evaluated on the same 228 validation prompts (Qwen3Guard judgments, analysis split).

## Every arm

| Model | Unsafe, decline prompts | Needless refusals, twins | Refusals, answer-carefully | Harmless answer length | Train loss (first → last quarter) |
|---|---:|---:|---:|---:|---:|
| Original model | 10.4% (12/115) | 10.0% (8/80) | 30.0% (9/30) | 1,072 | – |
| Layer 18, published adapter (original run) | 0.9% (1/117) | 20.0% (16/80) | 46.7% (14/30) | 700 | – |
| Layer 18 (selected (activation analysis)), seed 0 | 1.7% (2/117) | 25.0% (20/80) | 50.0% (15/30) | 691 | 1.20 → 0.70 |
| Layer 18 (selected (activation analysis)), seed 1 | 3.4% (4/118) | 22.5% (18/80) | 40.0% (12/30) | 650 | 1.23 → 0.69 |
| Layer 9 (control (never top-5)), seed 0 | 4.2% (5/118) | 17.5% (14/80) | 40.0% (12/30) | 327 | 1.21 → 0.66 |
| Layer 9 (control (never top-5)), seed 1 | 1.7% (2/116) | 18.8% (15/80) | 30.0% (9/30) | 457 | 1.22 → 0.63 |
| Layer 15 (runner-up), seed 0 | 1.7% (2/117) | 28.8% (23/80) | 43.3% (13/30) | 377 | 1.21 → 0.65 |
| Layer 15 (runner-up), seed 1 | 2.5% (3/118) | 28.8% (23/80) | 40.0% (12/30) | 343 | 1.24 → 0.64 |
| Layer 30 (control (never top-5)), seed 0 | 7.8% (9/116) | 13.8% (11/80) | 30.0% (9/30) | 1,018 | 1.18 → 0.85 |
| Layer 30 (control (never top-5)), seed 1 | 6.0% (7/116) | 12.5% (10/80) | 30.0% (9/30) | 1,033 | 1.21 → 0.84 |
| Broad LoRA (all layers), reference | 0.8% (1/118) | 36.3% (29/80) | 43.3% (13/30) | 292 | – |

## Per layer (mean across seeds, [min, max])

| Layer | Role | Unsafe, decline | Needless refusals, twins | Refusals, answer-carefully | Answer length |
|---:|---|---:|---:|---:|---:|
| 18 | selected (activation analysis) | 2.5% [1.7%, 3.4%] | 23.8% [22.5%, 25.0%] | 45.0% [40.0%, 50.0%] | 670 [650, 691] |
| 9 | control (never top-5) | 3.0% [1.7%, 4.2%] | 18.1% [17.5%, 18.8%] | 35.0% [30.0%, 40.0%] | 392 [327, 457] |
| 15 | runner-up | 2.1% [1.7%, 2.5%] | 28.8% [28.8%, 28.8%] | 41.7% [40.0%, 43.3%] | 360 [343, 377] |
| 30 | control (never top-5) | 6.9% [6.0%, 7.8%] | 13.1% [12.5%, 13.8%] | 30.0% [30.0%, 30.0%] | 1,025 [1,018, 1,033] |

## Each arm vs the original model (same prompts, exact McNemar)

| Arm | harmful: fixed / broken | p | twins: newly refused / now answered | p |
|---|---:|---:|---:|---:|
| L18_published | 10 / 0 | 0.002 | 9 / 1 | 0.021 |
| L18_s0 | 10 / 0 | 0.002 | 12 / 0 | 0.00049 |
| L18_s1 | 8 / 0 | 0.0078 | 11 / 1 | 0.0063 |
| L9_s0 | 10 / 1 | 0.012 | 6 / 0 | 0.031 |
| L9_s1 | 9 / 0 | 0.0039 | 7 / 0 | 0.016 |
| L15_s0 | 11 / 0 | 0.00098 | 15 / 0 | 6.1e-05 |
| L15_s1 | 10 / 0 | 0.002 | 16 / 1 | 0.00027 |
| L30_s0 | 5 / 2 | 0.45 | 3 / 0 | 0.25 |
| L30_s1 | 5 / 0 | 0.062 | 3 / 1 | 0.62 |

## Each other layer vs layer 18 with the same seed (same prompts)

Twins: prompts the other layer refuses but layer 18 answers, and the reverse. Positive change = the other layer refuses more harmless prompts than layer 18.

| Arm | vs | twins: refuses where L18 answers / answers where L18 refuses | p | change in twin refusal rate [seed-bootstrap 95% CI] | harmful: unsafe where L18 safe / safe where L18 unsafe |
|---|---|---:|---:|---:|---:|
| L9_s0 | L18_s0 | 2 / 8 | 0.11 | -7.5 pts [-16.2, +0.0] | 2 / 0 |
| L9_s1 | L18_s1 | 3 / 6 | 0.51 | -3.8 pts [-12.2, +3.8] | 0 / 1 |
| L15_s0 | L18_s0 | 6 / 3 | 0.51 | +3.7 pts [-3.6, +11.1] | 0 / 0 |
| L15_s1 | L18_s1 | 8 / 3 | 0.23 | +6.2 pts [-2.2, +14.5] | 1 / 2 |
| L30_s0 | L18_s0 | 0 / 9 | 0.0039 | -11.2 pts [-18.8, -4.8] | 7 / 0 |
| L30_s1 | L18_s1 | 1 / 9 | 0.021 | -10.0 pts [-18.2, -2.5] | 3 / 0 |
