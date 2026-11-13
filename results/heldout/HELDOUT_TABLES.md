# Heldout (final test) results

Qwen3Guard-Gen-0.6B judgments; 1066 heldout prompts per model (representative stratified sample, see docs/HELDOUT_PLAN.md). Not human-reviewed.

## Pure English/Japanese

| Model | Unsafe, decline | Unsafe, all harmful | Needless refusals, twins | Refusals, answer-carefully | Twin answer length |
|---|---:|---:|---:|---:|---:|
| Original | 8.9% (29/326) | 7.0% (30/426) | 5.5% (11/200) | 12.0% (12/100) | 1,032 |
| Restricted L18 (published) | 2.1% (7/331) | 1.9% (8/431) | 26.0% (52/200) | 33.0% (33/100) | 650 |
| Broad (all layers) | 2.4% (8/333) | 1.9% (8/432) | 42.5% (85/200) | 49.0% (49/100) | 279 |
| L9 seed 0 | 3.0% (10/331) | 2.6% (11/431) | 21.5% (43/200) | 29.0% (29/100) | 337 |
| L9 seed 1 | 3.0% (10/330) | 2.6% (11/430) | 16.0% (32/200) | 27.0% (27/100) | 445 |
| L18 seed 0 | 2.1% (7/332) | 1.9% (8/431) | 28.5% (57/200) | 37.0% (37/100) | 631 |
| L18 seed 1 | 3.0% (10/333) | 2.5% (11/433) | 22.5% (45/200) | 29.0% (29/100) | 636 |

## Code-mixed

| Model | Unsafe, decline | Unsafe, all harmful | Needless refusals, twins | Refusals, answer-carefully | Twin answer length |
|---|---:|---:|---:|---:|---:|
| Original | 17.8% (38/214) | 13.9% (38/273) | 3.3% (5/150) | 10.0% (6/60) | 894 |
| Restricted L18 (published) | 5.5% (12/218) | 4.3% (12/277) | 29.3% (44/150) | 31.7% (19/60) | 519 |
| Broad (all layers) | 6.4% (14/220) | 5.4% (15/280) | 38.0% (57/150) | 40.0% (24/60) | 252 |
| L9 seed 0 | 10.1% (22/217) | 8.3% (23/277) | 25.3% (38/150) | 23.3% (14/60) | 297 |
| L9 seed 1 | 9.7% (21/216) | 8.3% (23/276) | 18.7% (28/150) | 23.3% (14/60) | 410 |
| L18 seed 0 | 5.0% (11/218) | 4.0% (11/277) | 30.7% (46/150) | 35.0% (21/60) | 502 |
| L18 seed 1 | 8.2% (18/219) | 6.5% (18/279) | 26.0% (39/150) | 35.0% (21/60) | 546 |

## All heldout

| Model | Unsafe, decline | Unsafe, all harmful | Needless refusals, twins | Refusals, answer-carefully | Twin answer length |
|---|---:|---:|---:|---:|---:|
| Original | 12.4% (67/540) | 9.7% (68/699) | 4.6% (16/350) | 11.3% (18/160) | 973 |
| Restricted L18 (published) | 3.5% (19/549) | 2.8% (20/708) | 27.4% (96/350) | 32.5% (52/160) | 594 |
| Broad (all layers) | 4.0% (22/553) | 3.2% (23/712) | 40.6% (142/350) | 45.6% (73/160) | 267 |
| L9 seed 0 | 5.8% (32/548) | 4.8% (34/708) | 23.1% (81/350) | 26.9% (43/160) | 320 |
| L9 seed 1 | 5.7% (31/546) | 4.8% (34/706) | 17.1% (60/350) | 25.6% (41/160) | 430 |
| L18 seed 0 | 3.3% (18/550) | 2.7% (19/708) | 29.4% (103/350) | 36.3% (58/160) | 576 |
| L18 seed 1 | 5.1% (28/552) | 4.1% (29/712) | 24.0% (84/350) | 31.3% (50/160) | 597 |

## Same prompt, two models (exact McNemar)

| Comparison | Set | Harmful: fixed / broken | p | Twins: newly refused / now answered | p |
|---|---|---:|---:|---:|---:|
| Restricted L18 (published) vs Original | pure | 23 / 1 | 3e-06 | 41 / 0 | 9.1e-13 |
| Restricted L18 (published) vs Original | mixed | 29 / 2 | 4.6e-07 | 39 / 0 | 3.6e-12 |
| Restricted L18 (published) vs Original | all | 52 / 3 | 1.5e-12 | 80 / 0 | 1.7e-24 |
| Broad (all layers) vs Original | pure | 26 / 1 | 4.2e-07 | 74 / 0 | 1.1e-22 |
| Broad (all layers) vs Original | mixed | 31 / 6 | 4.1e-05 | 52 / 0 | 4.4e-16 |
| Broad (all layers) vs Original | all | 57 / 7 | 7.6e-11 | 126 / 0 | 2.4e-38 |
| L9 seed 0 vs Original | pure | 23 / 2 | 1.9e-05 | 32 / 0 | 4.7e-10 |
| L9 seed 0 vs Original | mixed | 23 / 4 | 0.00031 | 33 / 0 | 2.3e-10 |
| L9 seed 0 vs Original | all | 46 / 6 | 1e-08 | 65 / 0 | 5.4e-20 |
| L9 seed 1 vs Original | pure | 24 / 2 | 1e-05 | 21 / 0 | 9.5e-07 |
| L9 seed 1 vs Original | mixed | 23 / 6 | 0.0023 | 23 / 0 | 2.4e-07 |
| L9 seed 1 vs Original | all | 47 / 8 | 8.1e-08 | 44 / 0 | 1.1e-13 |
| L18 seed 0 vs Original | pure | 24 / 1 | 1.5e-06 | 46 / 0 | 2.8e-14 |
| L18 seed 0 vs Original | mixed | 29 / 2 | 4.6e-07 | 41 / 0 | 9.1e-13 |
| L18 seed 0 vs Original | all | 53 / 3 | 8.1e-13 | 87 / 0 | 1.3e-26 |
| L18 seed 1 vs Original | pure | 23 / 2 | 1.9e-05 | 34 / 0 | 1.2e-10 |
| L18 seed 1 vs Original | mixed | 26 / 4 | 5.9e-05 | 34 / 0 | 1.2e-10 |
| L18 seed 1 vs Original | all | 49 / 6 | 1.8e-09 | 68 / 0 | 6.8e-21 |
| Broad (all layers) vs Restricted L18 (published) | pure | 3 / 2 | 1 | 44 / 11 | 8.7e-06 |
| Broad (all layers) vs Restricted L18 (published) | mixed | 5 / 8 | 0.58 | 23 / 10 | 0.035 |
| Broad (all layers) vs Restricted L18 (published) | all | 8 / 10 | 0.81 | 67 / 21 | 9.2e-07 |
| L9 seed 0 vs L18 seed 0 | pure | 2 / 3 | 1 | 8 / 22 | 0.016 |
| L9 seed 0 vs L18 seed 0 | mixed | 2 / 13 | 0.0074 | 4 / 12 | 0.077 |
| L9 seed 0 vs L18 seed 0 | all | 4 / 16 | 0.012 | 12 / 34 | 0.0016 |
| L9 seed 1 vs L18 seed 1 | pure | 5 / 4 | 1 | 4 / 17 | 0.0072 |
| L9 seed 1 vs L18 seed 1 | mixed | 3 / 9 | 0.15 | 0 / 11 | 0.00098 |
| L9 seed 1 vs L18 seed 1 | all | 8 / 13 | 0.38 | 4 / 28 | 1.9e-05 |

## Seed-level robustness (all heldout prompts; 95% CI resampling source seeds)

| Comparison | Metric | prompts / seeds | change | 95% CI | seeds down / up |
|---|---|---:|---:|---:|---:|
| Restricted L18 (published) vs Original | unsafe, decline | 535 / 33 | -9.2 pts | [-14.0, -4.9] | 18 / 2 |
| Restricted L18 (published) vs Original | needless refusals | 350 / 56 | +22.9 pts | [+16.3, +29.8] | 0 / 34 |
| Broad (all layers) vs Original | unsafe, decline | 537 / 33 | -9.3 pts | [-14.6, -4.7] | 16 / 3 |
| Broad (all layers) vs Original | needless refusals | 350 / 56 | +36.0 pts | [+28.7, +43.3] | 0 / 39 |
| L9 seed 0 vs Original | unsafe, decline | 532 / 33 | -7.7 pts | [-11.9, -3.9] | 17 / 1 |
| L9 seed 0 vs Original | needless refusals | 350 / 56 | +18.6 pts | [+12.5, +25.0] | 0 / 28 |
| L9 seed 1 vs Original | unsafe, decline | 531 / 33 | -7.7 pts | [-12.3, -3.9] | 16 / 2 |
| L9 seed 1 vs Original | needless refusals | 350 / 56 | +12.6 pts | [+7.9, +17.7] | 0 / 23 |
| L18 seed 0 vs Original | unsafe, decline | 535 / 33 | -9.3 pts | [-14.3, -5.1] | 18 / 1 |
| L18 seed 0 vs Original | needless refusals | 350 / 56 | +24.9 pts | [+18.1, +32.0] | 0 / 35 |
| L18 seed 1 vs Original | unsafe, decline | 537 / 33 | -8.0 pts | [-12.5, -4.1] | 16 / 2 |
| L18 seed 1 vs Original | needless refusals | 350 / 56 | +19.4 pts | [+13.1, +26.2] | 0 / 28 |
| Broad (all layers) vs Restricted L18 (published) | unsafe, decline | 546 / 33 | +0.4 pts | [-1.2, +1.9] | 4 / 5 |
| Broad (all layers) vs Restricted L18 (published) | needless refusals | 350 / 56 | +13.1 pts | [+7.2, +19.2] | 8 / 27 |
| L9 seed 0 vs L18 seed 0 | unsafe, decline | 543 / 33 | +2.0 pts | [+0.2, +4.4] | 3 / 8 |
| L9 seed 0 vs L18 seed 0 | needless refusals | 350 / 56 | -6.3 pts | [-11.5, -1.4] | 17 / 5 |
| L9 seed 1 vs L18 seed 1 | unsafe, decline | 545 / 33 | +0.6 pts | [-0.9, +2.0] | 5 / 6 |
| L9 seed 1 vs L18 seed 1 | needless refusals | 350 / 56 | -6.9 pts | [-10.7, -3.2] | 18 / 4 |

## US vs neutral framing, pure prompts (Mantel–Haenszel by language)

| Model | odds ratio | p | n |
|---|---:|---:|---:|
| Original | 4.24 | 0.00587 | 426 |
| Restricted L18 (published) | 6.55 | 0.0962 | 431 |
| Broad (all layers) | 1.48 | 0.877 | 432 |
| L9 seed 0 | 8.55 | 0.0346 | 431 |
| L9 seed 1 | 3.80 | 0.154 | 430 |
| L18 seed 0 | 6.42 | 0.108 | 431 |
| L18 seed 1 | 4.20 | 0.0929 | 433 |

## By jailbreak technique (all heldout; harmful unsafe / twins refused)

| Model | control unsafe | rp unsafe | fc unsafe | rs unsafe | control refused | rp refused | fc refused | rs refused |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 7/145 | 14/196 | 31/176 | 16/182 | 5/69 | 6/75 | 0/105 | 5/101 |
| Restricted L18 (published) | 1/145 | 0/202 | 18/179 | 1/182 | 22/69 | 36/75 | 6/105 | 32/101 |
| Broad (all layers) | 0/144 | 3/201 | 16/184 | 4/183 | 31/69 | 34/75 | 29/105 | 48/101 |
| L9 seed 0 | 2/144 | 2/201 | 22/182 | 8/181 | 15/69 | 29/75 | 8/105 | 29/101 |
| L9 seed 1 | 1/144 | 2/201 | 21/178 | 10/183 | 14/69 | 22/75 | 3/105 | 21/101 |
| L18 seed 0 | 1/145 | 0/202 | 17/178 | 1/183 | 25/69 | 37/75 | 5/105 | 36/101 |
| L18 seed 1 | 1/145 | 0/202 | 23/182 | 5/183 | 21/69 | 33/75 | 3/105 | 27/101 |
