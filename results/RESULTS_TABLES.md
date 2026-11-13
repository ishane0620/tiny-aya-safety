# Results tables (recomputed from `pipeline5_answers.jsonl`)

Qwen3Guard-Gen-0.6B judgments on the 228 seed-disjoint validation prompts (`analysis` split).
Not human-validated, and not the untouched `heldout` split.

## Headline

| Metric | Original | Restricted LoRA | Broad LoRA |
|---|---:|---:|---:|
| Unsafe answers, harmful prompts that should be declined | 10.4% (12/115) | 0.9% (1/117) | 0.8% (1/118) |
| Needless refusals, harmless twins | 10.0% (8/80) | 20.0% (16/80) | 36.3% (29/80) |
| Refusals, borderline answer-carefully prompts | 30.0% (9/30) | 46.7% (14/30) | 43.3% (13/30) |
| Average answer length on harmless twins (characters) | 1,072 | 700 | 292 |

Unsafe-rate denominators exclude answers labelled Controversial or Unparsed (original 3, restricted 1, broad 0 among the 118 decline prompts).

## Paired before/after (same prompt, two models; exact McNemar test)

| Comparison | n | improved | worsened | unchanged | p |
|---|---:|---:|---:|---:|---:|
| restricted: harmful prompts, unsafe -> safe vs safe -> unsafe | 144 | 10 | 0 | 134 | 0.002 |
| restricted: harmless twins, refused -> answered vs answered -> refused | 80 | 1 | 9 | 70 | 0.021 |
| broad: harmful prompts, unsafe -> safe vs safe -> unsafe | 145 | 11 | 0 | 134 | 0.00098 |
| broad: harmless twins, refused -> answered vs answered -> refused | 80 | 1 | 22 | 57 | 5.7e-06 |
| broad vs restricted: harmless twins (worsened = broad refuses, restricted answers) | 80 | 3 | 16 | 61 | 0.0044 |

## Seed-level robustness

Several validation prompts are rewrites of one source seed, so prompt-level tests overstate the sample size. Here seeds are the unit: a 95% bootstrap interval resampling seeds, and a sign test counting seeds whose rate went down vs up.

| Comparison | prompts / seeds | change vs original | 95% CI (seed bootstrap) | seeds down / up | sign-test p |
|---|---:|---:|---:|---:|---:|
| restricted: unsafe rate, decline prompts | 114 / 41 | -8.8 pts | [-14.8, -3.8] | 9 / 0 | 0.0039 |
| restricted: refusal rate, harmless twins | 80 / 33 | +10.0 pts | [+2.5, +19.1] | 1 / 8 | 0.039 |
| broad: unsafe rate, decline prompts | 115 / 41 | -9.6 pts | [-14.7, -4.9] | 11 / 0 | 0.00098 |
| broad: refusal rate, harmless twins | 80 / 33 | +26.3 pts | [+17.1, +36.3] | 0 / 16 | 3.1e-05 |

## US vs neutral framing (Mantel-Haenszel, stratified by language, harmful prompts)

| Model | odds ratio | p | n |
|---|---:|---:|---:|
| baseline | 5.70 | 0.0347 | 145 |
| restricted | undefined (no unsafe answers left in one condition) | 0.986 | 147 |
| broad | undefined (no unsafe answers left in one condition) | 1 | 148 |

## By jailbreak technique (harmful: unsafe answers; twins: refusals)

| Technique | baseline harmful unsafe | restricted harmful unsafe | broad harmful unsafe | baseline twin refused | restricted twin refused | broad twin refused |
|---|---:|---:|---:|---:|---:|---:|
| control | 7/52 | 1/52 | 1/52 | 3/36 | 7/36 | 11/36 |
| rp | 2/56 | 0/58 | 0/58 | 3/12 | 4/12 | 4/12 |
| fc | 3/15 | 0/15 | 0/16 | 0/18 | 0/18 | 8/18 |
| rs | 0/22 | 0/22 | 0/22 | 2/14 | 5/14 | 6/14 |

## By language

| Language | baseline harmful unsafe | restricted harmful unsafe | broad harmful unsafe | baseline twin refused | restricted twin refused | broad twin refused |
|---|---:|---:|---:|---:|---:|---:|
| en | 4/58 | 1/59 | 0/60 | 3/44 | 6/44 | 20/44 |
| ja | 8/87 | 0/88 | 1/88 | 5/36 | 10/36 | 9/36 |
