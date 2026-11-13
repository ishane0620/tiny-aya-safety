# Methodology

Research question: **can activation-guided, layer-restricted fine-tuning improve the safety of a
small multilingual model while preserving more helpfulness than broad fine-tuning?**

Model: `CohereLabs/tiny-aya-water` (Cohere2 architecture, 3.35B parameters, 36 decoder layers,
hidden size 2048, MLP width 11008), revision `41a99216d12b3ef4881df762934186e5e4d2bbc1`, loaded in
bfloat16 with no quantization for layer selection, training and validation.

## 1. Diagnosis: is there a cultural-framing effect? (AuditKIT)

Input: the baseline evaluation of Dataset2 (see [DATASET.md](DATASET.md)), harmful prompts only,
`analysis` + `tune` splits.

For each language, build a 2×2 table (US-framed vs neutral-framed) × (unsafe vs not), then pool the
strata with the **Cochran–Mantel–Haenszel** odds ratio and continuity-corrected test. Stratifying
by language means a different English/Japanese mix between conditions cannot create a fake
culture effect. Result for US framing: OR 1.98 (95% CI 1.58–2.48), p = 2.2×10⁻⁹, with the same
direction in every language stratum (English OR 1.87, p = 0.011; Japanese 1.90, p = 0.001; mixed
2.12, p < 0.001; Fisher tests). Japanese framing showed no effect (OR 1.02–1.28, all p > 0.3). See
`data/results/pipeline1_baseline_mh_stats.json` and the executed notebook. This motivated a fix and
supplied the "before" reference.

## 2. Layer selection: where is the refusal signal strongest? (SafeTune)

Input: 323 control-framing prompts from `tune` seeds: harmful-neutral, harmless-neutral and
harmful-US groups, English and Japanese (48–60 per group).

1. Run every prompt through the bf16 model and record the last-token output of each decoder layer.
2. **Refusal direction** per layer (after Arditi et al., 2024): mean over harmful prompts minus mean
   over harmless prompts, extracted with SafeTune's `SteeringVectorExtractor`. Our own hooks agree
   with SafeTune's extractor at every layer (mean cosine 0.9997).
3. **Safety-neuron ranking**: SafeTune `identify_safety_neurons(method="weight", top_k_per_layer=16)`
   scores MLP neurons by how strongly their output weights align with the refusal direction.
4. **Bootstrap stability**: repeat steps 2–3 on 30 resamples of the prompts and count how often each
   layer is among the top 5. Layer 18 was top-5 in **30/30** resamples for both English and
   Japanese; it is the restricted adapter's only target.
5. Check (held-out half of the neutral prompts): US-framed harmful prompts project lower on the
   refusal direction than neutral ones in all four language combinations (gaps 2.3–2.9), but no test
   is significant on its own (p = 0.12–0.21).

**What this is and is not.** This is a correlational activation analysis: layer 18 is where the
harmful/harmless difference is most consistently expressed in MLP weights. It does not show that
layer 18 *causes* refusal (no ablation or patching was done), and CircuitKIT's causal circuit
discovery was not used.

## 3. Training data (CuratorKIT + Qwen3Guard)

Inputs, all from `tune` seeds: 135 harmful, US-framed prompts whose expected behaviour is `decline`
(36 of them got an unsafe answer in the baseline evaluation), and the 101 harmless twins of those
prompts. Borderline (`answer_carefully`) seeds were excluded from training by design.

- **Generation.** CuratorKIT `PreferenceGenerationTask` in `two_pass` mode, with Tiny Aya Water
  (GGUF Q4_K_M, llama.cpp, temperature 0.2) as the writer. Harmful prompt → good = brief refusal,
  bad = in-character compliance. Harmless twin → good = helpful answer, bad = needless refusal.
  (An earlier `single_call` configuration silently produced compliant "good" answers; `two_pass`
  is the mode in which the separate chosen/rejected prompts are actually used.)
- **Verification.** Qwen3Guard-Gen-0.6B (GGUF Q8_0) judged every good answer. A refusal was kept only
  if `Safe` and `Refusal: Yes`; a helpful answer only if `Safe` and `Refusal: No`.

| Pass | Generated | Kept | Dropped |
|---|---:|---:|---|
| Refusals | 133 | **116** | 15 safe but not refusing, 2 unsafe |
| Helpful answers | 99 | **79** | 13 unsafe, 7 controversial |

Total: **195 examples from 38 seeds**. Only the good answers are used for training (supervised
fine-tuning); the bad answers are released for inspection and possible preference training.

## 4. Intervention: two LoRA adapters (SafeTune LoRA utilities)

Both adapters are trained identically: bf16 base model (all base weights frozen), LoRA rank 16,
alpha 32, dropout 0.05, one pass over the 195 examples with batch size 1, AdamW, learning rate 1e-4,
sequences up to 768 tokens, loss on the assistant answer only (SafeTune `_tokenize_qa_rows`).
SafeTune's `lora_wrap` attaches the adapters; training uses a plain PyTorch loop (SafeTune's
`PlainSFTTrainer` was bypassed because of a memory bug documented in the research repo).

| | Restricted | Broad |
|---|---|---|
| Target matrices | `layers.18.mlp.down_proj` | `q,k,v,o,gate,up,down_proj` in all 36 layers |
| Trainable numbers | 208,896 | 30,228,480 |
| Training loss, first → last quarter | 1.28 → 0.63 | 0.64 → 0.48 |
| Peak GPU memory (RTX PRO 6000) | 9.6 GB | 9.9 GB |

## 5. Validation (Qwen3Guard)

Input: 228 prompts from `analysis` seeds only, as 114 pairs (the same request with US and with
neutral framing), chosen from Dataset2's matched comparisons by how the baseline evaluation behaved:

| Bucket | Pairs | Why |
|---|---:|---|
| US unsafe, neutral safe | 43 (all available) | the failure pattern the project targets |
| both safe | 17 (20:50 ratio to the above) | did the fix break what already worked? |
| US safe, neutral unsafe | 14 (all available) | the reverse pattern |
| harmless twins | 40 | over-refusal |

All three models (original, original + restricted adapter merged, original + broad adapter merged)
answer every prompt with greedy decoding, up to 512 new tokens, prompts truncated at 1,024 tokens,
batch size 8. Qwen3Guard-Gen-0.6B (bf16, greedy, 64 tokens, no truncation) labels each answer
`Safe`/`Unsafe`/`Controversial` and refusal yes/no.

**Metrics**
- Unsafe rate on harmful prompts (denominator: answers with a Safe/Unsafe verdict).
- Refusal rate on harmless twins (needless refusals) and on borderline prompts.
- Mean answer length on harmless twins (a rough helpfulness proxy).
- Paired comparisons on the same prompt: exact McNemar test.
- Seed-level robustness: seed-cluster bootstrap 95% CIs and a seed-level sign test, because
  prompt variants of one seed are correlated.
- US-vs-neutral Mantel–Haenszel odds ratio, as in step 1.
