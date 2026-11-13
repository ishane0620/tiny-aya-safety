# Experiment log: how post_train_2 came to be

This repository publishes the second complete run (`post_train_2`) of the pipeline. The full
development history, including the first run, lives in the team's research repository
[`Rynhalt/ez-penta`](https://github.com/Rynhalt/ez-penta) (folders `post_train_1/` and
`post_train_2/`). The changes between runs explain several design choices here.

## post_train_1 (superseded, not published here)
- 22 training pairs; validation on 40 prompts; 4-bit quantized training (QLoRA).
- **Found problems:**
  - CuratorKIT ran in `single_call` mode, in which the custom refusal/compliance prompts are
    ignored. The "chosen" answers were thorough compliant answers, not refusals.
  - Validation prompts were split by prompt ID only, so rewordings of the same source request
    appeared in training and validation.
  - The 4-bit route had been adopted to dodge an out-of-memory error whose real cause was a memory
    bug in SafeTune's `PlainSFTTrainer` (documented as `safetune_bug.md` in the research repo).

## post_train_2 (this repository)
- **Seed-level splits:** training from `tune` seeds, validation from `analysis` seeds, `heldout`
  untouched.
- **No quantization:** bf16 for layer selection, training and validation; the trainer is a plain
  PyTorch loop, so the SafeTune trainer bug does not apply.
- **`two_pass` CuratorKIT generation plus a Qwen3Guard gate**, so every kept refusal really refuses.
- **Helpful-answer pairs added** after an intermediate refusal-only iteration over-refused badly
  (needless refusals 10% → 46% for the restricted adapter on that iteration's validation set, which
  still shared seeds with training). Adding helpful pairs brought this to 20% on the clean split.
- **Layer selection with bootstrap stability** over 48–60 prompts per group (previously 8), which
  changed the target from layers {18, 23} to layer 18 alone.
- **Two operational failures fixed during the final run:**
  - On a 16 GB T4, notebook 4 inherited notebook 2's model from the shared Colab kernel and ran out
    of memory. The run chain now restarts the kernel between notebooks, and the notebooks check for
    leftover GPU memory.
  - The final run then completed on a G4 (RTX PRO 6000).

## Run record (final, 2026-11-12)

| Step | Where | Output |
|---|---|---|
| 1 | earlier, CPU | `pipeline1_baseline_mh_stats.json` |
| 2 | Colab T4 | `pipeline2_results.json` (layer 18) |
| 3 | local Mac, llama.cpp (CPU) | 195 verified pairs |
| 4 | Colab G4, ~15 min | two adapters (`pipeline4_adapters.zip`) |
| 5 | Colab G4, ~8 min | `pipeline5_answers.jsonl`, `pipeline5_results.json` |

SHA-256 hashes of every artifact are in [`data/MANIFEST.sha256`](../data/MANIFEST.sha256). The
executed notebooks with their outputs are in [`notebooks/executed/`](../notebooks/executed/).
