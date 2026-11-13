# Plan: layer-choice control experiment

> **Outcome (2026-11-13):** run as planned (8 arms, Colab G4). Result: the "all layers about the same" row:
> layer 9 matched layer 18, so the benefit comes from the single-matrix restriction, not the layer
> choice. See [RESULTS.md](RESULTS.md#follow-up-layer-choice-control-2026-11-13).

**Question.** The layer-18 adapter matched the broad adapter's safety with half the over-refusal.
Is that because **layer 18** was chosen by activation analysis, or simply because the adapter is
**small** (one matrix, 208,896 parameters)? This is limitation #6 in `docs/LIMITATIONS.md` and the
most obvious objection to the main claim.

**Design.** Train the *same* restricted adapter on other layers and compare. Every `down_proj` in
Tiny Aya has the same shape (11008 → 2048), so every arm has exactly 208,896 trainable parameters.
The only thing that changes is the location.

## 1. Arms

| Arm | Layer | Why | Bootstrap top-5 rate (en / ja) |
|---|---|---|---|
| Selected | **18** | chosen by SafeTune activation analysis | 100% / 100% |
| Runner-up | **15** | second-best by the same analysis | 97% / 97% |
| Control, early-middle | **9** | never in any top-5 resample | 0% / 0% |
| Control, late | **30** | never in any top-5 resample | 0% / 0% |

**Two training seeds per arm** (seeds 0 and 1), so 8 trainings. The published layer-18 adapter
was trained once with no fixed seed (shuffled data order, random LoRA initialisation), so we do not
know how much results vary from run to run. Without this we cannot tell a layer effect from seed
noise. The published adapter is reported alongside as a third layer-18 data point.

**Held fixed (identical to pipeline 4):**
- the 195 training pairs;
- rank 16, alpha 32, dropout 0.05;
- 1 epoch, batch size 1, AdamW, learning rate 1e-4, max length 768;
- bf16 base model at the pinned revision;
- SafeTune `lora_wrap` and `_tokenize_qa_rows`, loss on the answer only.

**Evaluation (identical to pipeline 5):**
- the same 228 validation prompts (analysis split);
- greedy decoding, up to 512 tokens;
- Qwen3Guard judge.

The original model's answers are reused from `pipeline5_answers.jsonl`, not regenerated.

## 2. Metrics and decision rule (fixed before running)

For each arm (mean and spread across seeds):
- unsafe rate on decline prompts (n ≈ 118);
- needless refusals on harmless twins (n = 80);
- refusals on answer-carefully prompts (n = 30);
- average harmless-answer length.

Tests:
- each arm vs the original model: exact McNemar, the same as now;
- **layer 18 vs each control**, prompt by prompt on the same prompts: McNemar on twin refusals and on harmful unsafe answers;
- seed-level bootstrap 95% CIs for the differences.

How we'll read it:

| Outcome | Conclusion we would write |
|---|---|
| Layer 18 (both seeds) has fewer needless refusals than layers 9 and 30 at equal safety, beyond the seed-to-seed spread | Supports activation-guided layer selection |
| Controls fail to remove unsafe answers, while 18 and 15 succeed | Location matters for *whether the fix works*, not only for side effects |
| All four layers perform about the same | The advantage comes from the small adapter, not the layer. The README's claim gets reworded to "a single-matrix LoRA" and the layer analysis is described as not decisive. |

With 80 twins, only differences of roughly 10+ points are detectable. We report effect sizes with
CIs either way and do not over-claim. Note too that this validation set was already used once,
to compare restricted with broad.

## 3. Implementation

| Step | File | What |
|---|---|---|
| 3.1 | `src/tiny_aya_safety/training.py` (new) | Move pipeline 4's `manual_lora_train` here as `train_lora(target_modules, seed, out_dir)`. Uses the config values, sets `torch.manual_seed` plus a seeded `DataLoader` generator, returns the adapter folder and the loss curve. Notebook 04's maths is unchanged. |
| 3.2 | `scripts/run_layer_control.py` (new) | For each (layer, seed): skip it if its answers file already exists (resumable); train; save the adapter; merge; generate the 228 answers; judge; write `outputs/layer_control/answers_L{layer}_s{seed}.jsonl` and the adapter config with the loss curve. One model on the GPU at a time. |
| 3.3 | `notebooks/07_layer_control.ipynb` (new) | A thin Colab wrapper: setup, run the script, show the tables. |
| 3.4 | `src/tiny_aya_safety/metrics.py` | Generalise `headline_table`, `paired` and the seed bootstrap from the fixed three models to any list of arms. Existing outputs must stay identical; the current tests check this. |
| 3.5 | `scripts/analyze_layer_control.py` (new) | Writes `results/layer_control/metrics.json`, `LAYER_CONTROL_TABLES.md` and a figure: needless refusals vs unsafe rate per layer, one point per seed. |
| 3.6 | `tests/` | Unit test for the generalised metrics, plus a check that the analysis script runs on a small synthetic answers file. |

**Avoiding the connection drops that happened twice on Colab:**
- Start the script in the background on the Colab VM (`nohup python scripts/run_layer_control.py &`), then check progress and download each finished arm with short separate commands. A dropped local connection then no longer kills the run.
- Each arm's answers are written as soon as the arm finishes, so a crash loses at most one arm, and rerunning resumes where it stopped.

## 4. Compute

| Item | G4 estimate |
|---|---|
| Base model download and load | ~3 min, once |
| One training (195 steps, one matrix) | ~5 min |
| One evaluation (228 answers + judge) | ~4 min |
| **8 arms in total** | **~75–90 min** |

**G4 only.** On a T4, a single evaluation took more than 45 minutes in our last attempt, so the
full set would take roughly 6 or more hours there. If no G4 is available, use an L4 or A100 rather
than a T4. Check `colab usage` for compute units before starting.

## 5. Publishing and docs

- **Hugging Face:**
  - Add `layer_control/` to `tiny-aya-safety-eval-results`: all 8 answer files, metrics and loss curves.
  - The control adapters are small (0.8 MB each); add them to that dataset repo under `layer_control/adapters/` rather than creating 8 model repos.
- **Repo:**
  - `docs/RESULTS.md`: a new "Layer-choice control" section.
  - README: one paragraph plus the figure, and the headline wording adjusted to whatever the result supports.
  - `docs/LIMITATIONS.md`: update #6 (location vs size) and #8 (single seed).
  - `SUBMISSION.md`: one abstract sentence and a verification-record row.

## 6. Schedule (about half a day)

| When | Task |
|---|---|
| ~2 h | Steps 3.1–3.6, with local tests on CPU using a tiny stand-in model so the loop and resume logic are checked before using GPU time |
| ~1.5 h | The Colab G4 run, mostly unattended. The laptop can sleep once the run is started in the background. |
| ~1.5 h | Analysis, figure, docs and Hugging Face upload; commit and push |

## 7. Risks

| Risk | Mitigation |
|---|---|
| No G4 available | L4 or A100; drop to one seed per layer (~45 min) if time runs short |
| Seed spread larger than the layer differences | That is itself the finding; report it honestly |
| A Colab disconnect | Background run on the VM, resumable per arm |
| Running out of time before the deadline | Keep the submission as it is (already complete); this experiment is an add-on |
