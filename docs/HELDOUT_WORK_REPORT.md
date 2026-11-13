# Work report: heldout (final test) evaluation and review

Session of 2026-11-13. This report records **what was done, how, what went wrong along the way, and
what a review found and fixed**. The scientific results are in
[`HELDOUT_FINDINGS.md`](HELDOUT_FINDINGS.md); the pre-registered design is in
[`HELDOUT_PLAN.md`](HELDOUT_PLAN.md).

## 1. Starting point

Two follow-ups had already been done by other sessions:

| Follow-up | Where | Result used here |
|---|---|---|
| Layer-choice control | [`LAYER_CONTROL_FINDINGS.md`](LAYER_CONTROL_FINDINGS.md) | Layers 9 and 18 performed similarly on validation; 15 over-refused more; 30 barely fixed anything. Seeded layer-9 and layer-18 adapters existed in `results/layer_control/adapters/`. |
| Mixed-language evaluation (pipeline 6) | `ez-penta/post_train_2/pipeline6_mixed_language_report.md` | The fix transfers to code-mixed prompts on the analysis split. |

**Task:** evaluate on the untouched heldout split, including code-mixed prompts and the other layers
that performed comparably, with checkpoints along the way.

## 2. What was built

| Step | File | What it does |
|---|---|---|
| Plan, written first | `docs/HELDOUT_PLAN.md` (commit `bd29150`) | Design, models, metrics, primary questions and checkpointing, fixed before any model saw a heldout prompt |
| Prompt set | `scripts/build_heldout_prompts.py` → `data/prompts/heldout_test_prompts.json` | 1,066 prompts, 57 seeds (below) |
| Runner | `scripts/run_heldout_eval.py` | Generates, judges and checkpoints; resumable |
| Analysis | `scripts/analyze_heldout.py` → `results/heldout/` | Tables, metrics JSON, figure |
| Notebook | `notebooks/08_heldout_eval.ipynb` | View the results (CPU) or rerun (GPU) |
| Driver (local, not in repo) | the session scratchpad | Ran and babysat the Colab job |
| Write-up | `docs/HELDOUT_FINDINGS.md`, README section "Final test on the heldout split", `SUBMISSION.md`, `docs/LIMITATIONS.md` | |

### 2.1 Design decisions and why

- **A representative sample, not the validation design.**
  - Validation had been enriched with hard cases: every "US unsafe / neutral safe" pair.
  - The heldout split has only 21 such pairs, too few to copy that design. A final test should also
    estimate typical behaviour.
  - **The set:**
    - every eligible clearly harmful (decline) heldout prompt: 335 pure, 221 mixed;
    - stratified samples of borderline prompts (100 pure, 60 mixed);
    - stratified samples of harmless twins (200 pure, 150 mixed).
  - **Sampling:** strata are language or mix direction × framing × culture; prompts are taken
    round-robin over seeds, with random seed 42.
- **Models, in priority order.** The run is resumable, so later arms were optional if time ran out.
  - The original model, the published restricted (layer 18) and broad adapters: the main claims.
  - Layer 9, seeds 0 and 1: comparable on validation.
  - Layer 18, seeds 0 and 1: a fair same-seed comparison with layer 9.
  - Layers 15 and 30 excluded: worse on validation.
- **Settings identical to pipeline 5**, except the batch size (§3).

### 2.2 Checkpointing
1. **On the VM:** after every 128 prompts, answers are judged by Qwen3Guard and appended to
   `partial_<model>.jsonl` (flushed and fsynced). `progress.json` records counts, speed and ETA.
2. **Resume:** a rerun skips prompt IDs already in the partial file and ignores a line torn by a
   crash. `answers_<model>.jsonl` appears atomically, in prompt-file order, only when a model is
   complete.
3. **On the Mac:** every 4 minutes the driver archived the checkpoint folder on the VM, downloaded
   it and unpacked it into `results/heldout/`, so a lost VM would cost at most one chunk.
4. **Recovery:** if the VM disappeared, the driver would create a new one, re-upload the local
   checkpoints and relaunch (up to 3 times). If the runner process died, it relaunched it.
5. **Cleanup:** the session is always stopped when the driver exits.

## 3. Run log

| Time (EST) | Event |
|---|---|
| 04:57 | G4 unavailable ("Failed to issue request"); fell back to an **A100** |
| 05:02 | **Quick check:** 16 prompts × 3 model types (original, Hugging Face adapter, local layer-9 adapter). All loaded, generated and were judged; 0 unparsed |
| 05:05 | Full run started at generation batch size 8 |
| 05:13 | **Stopped by me:** 2.8 s per prompt would have meant about 6 h of A100 time. Raised the generation batch size to **32**, recorded the deviation in the plan (commit `ecbdac9`) and **discarded the 128 answers already made**, so the whole run uses one setting. No labels had been looked at. |
| 05:17 | Full run restarted on a new A100: 0.6–0.9 s per prompt |
| 05:17–06:54 | 7 models, 10.4–16 min each; checkpoints mirrored every 4 min; no resume needed |
| 06:14 | **Interim checkpoint** (4 models) analysed; it already answered the three primary questions |
| 06:54 | All done; session stopped; integrity checks passed |

**Integrity checks:**
- 7 × 1,066 = **7,462** answers, IDs in prompt order for every model;
- 0 empty answers, 0 unparsed verdicts;
- no tokens in any file;
- each adapter changed nearly all answers: only 5–10 of 1,066 were identical to the original
  model's, which confirms the adapters were applied.

## 4. Headline results (details in HELDOUT_FINDINGS.md)

| All 1,066 heldout prompts | Unsafe, decline | Needless refusals | Harmless answer length |
|---|---:|---:|---:|
| Original | 12.4% | 4.6% | 973 |
| Restricted L18 (published) | 3.5% | 27.4% | 594 |
| Broad | 4.0% | 40.6% | 267 |
| L18 seeds 0 / 1 | 3.3% / 5.1% | 29.4% / 24.0% | 576 / 597 |
| L9 seeds 0 / 1 | 5.8% / 5.7% | 23.1% / 17.1% | 320 / 430 |

1. **The safety gain replicates:** −9.2 points, 95% seed CI −14.0 to −4.9.
2. **Restricted beats broad:** equal safety, 13.1 points fewer needless refusals (CI 7.2–19.2).
3. **The over-refusal cost is larger than on validation:** +22.9 points.
4. **Layer 9 is a different trade-off, not a better fix:** fewer refusals, but less safe on
   code-mixed prompts.
5. **The US-framing effect replicates:** OR 4.24, p = 0.006.

The results are also published at
https://huggingface.co/datasets/isaac369369/tiny-aya-safety-eval-results/tree/main/heldout.

## 5. Review: what was checked, and what was wrong

A review after the run re-read the code and every stated number, and ran a few seconds-long checks.

### 5.1 Checked and correct
- **Every number** in `HELDOUT_FINDINGS.md`, the README section, the abstract and `LIMITATIONS.md`
  matches `results/heldout/HELDOUT_TABLES.md` and `metrics.json`.
- **Comparison labels:** in the paired tables, "fixed / broken" and "newly refused / now answered"
  are oriented correctly, including the layer-9 vs layer-18 rows.
- **Runner:**
  - resume logic and torn-line handling;
  - the progress count;
  - the final file is in prompt order and written atomically;
  - duplicate texts are handled by ID, not text.
- **Separation:**
  - heldout seeds share nothing with any other split (asserted);
  - 0 texts are shared with training, layer selection or validation (asserted);
  - 0 texts and 0 seeds are shared with pipeline 6's mixed prompts (checked).
- **The abstract** is 480 words, under the 500 limit.
- **Tests:** 64 pass.

### 5.2 Found and fixed
| # | Problem | Fix |
|---|---|---|
| 1 | **Confounded comparison.** "Code-mixed prompts were roughly twice as often unsafe (17.8% vs 8.9%)" compares sets with different framings: mixed prompts have no direct requests, which are rarely unsafe (6.2%). | Reworded. Within jailbreak framings only, it's **17.8% vs 10.3%**: higher, but not double. |
| 2 | **Conflicts with pipeline 6.** That analysis-split study concluded the mixed-language weak spot was "largely a quantization effect" (bf16 mixed 11.1% vs pure 10.4%). The heldout bf16 run shows a real mixed gap. | Stated explicitly in the findings: the two samples differ, the size of the gap is uncertain, and on heldout it is present. |
| 3 | **Overstatement.** "Every adapter nearly eliminates unsafe answers under role-play, refusal suppression and direct requests." Layer 9 still leaves 8–10 under refusal suppression. | Restricted the claim to the layer-18 and broad adapters (0–5 per framing). |
| 4 | **Stale claim.** The README summary and `LIMITATIONS` #6 still said layer 9 "did as well as layer 18", which the heldout result contradicts (it's less safe on code-mixed prompts). | Both now say: comparable on validation; on heldout a different trade-off, not a better fix. |
| 5 | **Overstated check.** The plan and findings said the builder *asserts* no overlap with the mixed-language evaluation. It doesn't; that file is in the research repo. | Wording corrected; the separate check (0 texts, 0 seeds) is recorded. The plan edit is marked as made after the run, and git history keeps the original. |
| 6 | **Missing caveats.** The heldout set's composition differs from validation (Japan-framed prompts, more fiction and refusal-suppression twins), and 23 prompt texts appear twice (Dataset2's `duplicate_text_flag`, same labels). | Both added to the findings' caveats. Restricting to US/neutral twins gives the same over-refusal picture (5.2% → 26.7%), so it isn't a composition artifact. The duplicates are about 2% of the set. |

### 5.3 Known, not fixed (follow-ups)
- **Notebook 07 bug** (from the layer-control session): it shows the repo's tables even when
  `RUN = True`, instead of the freshly computed ones. Notebook 08 doesn't have this bug.
- **Hugging Face cards:** the `eval-results` dataset card doesn't yet describe the new `heldout/`
  folder, and the adapter model cards still report only validation numbers.
- **The mixed-language report (pipeline 6)** lives only in the research repo (`ez-penta`), not in
  `tiny-aya-safety`.
- **Driver log filter (cosmetic):** it showed only the last 4 log lines per poll, so "COMPLETE"
  lines were sometimes hidden. Progress was read from `progress.json` instead.
- **No human review** of any judgments.

## 6. Reproduce

```bash
# Re-analyse (CPU, seconds)
python scripts/analyze_heldout.py results/heldout

# Rebuild the prompt set (needs data/dataset2/observations.parquet from the HF dataset)
python scripts/build_heldout_prompts.py

# Rerun everything (GPU: A100/G4/L4, ~1.7 h on an A100; HF_TOKEN with Tiny Aya access).
# Checkpointed and resumable:
python scripts/run_heldout_eval.py --out outputs/heldout
```

Or open `notebooks/08_heldout_eval.ipynb` in Colab (`RUN = True` to rerun).
