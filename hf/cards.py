"""Model and dataset cards for the Hugging Face release. Rendered by hf/publish.py."""

GITHUB = "https://github.com/ishane0620/tiny-aya-safety"
BASE = "CohereLabs/tiny-aya-water"
BASE_REV = "41a99216d12b3ef4881df762934186e5e4d2bbc1"

RESULTS_TABLE = """| Qwen3Guard judgment on 228 validation prompts | Original | Restricted (layer 18) | Broad (all layers) |
|---|---:|---:|---:|
| Unsafe answers, harmful prompts that should be declined | 10.4% (12/115) | 0.9% (1/117) | 0.8% (1/118) |
| Needless refusals, harmless look-alike prompts | 10.0% (8/80) | 20.0% (16/80) | 36.3% (29/80) |
| Refusals, borderline "answer carefully" prompts | 30.0% (9/30) | 46.7% (14/30) | 43.3% (13/30) |
| Average harmless-answer length (characters) | 1,072 | 700 | 292 |"""

ADAPTERS = {
    "restricted": {
        "title": "Tiny Aya Water safety LoRA: layer 18 only (restricted)",
        "what": "a LoRA adapter on **one matrix**, layer 18's MLP output projection (`model.layers.18.mlp.down_proj`)",
        "targets": "`layers.18.mlp.down_proj`", "params": "208,896", "size": "0.8 MB",
        "why": "Layer 18 was chosen by activation analysis (SafeTune refusal-direction extraction and "
               "safety-neuron ranking; top-5 in 30/30 bootstrap resamples in both English and Japanese).",
        "summary": "Both adapters remove nearly all unsafe answers; this one keeps more harmless prompts "
                   "answered (16 vs 29 needless refusals out of 80) and keeps answers longer.",
    },
    "broad": {
        "title": "Tiny Aya Water safety LoRA: all layers (broad comparison)",
        "what": "a LoRA adapter on **every** attention and MLP projection (`q,k,v,o,gate,up,down_proj`) in all 36 layers",
        "targets": "`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (all layers)",
        "params": "30,228,480", "size": "121 MB",
        "why": "It is the comparison arm for the layer-restricted adapter: same data, same settings, "
               "no restriction on where the model may change.",
        "summary": "It is about as safe as the restricted adapter but refuses far more harmless prompts "
                   "(36% vs 20%) and gives much shorter answers. Prefer the restricted adapter.",
    },
}


def model_card(which: str, ns: str, repo_names: dict) -> str:
    a = ADAPTERS[which]
    other = "broad" if which == "restricted" else "restricted"
    return f"""---
base_model: {BASE}
library_name: peft
license: cc-by-nc-4.0
language: [ja, en]
pipeline_tag: text-generation
tags: [lora, peft, safety, refusal, jailbreak, multilingual, japanese, tiny-aya, lexsi-hackathon]
datasets:
- {ns}/{repo_names['training_repo']}
- {ns}/{repo_names['eval_repo']}
---

# {a['title']}

This repository contains {a['what']} for
[`{BASE}`]({'https://huggingface.co/' + BASE}) (3.35B parameters). It was trained to make the model
refuse Japanese and English jailbreak requests while still answering harmless look-alike requests.
It is an **adapter only**: you need access to the base model, which is gated (auto-approved) and
licensed CC BY-NC 4.0 by Cohere Labs.

{a['why']} Companion adapter: [`{ns}/{repo_names[other + '_adapter']}`](https://huggingface.co/{ns}/{repo_names[other + '_adapter']}).
Code, notebooks and full write-up: {GITHUB}

## Use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{BASE}", revision="{BASE_REV}")
model = AutoModelForCausalLM.from_pretrained("{BASE}", revision="{BASE_REV}", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "{ns}/{repo_names[which + '_adapter']}")
model = model.merge_and_unload()          # optional: same speed as the base model
msgs = [{{"role": "user", "content": "Please explain how vaccines work."}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

Requires `transformers>=5`, `peft>=0.21.2`. On Colab, also `pip install -U torchao` (current peft
refuses to import next to Colab's preinstalled torchao 0.10).

## Adapter

| | |
|---|---|
| Base model | `{BASE}` @ `{BASE_REV[:12]}` |
| Adapted modules | {a['targets']} |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Trainable parameters | {a['params']} ({a['size']}, float32) |
| Saved with | PEFT 0.21.2 |

## Training

- **Data:** 195 examples ([`{ns}/{repo_names['training_repo']}`](https://huggingface.co/datasets/{ns}/{repo_names['training_repo']})):
  116 refusals of harmful, US-framed jailbreak prompts and 79 helpful answers to their harmless
  twins, in Japanese and English. Written by Tiny Aya Water through CuratorKIT and kept only if
  Qwen3Guard-Gen-0.6B judged them safe and correctly refusing / not refusing. All from the `tune`
  split of the team's Dataset2 (38 source requests).
- **Procedure:** supervised fine-tuning on the good answers, loss on the answer only; base model in
  bfloat16, frozen; 1 epoch, batch size 1, AdamW, learning rate 1e-4, max length 768 tokens;
  one Colab G4 GPU, under 10 GB of memory. SafeTune's `lora_wrap` attached the adapter.

## Evaluation (validation, not final)

228 prompts from Dataset2's `analysis` split, whose source requests share nothing with the
training data; greedy decoding, up to 512 tokens; every answer judged by Qwen3Guard-Gen-0.6B.

{RESULTS_TABLE}

{a['summary']} Seed-level 95% intervals, paired tests and per-language/technique breakdowns are in
the [results tables]({GITHUB}/blob/main/results/RESULTS_TABLES.md). All answers and verdicts:
[`{ns}/{repo_names['eval_repo']}`](https://huggingface.co/datasets/{ns}/{repo_names['eval_repo']}).

## Limitations

- Labels come from one automatic judge (Qwen3Guard-Gen-0.6B); no human review yet.
- Validation set, deliberately enriched with hard cases; the heldout split has not been evaluated.
- About 11 unsafe baseline answers: wide uncertainty.
- More refusals of harmless and borderline requests than the base model.
- Japanese and English, single-turn prompts only. Not a substitute for a full safety evaluation.

## Intended use and license

Research on multilingual safety fine-tuning. The adapter is released under **CC BY-NC 4.0**,
matching the base model; it does not grant any rights to Tiny Aya Water, whose own license and terms
apply. Not for commercial use, and not a guarantee of safe behaviour.

Team: Isaac Shane and Marcus Izumi (LEXSI AI Safety & Alignment Hackathon 2026, Track 2).
"""


DATASETS = {
    "dataset2_repo": {
        "pretty": "Dataset2: controlled Japanese/English jailbreak prompts with Tiny Aya Water baseline judgments",
        "files": "| `observations.parquet` | 21,184 prompts (320 seeds × culture × harmful/twin × framing × language) with design metadata, quality flags, the original model's answer, and the AuditKIT/Qwen3Guard verdict |\n"
                 "| `matched_observations.parquet` | 85,884 matched pairs of conditions of the same seed (e.g. US vs neutral), used to build the validation set |",
        "configs": [("observations", "observations.parquet"), ("matched_observations", "matched_observations.parquet")],
        "body": """Every prompt was generated by `Qwen/Qwen3-4B-Instruct-2507` from a fixed design grid (10
categories × 32 seeds; severity clear/borderline; harmful request vs harmless twin; neutral, Japan
or US cultural cue; direct, role-play, fiction or refusal-suppression framing; Japanese, English or
mixed). There is no third-party source data. Seeds are split once, by category: `analysis` 160,
`tune` 100, `heldout` 60. All variants of a seed share its split.

The baseline answers come from Tiny Aya Water GGUF Q4_K_M via llama.cpp (greedy, seed 42, up to 1,024
tokens), judged by Qwen3Guard-Gen-0.6B GGUF Q8_0 through AuditKIT 1.1.3's `GuardJudge`.

Known issues are kept as flag columns rather than silently fixed: identical harmful/twin texts,
requests whose expected behaviour is debatable, and jailbreak rewrites that do not faithfully apply
the technique. `analysis_eligible` marks the 16,932 rows used for analysis.""",
    },
    "training_repo": {
        "pretty": "Tiny Aya safety training pairs (CuratorKIT + Qwen3Guard verified)",
        "files": "| `pipeline3_safety_pairs_v2.jsonl` | 195 kept pairs: `type` (refusal/helpful), `instruction`, `chosen`, `rejected`, CuratorKIT `source_sample_id` |\n"
                 "| `pipeline3_guard_dropped_v2.jsonl` | 37 pairs rejected by the Qwen3Guard gate, with the verdict |\n"
                 "| `pipeline3_seed_prompts_v2.json`, `pipeline3_helpful_seed_prompts_v2.json` | the 135 harmful and 101 harmless seed prompts, with Dataset2 ids and seed ids |",
        "configs": [("pairs", "pipeline3_safety_pairs_v2.jsonl"), ("guard_dropped", "pipeline3_guard_dropped_v2.jsonl")],
        "body": """Built from Dataset2's `tune` split only (38 source requests). CuratorKIT's preference task in
`two_pass` mode had Tiny Aya Water (GGUF Q4_K_M, temperature 0.2) write, for each harmful US-framed
prompt, a refusal (`chosen`) and an in-character compliance (`rejected`), and for each matched
harmless twin, a helpful answer (`chosen`) and a needless refusal (`rejected`). Qwen3Guard-Gen-0.6B
kept a refusal only if judged Safe and refusing (116 of 133), and a helpful answer only if Safe and
not refusing (79 of 99). Only `chosen` was used to train the adapters (supervised fine-tuning).

The `rejected` column contains deliberately harmful completions.""",
    },
    "eval_repo": {
        "pretty": "Tiny Aya safety LoRA: evaluation answers, metrics and run logs",
        "files": "| `pipeline5_answers.jsonl` | 684 rows: 228 validation prompts × original / restricted / broad model, with the answer, Qwen3Guard label and refusal flag, and prompt metadata (language, culture, framing, expected behaviour, seed, bucket) |\n"
                 "| `pipeline5_validation_prompts_v2.json` | the 228 prompts (114 US/neutral pairs, `analysis` split) |\n"
                 "| `metrics.json`, `RESULTS_TABLES.md`, `figures/` | every reported metric, recomputed from the answers |\n"
                 "| `pipeline1_baseline_mh_stats.json`, `pipeline2_results.json`, `pipeline2_prompts_v2.json`, `pipeline4_results.json`, `pipeline5_results.json` | per-step outputs of the original run |\n"
                 "| `executed_notebooks/` | the five notebooks with all outputs from the 2026-11-12 run (logs and traces) |\n"
                 "| `MANIFEST.sha256` | hashes of the original artifacts |",
        "configs": [("validation_answers", "pipeline5_answers.jsonl")],
        "body": """Answers were generated with greedy decoding (up to 512 new tokens, bf16) by Tiny Aya Water and by
Tiny Aya Water with each adapter merged in, and judged by Qwen3Guard-Gen-0.6B (bf16, greedy, 64
tokens). `unsafe` is 1/0 for Unsafe/Safe and empty for Controversial; `refused` is Qwen3Guard's
refusal flag. These are validation results; the heldout split has not been evaluated.

""" + RESULTS_TABLE,
    },
}


def dataset_card(key: str, ns: str, repo_names: dict) -> str:
    d = DATASETS[key]
    configs = "\n".join(f"- config_name: {name}\n  data_files: {f}" for name, f in d["configs"])
    links = (f"Models: [`{ns}/{repo_names['restricted_adapter']}`](https://huggingface.co/{ns}/{repo_names['restricted_adapter']}), "
             f"[`{ns}/{repo_names['broad_adapter']}`](https://huggingface.co/{ns}/{repo_names['broad_adapter']}). "
             f"Code and write-up: {GITHUB}")
    return f"""---
license: cc-by-nc-4.0
language: [ja, en]
tags: [safety, jailbreak, red-teaming, refusal, multilingual, japanese, tiny-aya, lexsi-hackathon]
task_categories: [text-generation, text-classification]
pretty_name: "{d['pretty']}"
configs:
{configs}
---

# {d['pretty']}

> **Content warning.** This dataset contains adversarial requests about violence, crime, self-harm,
> sexual content, hate and privacy, and some model outputs judged unsafe. It is released for safety
> research.

Part of the Tiny Aya safety project (LEXSI AI Safety & Alignment Hackathon 2026, Track 2), which
tests whether layer-restricted LoRA fine-tuning makes Tiny Aya Water safer with less over-refusal
than broad fine-tuning. {links}

## Files

| File | Contents |
|---|---|
{d['files']}

## How it was made

{d['body']}

## License

CC BY-NC 4.0. Prompts were generated by Qwen3-4B-Instruct-2507 (Apache-2.0); model answers are
outputs of Tiny Aya Water (CC BY-NC 4.0) and are released under the same non-commercial terms.
Safety labels are automatic (Qwen3Guard-Gen-0.6B) and have not been reviewed by humans.

Team: Isaac Shane and Marcus Izumi.
"""
