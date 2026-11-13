"""LoRA training loop of pipeline 4 (notebook 04), as a reusable function.

The arithmetic is unchanged from the original run: SafeTune `lora_wrap` attaches the adapter to the
modules set with `safetune.configure(lora_target_modules=...)`, SafeTune `_tokenize_qa_rows`
tokenizes (loss on the answer only), then a plain PyTorch loop: AdamW, batch size 1, one epoch,
all-padding columns dropped (they are labelled -100, so the loss is unchanged). The only addition
is `seed`, which fixes the LoRA initialisation and the data order; the original run did not set one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import CONFIG

LORA = CONFIG["lora"]


def load_pairs(path) -> list[tuple[str, str]]:
    pairs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(p["instruction"], p["chosen"]) for p in pairs if p.get("chosen")]


def load_base_for_training(base_id: str, revision: str | None = None):
    """bf16 base model with gradient checkpointing, exactly as notebook 04 loads it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_id, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_id, revision=revision, dtype=torch.bfloat16,
                                                 device_map="auto")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return tok, model


def train_lora(model, tokenizer, rows, target_modules, seed: int, epochs=LORA["epochs"],
               batch_size=LORA["batch_size"], lr=LORA["learning_rate"], max_len=LORA["max_len"],
               log_every=25):
    """Wrap `model` with a LoRA adapter on `target_modules`, train it, return (peft_model, log)."""
    import safetune
    import torch
    from datasets import Dataset
    from safetune.runner.utils.data_utils import _tokenize_qa_rows
    from safetune.runner.utils.model_utils import lora_wrap
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    torch.manual_seed(seed)                          # LoRA A initialisation
    safetune.configure(lora_target_modules=tuple(target_modules))
    wrapped = lora_wrap(model)
    dataset = Dataset.from_list(_tokenize_qa_rows(tokenizer, rows, max_len=max_len))
    cols = [c for c in dataset.column_names if c in ("input_ids", "attention_mask", "labels")]
    loader = DataLoader(dataset.with_format("torch", columns=cols), batch_size=batch_size, shuffle=True,
                        collate_fn=default_data_collator,
                        generator=torch.Generator().manual_seed(seed))   # data order

    wrapped.train()
    trainable = [p for p in wrapped.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    losses, t0 = [], time.time()
    for _ in range(epochs):
        for batch in loader:
            keep = batch["attention_mask"].bool().any(dim=0)
            batch = {k: v[:, keep].to(wrapped.device) for k, v in batch.items()}
            out = wrapped(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(out.loss.item()))
            if log_every and len(losses) % log_every == 0:
                print(f"    step {len(losses)}: loss={losses[-1]:.4f}", flush=True)
    lora_b = [p for n, p in wrapped.named_parameters() if "lora_B" in n]
    max_b = max(float(p.detach().abs().max()) for p in lora_b)
    assert max_b > 0, "LoRA B weights are still all zero: training did not update the adapter"
    q = max(1, len(losses) // 4)
    log = {"seed": seed, "target_modules": list(target_modules),
           "trainable_params": sum(p.numel() for p in trainable), "steps": len(losses),
           "loss_first_quarter": sum(losses[:q]) / q, "loss_last_quarter": sum(losses[-q:]) / q,
           "losses": losses, "max_abs_lora_B": max_b, "train_seconds": round(time.time() - t0, 1),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None}
    return wrapped, log


def prepare_for_generation(peft_model):
    """Merge the adapter into the base weights (as pipeline 4 did) and switch to inference mode."""
    merged = peft_model.merge_and_unload()
    merged.gradient_checkpointing_disable()
    merged.config.use_cache = True
    merged.eval()
    return merged
