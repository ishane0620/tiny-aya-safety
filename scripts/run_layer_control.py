"""Layer-choice control: train the same one-matrix LoRA on several layers and evaluate each.

    python scripts/run_layer_control.py                       # layers 18 15 9 30, seeds 0 1 (GPU)
    python scripts/run_layer_control.py --layers 18 9 --seeds 0

Every arm uses the 195 training pairs, the pipeline 4 settings and a fresh copy of the base model,
then answers the 228 validation prompts with the pipeline 5 settings and is judged by Qwen3Guard.
Per arm it writes, under <out>/:
    answers_L{layer}_s{seed}.jsonl   same schema as pipeline5_answers.jsonl (model = "L{layer}_s{seed}")
    adapters/L{layer}_s{seed}/       the trained adapter (+ train_log.json)
Arms whose answers file exists are skipped, so an interrupted run resumes where it stopped.
progress.json records what is done, for polling from outside.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_aya_safety import generation, training  # noqa: E402
from tiny_aya_safety.config import CONFIG, data_path, output_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 15, 9, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--target-template", default="layers.{layer}.mlp.down_proj")
    # Overrides, for testing the loop with a tiny stand-in model on CPU.
    ap.add_argument("--base-model", default=CONFIG["base_model"]["id"])
    ap.add_argument("--base-revision", default=CONFIG["base_model"]["revision"])
    ap.add_argument("--guard-model", default=CONFIG["judge_model"]["id"])
    ap.add_argument("--guard-revision", default=CONFIG["judge_model"]["revision"])
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-prompts", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=CONFIG["generation"]["max_new_tokens"])
    args = ap.parse_args()
    if args.base_revision in ("", "none"):
        args.base_revision = None
    if args.guard_revision in ("", "none"):
        args.guard_revision = None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = args.out or (output_dir() / "layer_control")
    (out / "adapters").mkdir(parents=True, exist_ok=True)
    rows = training.load_pairs(data_path("pipeline3_safety_pairs_v2.jsonl"))[: args.limit_train]
    val = json.loads(data_path("pipeline5_validation_prompts_v2.json").read_text(encoding="utf-8"))
    val = val[: args.limit_prompts]
    prompts = [r["prompt"] for r in val]
    arms = [(layer, seed) for layer in args.layers for seed in args.seeds]
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    print(f"{len(arms)} arms, {len(rows)} training pairs, {len(prompts)} validation prompts -> {out}", flush=True)

    guard = None
    for layer, seed in arms:
        name = f"L{layer}_s{seed}"
        answers_path = out / f"answers_{name}.jsonl"
        if answers_path.exists():
            print(f"[{name}] already done, skipping", flush=True)
            continue
        t0 = time.time()
        print(f"[{name}] training", flush=True)
        tok, model = training.load_base_for_training(args.base_model, args.base_revision)
        target = [args.target_template.format(layer=layer)]
        peft_model, log = training.train_lora(model, tok, rows, target, seed)
        adapter_dir = out / "adapters" / name
        peft_model.save_pretrained(adapter_dir)
        print(f"[{name}] trained: {log['trainable_params']:,} params, loss {log['loss_first_quarter']:.3f} -> "
              f"{log['loss_last_quarter']:.3f}, {log['train_seconds']}s", flush=True)

        merged = training.prepare_for_generation(peft_model)
        gen_tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision, padding_side="left")
        if gen_tok.pad_token_id is None:
            gen_tok.pad_token = gen_tok.eos_token
        answers = generation.generate_answers(merged, gen_tok, prompts, max_new_tokens=args.max_new_tokens,
                                              verbose=False)
        del merged, peft_model, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if guard is None:
            gtok = AutoTokenizer.from_pretrained(args.guard_model, revision=args.guard_revision, padding_side="left")
            if gtok.pad_token_id is None:
                gtok.pad_token = gtok.eos_token
            gmodel = AutoModelForCausalLM.from_pretrained(args.guard_model, revision=args.guard_revision,
                                                          dtype=torch.bfloat16, device_map="auto").eval()
            guard = (gtok, gmodel)
        verdicts = generation.judge_answers(guard[1], guard[0], prompts, answers)

        log.update(layer=layer, total_seconds=round(time.time() - t0, 1),
                   labels={k: sum(v["label"] == k for v in verdicts) for k in ("Safe", "Unsafe", "Controversial", "Unparsed")})
        (adapter_dir / "train_log.json").write_text(json.dumps(log, indent=2))
        tmp = answers_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r, a, v in zip(val, answers, verdicts):
                f.write(json.dumps({**r, "model": name, "answer": a, **v}, ensure_ascii=False) + "\n")
        os.replace(tmp, answers_path)                       # appears only when complete
        progress[name] = {"done": True, "seconds": log["total_seconds"], "labels": log["labels"]}
        progress_path.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] done in {log['total_seconds']}s: {log['labels']}", flush=True)
    print("ALL ARMS DONE", flush=True)


if __name__ == "__main__":
    main()
