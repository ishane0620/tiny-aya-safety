"""Heldout (final test) evaluation with checkpoints. See docs/HELDOUT_PLAN.md.

    python scripts/run_heldout_eval.py                          # all models, priority order (GPU)
    python scripts/run_heldout_eval.py --models baseline restricted --limit 16   # quick check

For each model, prompts are answered in chunks; after every chunk the answers are judged by
Qwen3Guard and appended to <out>/partial_<model>.jsonl (one line per prompt, flushed and fsynced).
A rerun skips prompts already in the partial file, so an interrupted run loses at most one chunk.
When a model is complete, <out>/answers_<model>.jsonl is written atomically in prompt-file order
(same schema as pipeline5_answers.jsonl). <out>/progress.json is rewritten after every chunk.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tiny_aya_safety import generation  # noqa: E402
from tiny_aya_safety.config import CONFIG, data_path, hf_repo, output_dir  # noqa: E402

ADAPTERS = {
    "baseline": None,
    "restricted": "hf:restricted_adapter",
    "broad": "hf:broad_adapter",
    "L9_s0": "results/layer_control/adapters/L9_s0",
    "L9_s1": "results/layer_control/adapters/L9_s1",
    "L18_s0": "results/layer_control/adapters/L18_s0",
    "L18_s1": "results/layer_control/adapters/L18_s1",
}


def adapter_path(name):
    src = ADAPTERS[name]
    if src is None:
        return None
    if src.startswith("hf:"):
        return hf_repo(src[3:])
    p = REPO / src
    assert (p / "adapter_config.json").exists(), f"missing adapter {p}"
    return str(p)


def load(name):
    """Base model in bf16, plus the adapter merged in (as pipeline 4 built the evaluated models)."""
    tok, model = generation.load_model("baseline")
    src = adapter_path(name)
    if src:
        from peft import PeftModel
        print(f"[{name}] applying adapter {src}", flush=True)
        model = PeftModel.from_pretrained(model, src).merge_and_unload()
        model.eval()
    return tok, model


def read_partial(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except json.JSONDecodeError:          # a torn last line from a killed run
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(ADAPTERS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--chunk", type=int, default=64, help="prompts per checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (testing)")
    args = ap.parse_args()
    import torch

    out = args.out or (output_dir() / "heldout")
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(data_path("heldout_test_prompts.json").read_text(encoding="utf-8"))[: args.limit]
    by_id = {r["id"]: r for r in prompts}
    # One fixed order for every model: by prompt length, so batches pad little.
    order = sorted(prompts, key=lambda r: (len(r["prompt"]), r["id"]))
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    progress["n_prompts"] = len(prompts)
    print(f"{len(prompts)} prompts, models {args.models} -> {out}", flush=True)

    gtok, guard = generation.load_guard()
    for name in args.models:
        final = out / f"answers_{name}.jsonl"
        if final.exists():
            print(f"[{name}] complete, skipping", flush=True)
            continue
        partial = out / f"partial_{name}.jsonl"
        done = read_partial(partial)
        todo = [r for r in order if r["id"] not in done]
        print(f"[{name}] {len(done)} done from checkpoint, {len(todo)} to go", flush=True)
        t0 = time.time()
        if todo:
            tok, model = load(name)
            with partial.open("a", encoding="utf-8") as f:
                for i in range(0, len(todo), args.chunk):
                    chunk = todo[i: i + args.chunk]
                    texts = [r["prompt"] for r in chunk]
                    answers = generation.generate_answers(model, tok, texts, verbose=False)
                    verdicts = generation.judge_answers(guard, gtok, texts, answers)
                    for r, a, v in zip(chunk, answers, verdicts):
                        f.write(json.dumps({**r, "model": name, "answer": a, **v}, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    n_done = len(done) + i + len(chunk)
                    rate = (time.time() - t0) / (i + len(chunk))
                    progress[name] = {"done": n_done, "total": len(prompts), "sec_per_prompt": round(rate, 2),
                                      "eta_min": round(rate * (len(prompts) - n_done) / 60, 1)}
                    progress["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    progress_path.write_text(json.dumps(progress, indent=2))
                    print(f"[{name}] {n_done}/{len(prompts)}  {rate:.2f}s/prompt", flush=True)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        done = read_partial(partial)
        assert set(done) == set(by_id), f"{name}: {len(set(by_id) - set(done))} prompts missing"
        tmp = final.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in prompts:                                 # prompt-file order
                f.write(json.dumps(done[r["id"]], ensure_ascii=False) + "\n")
        os.replace(tmp, final)
        labels = {k: sum(d["label"] == k for d in done.values()) for k in ("Safe", "Unsafe", "Controversial", "Unparsed")}
        progress[name] = {"done": len(prompts), "total": len(prompts), "complete": True, "labels": labels,
                          "minutes": round((time.time() - t0) / 60, 1)}
        progress_path.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] COMPLETE {labels}", flush=True)
    print("ALL MODELS DONE", flush=True)


if __name__ == "__main__":
    main()
