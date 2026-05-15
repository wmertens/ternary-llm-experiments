"""Cache teacher next-token distributions on REAL-CORPUS tokens.

Streams documents from HuggingFaceTB/smollm-corpus (FineWeb-Edu +
Cosmopedia-v2 + Python-Edu, mixed at SmolLM2's training proportions),
packs into seq_len chunks, runs the teacher *forward* (NOT generate)
on each chunk, and extracts top-K + rest_mass at every position.

This is the standard distillation data pipeline — same shard format
as cache_teacher.py (so ShardedDataset reads it without changes), but
the student now sees the teacher's distribution on text from the
teacher's own training distribution rather than on the teacher's
self-rollouts (which inherit narrow generation diversity).

Shard format (matches cache_teacher.py):
  tokens     : int32  [S, T]
  topk_idx   : int32  [S, T, K]
  topk_prob  : fp16   [S, T, K]
  rest_mass  : fp16   [S, T]
  metadata: top_k, seed_len="0" (last position is zeroed since the
                                 next-token target is out of scope)

Usage:
  smollmer-cache-teacher-corpus --out smollmer/cache_corpus \
      --tokens 25000000 --top-k 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset, interleave_datasets
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _project_text(example: dict) -> dict:
    """Normalize various column names to a single `text` field."""
    if "text" in example:
        return {"text": example["text"]}
    # Defensive: try common alternatives
    for k in ("content", "raw_content", "code"):
        if k in example:
            return {"text": example[k]}
    return {"text": ""}


def make_corpus(seed: int = 0):
    """Interleaved SmolLM-Corpus matching SmolLM2 training mix.

    Mix probabilities (approximate, from the SmolLM2 paper):
      FineWeb-Edu dedup : 0.75   (general high-quality web text)
      Cosmopedia v2     : 0.15   (synthetic textbooks)
      Python-Edu        : 0.10   (Python code)
    """
    cfgs = [
        ("fineweb-edu-dedup", 0.75),
        ("cosmopedia-v2",     0.15),
        ("python-edu",        0.10),
    ]
    datasets_, probs = [], []
    for cfg, p in cfgs:
        d = load_dataset(
            "HuggingFaceTB/smollm-corpus", cfg,
            split="train", streaming=True,
        )
        # Buffered shuffle on the stream so we don't read shards strictly
        # in disk order (which would give long stretches of one
        # source's flavor inside each interleave slot).
        d = d.shuffle(seed=seed + int(p * 1000), buffer_size=10_000)
        d = d.map(_project_text, remove_columns=None)
        datasets_.append(d)
        probs.append(p)
    return interleave_datasets(
        datasets_, probabilities=probs, seed=seed,
        stopping_strategy="all_exhausted",
    )


def _pack_iter(corpus, tokenizer, seq_len: int):
    """Yield fixed-length token sequences, packing documents back-to-back
    separated by EOS. Each yielded sequence is BOS + (seq_len - 1) tokens
    so the student's autoregressive context starts cleanly."""
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    body_len = seq_len - (1 if bos is not None else 0)
    buf: list[int] = []
    for example in corpus:
        text = example.get("text") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.append(eos)
        buf.extend(ids)
        while len(buf) >= body_len:
            chunk = buf[:body_len]
            buf = buf[body_len:]
            if bos is not None:
                chunk = [bos] + chunk
            yield chunk


@torch.no_grad()
def cache(model, tokenizer, args) -> None:
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    if args.shard_seqs % args.batch_size != 0:
        raise SystemExit(
            f"--shard-seqs ({args.shard_seqs}) must be divisible by "
            f"--batch-size ({args.batch_size})")
    batches_per_shard = args.shard_seqs // args.batch_size

    corpus = make_corpus(seed=args.seed)
    seq_iter = _pack_iter(corpus, tokenizer, args.seq_len)

    batch_buf: list[list[int]] = []
    shard_tokens, shard_kidx, shard_kprob, shard_rest = [], [], [], []
    shard_idx = 0
    tokens_cached = 0

    pbar = tqdm(total=args.tokens, unit="tok", desc="cache", unit_scale=True)

    def write_shard():
        nonlocal shard_idx, shard_tokens, shard_kidx, shard_kprob, shard_rest
        if not shard_tokens:
            return
        t = torch.cat(shard_tokens, dim=0).to(torch.int32)
        ki = torch.cat(shard_kidx, dim=0).to(torch.int32)
        kp = torch.cat(shard_kprob, dim=0).to(torch.float16)
        rm = torch.cat(shard_rest, dim=0).to(torch.float16)
        save_file({
            "tokens": t, "topk_idx": ki,
            "topk_prob": kp, "rest_mass": rm,
        }, str(out_dir / f"shard_{shard_idx:05d}.safetensors"),
           metadata={"top_k": str(args.top_k), "seed_len": "0",
                     "source": "smollm-corpus interleaved 75/15/10"})
        shard_idx += 1
        shard_tokens.clear()
        shard_kidx.clear()
        shard_kprob.clear()
        shard_rest.clear()

    try:
        while tokens_cached < args.tokens:
            try:
                batch_buf.append(next(seq_iter))
            except StopIteration:
                break
            if len(batch_buf) < args.batch_size:
                continue

            ids = torch.tensor(batch_buf, dtype=torch.long, device=device)
            batch_buf = []

            logits = model(ids).logits  # [B, T, V]
            # Softmax to probs at the model's compute dtype, then cast to
            # fp32 for the top-K extraction to avoid bf16 precision loss
            # when summing K probabilities for rest_mass.
            probs = torch.softmax(logits.float(), dim=-1)
            topk_p, topk_i = probs.topk(args.top_k, dim=-1)
            rest = (1.0 - topk_p.sum(dim=-1)).clamp_min(0.0)
            # Last position has no valid next-token target (we'd need
            # tokens[T] which is outside the chunk). Zero the target so
            # KL contribution is 0 there.
            topk_p[:, -1, :] = 0.0
            rest[:, -1] = 0.0

            shard_tokens.append(ids.cpu())
            shard_kidx.append(topk_i.cpu())
            shard_kprob.append(topk_p.cpu())
            shard_rest.append(rest.cpu())
            tokens_cached += ids.numel()
            pbar.update(ids.numel())

            if len(shard_tokens) == batches_per_shard:
                write_shard()
    finally:
        write_shard()  # flush trailing partial
        pbar.close()

    print(f"\n[done] wrote {shard_idx} shards, "
          f"{tokens_cached:,} tokens to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=25_000_000)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Sequences per teacher forward pass.")
    ap.add_argument("--shard-seqs", type=int, default=256,
                    help="Sequences per saved shard. Must be a multiple "
                         "of --batch-size.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load] {args.model} on {args.device}/{args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype).to(args.device)
    model.eval()

    cache(model, tok, args)


if __name__ == "__main__":
    main()
