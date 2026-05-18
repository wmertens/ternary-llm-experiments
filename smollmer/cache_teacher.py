"""Cache teacher top-K logits for distillation.

Two source modes (--source):
  rollout  : self-rollouts via model.generate(output_logits=True); no extra
             forward pass needed — logits are the sampling-time distributions.
  corpus   : forward pass on real FineWeb-Edu/Cosmopedia-v2/Python-Edu text
             streamed from HuggingFaceTB/smollm-corpus (75/15/10 mix).

Both modes write identical shard files readable by ShardedDataset:
  tokens    : int32  [S, T]
  topk_idx  : int32  [S, T, K]
  topk_prob : fp16   [S, T, K]
  rest_mass : fp16   [S, T]   (1 − sum(topk_prob), positions with no target = 0)

Append support: if --out already contains shard_NNNNN.safetensors files, new
shards are numbered from max(existing) + 1 onward; existing shards are untouched.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_start_shard(out_dir: Path) -> int:
    """Return the next shard index (0 if directory is empty/new)."""
    existing = sorted(out_dir.glob("shard_?????.safetensors"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[1]) + 1


def _save_shard(
    out_dir: Path,
    shard_idx: int,
    tokens: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_prob: torch.Tensor,
    rest_mass: torch.Tensor,
    metadata: dict[str, str],
) -> None:
    save_file(
        {
            "tokens": tokens.to(torch.int32).contiguous(),
            "topk_idx": topk_idx.to(torch.int32).contiguous(),
            "topk_prob": topk_prob.to(torch.float16).contiguous(),
            "rest_mass": rest_mass.to(torch.float16).contiguous(),
        },
        str(out_dir / f"shard_{shard_idx:05d}.safetensors"),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Rollout mode
# ---------------------------------------------------------------------------

SEED_PROMPTS: tuple[str, ...] = (
    "The", "A", "In", "On", "When", "Because", "However", "Today",
    "She", "He", "They", "We", "It", "There", "Here", "Once",
    "According to", "In the", "On the", "At the", "After the",
    "Before the", "During the", "While the", "Although",
    "Scientists", "Researchers", "The system", "The function",
    "The user", "The model", "Let me", "I think", "Imagine",
    "Consider", "Suppose", "If you", "When you", "Yesterday",
    "Tomorrow", "Last year", "Next week", "The author",
)


def _seed_token_ids(tokenizer) -> list[int]:
    ids = []
    for s in SEED_PROMPTS:
        t = tokenizer.encode(s, add_special_tokens=False)
        if t:
            ids.append(t[0])
    return list(dict.fromkeys(ids)) or [0]


@torch.no_grad()
def _generate_batch_with_topk(
    model, tokenizer, batch_size: int, seq_len: int, device: str,
    rng: random.Random, temperature: float, min_p: float,
    seed_token_ids: list[int], top_k: int, chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Generate one batch; extract top-K from sampling-time logits.

    Returns (tokens [B,T] int32, topk_idx [B,T,K], topk_prob [B,T,K] fp16,
             rest_mass [B,T] fp16, seed_len).
    """
    bos = tokenizer.bos_token_id
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    first = [rng.choice(seed_token_ids) for _ in range(batch_size)]
    if bos is not None:
        ids = torch.tensor([[bos, t] for t in first], dtype=torch.long)
    else:
        ids = torch.tensor([[t] for t in first], dtype=torch.long)
    seed_len = ids.shape[1]
    ids = ids.to(device)

    result = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=seq_len - seed_len,
        do_sample=True,
        temperature=temperature,
        min_p=min_p,
        pad_token_id=pad,
        return_dict_in_generate=True,
        output_logits=True,
    )

    seqs = result.sequences
    if seqs.shape[1] < seq_len:
        pad_block = torch.full(
            (batch_size, seq_len - seqs.shape[1]), pad,
            dtype=seqs.dtype, device=seqs.device)
        seqs = torch.cat([seqs, pad_block], dim=1)
    else:
        seqs = seqs[:, :seq_len]

    out_idx = torch.zeros((batch_size, seq_len, top_k), dtype=torch.int32)
    out_prob = torch.zeros((batch_size, seq_len, top_k), dtype=torch.float16)
    out_rest = torch.zeros((batch_size, seq_len), dtype=torch.float16)

    # result.logits[i] is the distribution that produced token at position
    # seed_len + i, so it belongs at write position seed_len + i - 1.
    logits_list: list[torch.Tensor | None] = list(result.logits)
    n_gen = len(logits_list)
    write_start = seed_len - 1
    for s in range(0, n_gen, chunk):
        e = min(s + chunk, n_gen)
        ch = torch.stack(logits_list[s:e], dim=1).float()  # [B, ch, V]
        probs = torch.softmax(ch, dim=-1)
        vals, idx = torch.topk(probs, k=top_k, dim=-1)
        rest = (1.0 - vals.sum(dim=-1)).clamp_min(0.0)
        out_idx[:, write_start + s: write_start + e] = idx.to(torch.int32).cpu()
        out_prob[:, write_start + s: write_start + e] = vals.to(torch.float16).cpu()
        out_rest[:, write_start + s: write_start + e] = rest.to(torch.float16).cpu()
        for i in range(s, e):
            logits_list[i] = None
        del ch, probs, vals, idx, rest

    return seqs.cpu().to(torch.int32), out_idx, out_prob, out_rest, seed_len


def run_rollout(model, tok, args, start_shard: int) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    seed_ids = _seed_token_ids(tok)

    n_seqs = math.ceil(args.tokens / args.seq_len)
    n_shards = math.ceil(n_seqs / args.shard_seqs)
    print(f"[rollout] {n_seqs} sequences → {n_shards} shards "
          f"(starting at shard {start_shard:05d})")

    pbar = tqdm(total=n_seqs, desc="cache", unit="seq")
    seqs_done = 0
    seed_len_meta = 0
    for i in range(n_shards):
        shard_idx = start_shard + i
        in_shard = min(args.shard_seqs, n_seqs - seqs_done)
        token_chunks, idx_chunks, prob_chunks, rest_chunks = [], [], [], []
        gathered = 0
        while gathered < in_shard:
            bs = min(args.batch_size, in_shard - gathered)
            t, ki, kp, rm, seed_len_meta = _generate_batch_with_topk(
                model, tok, bs, args.seq_len, args.device, rng,
                args.temperature, args.min_p, seed_ids,
                args.top_k, args.logit_chunk,
            )
            token_chunks.append(t)
            idx_chunks.append(ki)
            prob_chunks.append(kp)
            rest_chunks.append(rm)
            gathered += bs
            pbar.update(bs)
        _save_shard(
            args.out, shard_idx,
            torch.cat(token_chunks, dim=0),
            torch.cat(idx_chunks, dim=0),
            torch.cat(prob_chunks, dim=0),
            torch.cat(rest_chunks, dim=0),
            metadata={"seed_len": str(seed_len_meta), "top_k": str(args.top_k),
                      "source": "rollout"},
        )
        seqs_done += in_shard
    pbar.close()
    print(f"[done] wrote shards {start_shard:05d}–{start_shard + n_shards - 1:05d} "
          f"to {args.out}")


# ---------------------------------------------------------------------------
# Corpus mode
# ---------------------------------------------------------------------------

def _project_text(example: dict) -> dict:
    if "text" in example:
        return {"text": example["text"]}
    for k in ("content", "raw_content", "code"):
        if k in example:
            return {"text": example[k]}
    return {"text": ""}


def _make_corpus(seed: int):
    """Interleaved SmolLM-Corpus at SmolLM2 training proportions (75/15/10)."""
    from datasets import load_dataset, interleave_datasets

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
        d = d.shuffle(seed=seed + int(p * 1000), buffer_size=10_000)
        d = d.map(_project_text, remove_columns=None)
        datasets_.append(d)
        probs.append(p)
    return interleave_datasets(datasets_, probabilities=probs, seed=seed,
                               stopping_strategy="all_exhausted")


def _pack_iter(corpus, tokenizer, seq_len: int, docs_counter: list[int]):
    """Yield fixed-length token sequences (BOS + seq_len-1 body tokens).

    Increments docs_counter[0] for each corpus document consumed, so callers
    can persist the exact position for later append/resume via corpus.skip().
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    body_len = seq_len - (1 if bos is not None else 0)
    buf: list[int] = []
    for example in corpus:
        text = example.get("text") or ""
        if not text:
            continue
        docs_counter[0] += 1
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.append(eos)
        buf.extend(ids)
        while len(buf) >= body_len:
            chunk = buf[:body_len]
            buf = buf[body_len:]
            if bos is not None:
                chunk = [bos] + chunk
            yield chunk


_AVG_DOC_TOKENS = 1200  # empirical estimate for smollm-corpus 75/15/10 mix


def _docs_to_skip(out_dir: Path, start_shard: int, seq_len: int) -> int:
    """Return how many corpus documents to skip when appending.

    Reads docs_consumed from the last shard's metadata if available;
    otherwise estimates from total tokens cached / avg doc length.
    """
    from safetensors import safe_open
    shards = sorted(out_dir.glob("shard_?????.safetensors"))

    with safe_open(str(shards[-1]), framework="pt") as f:
        meta = f.metadata() or {}
    if "docs_consumed" in meta:
        n = int(meta["docs_consumed"])
        print(f"[append] resuming from docs_consumed={n:,} (from shard metadata)")
        return n

    # Estimate: total_tokens_cached / avg_doc_len
    with safe_open(str(shards[0]), framework="pt") as f:
        seqs_per_shard = f.get_slice("tokens").get_shape()[0]
    total_tokens = start_shard * seqs_per_shard * seq_len
    estimated = total_tokens // _AVG_DOC_TOKENS
    print(f"[append] no docs_consumed metadata; estimating {estimated:,} docs to skip "
          f"({total_tokens:,} tokens ÷ {_AVG_DOC_TOKENS} avg tokens/doc)")
    return estimated


@torch.no_grad()
def run_corpus(model, tok, args, start_shard: int) -> None:
    if args.shard_seqs % args.batch_size != 0:
        raise SystemExit(
            f"--shard-seqs ({args.shard_seqs}) must be divisible by "
            f"--batch-size ({args.batch_size})")
    batches_per_shard = args.shard_seqs // args.batch_size

    corpus = _make_corpus(seed=args.seed)
    docs_counter = [0]

    if start_shard > 0:
        n_skip = _docs_to_skip(args.out, start_shard, args.seq_len)
        print(f"[append] calling corpus.skip({n_skip:,}) …", flush=True)
        corpus = corpus.skip(n_skip)
        docs_counter[0] = n_skip

    seq_iter = _pack_iter(corpus, tok, args.seq_len, docs_counter)

    batch_buf: list[list[int]] = []
    shard_tokens, shard_kidx, shard_kprob, shard_rest = [], [], [], []
    shard_idx = start_shard
    tokens_cached = 0

    n_shards_est = math.ceil(args.tokens / (args.shard_seqs * args.seq_len))
    print(f"[corpus] ~{args.tokens:,} tokens → ~{n_shards_est} shards "
          f"(starting at shard {start_shard:05d})")
    pbar = tqdm(total=args.tokens, unit="tok", desc="cache", unit_scale=True)

    def write_shard():
        nonlocal shard_idx, shard_tokens, shard_kidx, shard_kprob, shard_rest
        if not shard_tokens:
            return
        _save_shard(
            args.out, shard_idx,
            torch.cat(shard_tokens, dim=0),
            torch.cat(shard_kidx, dim=0),
            torch.cat(shard_kprob, dim=0),
            torch.cat(shard_rest, dim=0),
            metadata={"top_k": str(args.top_k), "seed_len": "0",
                      "source": "smollm-corpus interleaved 75/15/10",
                      "docs_consumed": str(docs_counter[0])},
        )
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

            ids = torch.tensor(batch_buf, dtype=torch.long, device=args.device)
            batch_buf = []

            probs = torch.softmax(model(ids).logits.float(), dim=-1)
            topk_p, topk_i = probs.topk(args.top_k, dim=-1)
            rest = (1.0 - topk_p.sum(dim=-1)).clamp_min(0.0)
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
        write_shard()
        pbar.close()

    print(f"[done] wrote shards {start_shard:05d}–{shard_idx - 1:05d}, "
          f"{tokens_cached:,} tokens to {args.out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cache teacher top-K logits (rollout or corpus mode).")
    ap.add_argument("--source", default="corpus",
                    choices=["rollout", "corpus"],
                    help="rollout: self-generated text via model.generate; "
                         "corpus: real text via model.forward on smollm-corpus.")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", "--total-tokens", type=int, default=None,
                    dest="tokens",
                    help="Target total tokens. Default: 10M (rollout), 25M (corpus).")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="Sequence length. Default: 1024 (rollout), 512 (corpus).")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Top-K vocab entries per position. Default: 64 (rollout), 32 (corpus).")
    ap.add_argument("--shard-seqs", type=int, default=None,
                    help="Sequences per shard file. Default: 128 (rollout), 256 (corpus).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Sequences per GPU call. Default: 16 (rollout), 8 (corpus).")
    # rollout-only args
    ap.add_argument("--logit-chunk", type=int, default=64,
                    help="[rollout] Positions per softmax/topk chunk on GPU.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="[rollout] Sampling temperature.")
    ap.add_argument("--min-p", type=float, default=0.05,
                    help="[rollout] min-P sampling threshold.")
    # shared
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Apply per-source defaults for unset args
    is_rollout = args.source == "rollout"
    if args.tokens is None:
        args.tokens = 10_000_000 if is_rollout else 25_000_000
    if args.seq_len is None:
        args.seq_len = 1024 if is_rollout else 512
    if args.top_k is None:
        args.top_k = 64 if is_rollout else 32
    if args.shard_seqs is None:
        args.shard_seqs = 128 if is_rollout else 256
    if args.batch_size is None:
        args.batch_size = 16 if is_rollout else 8

    args.out.mkdir(parents=True, exist_ok=True)
    start_shard = _find_start_shard(args.out)
    if start_shard > 0:
        print(f"[append] found {start_shard} existing shards; "
              f"new shards start at {start_shard:05d}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[load] {args.model} on {args.device}/{args.dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype).to(args.device).eval()

    if is_rollout:
        run_rollout(model, tok, args, start_shard)
    else:
        run_corpus(model, tok, args, start_shard)


if __name__ == "__main__":
    main()
