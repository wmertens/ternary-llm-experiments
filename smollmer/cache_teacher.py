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


# SmolLM2 training proportions, expressed as integer counts per cadence cycle.
# 60-cycle ⇒ every shard divisible by 60 sees exactly 45F/9C/6P at the target
# 75/15/10 mix, with sub-cycle shuffling so smaller batch sizes also see a
# good mix (every batch size that divides 60: 1,2,3,4,5,6,10,12,15,20,30,60).
_SOURCES: tuple[tuple[str, int], ...] = (
    ("fineweb-edu-dedup", 51),
    ("cosmopedia-v2",      9),
    # python-edu dropped 2026-05-25: HF schema now ships metadata only
    # (blob_id/repo_name/path), no text/code. Re-add via a replacement
    # code source (e.g. bigcode/the-stack-smol-py) when needed.
)
_CADENCE = sum(n for _, n in _SOURCES)  # 60
_N_SOURCES = len(_SOURCES)


def _make_streams(seed: int):
    """Return one streaming dataset per source (no interleaving)."""
    from datasets import load_dataset

    streams = []
    for i, (cfg, _) in enumerate(_SOURCES):
        d = load_dataset(
            "HuggingFaceTB/smollm-corpus", cfg,
            split="train", streaming=True,
        )
        d = d.shuffle(seed=seed + 1000 * (i + 1), buffer_size=10_000)
        d = d.map(_project_text, remove_columns=None)
        streams.append(d)
    return streams


def _pack_one_source(stream, tokenizer, seq_len: int, docs_counter: list[int]):
    """Yield fixed-length token sequences from a single source's stream.

    Increments docs_counter[0] for each consumed document so callers can
    persist the per-source skip position.
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    body_len = seq_len - (1 if bos is not None else 0)
    buf: list[int] = []
    for example in stream:
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


def _cycle_order(seed: int, cycle_idx: int) -> list[int]:
    """Return the deterministic per-cycle source-index pattern (length _CADENCE).

    Identical RNG used by both the streaming iterator and the shard-assembly
    step, so a filter-free run lays sequences out in the same order as before.
    """
    rng = random.Random(seed * 1_000_003 + cycle_idx)
    order: list[int] = []
    for i, (_, n) in enumerate(_SOURCES):
        order.extend([i] * n)
    rng.shuffle(order)
    return order


def _stratified_seq_iter(packers, seed: int, start_cycle: int = 0):
    """Yield (src_idx, chunk) pairs in cadence-_CADENCE cycles.

    src_idx lets the caller (a) know which per-source pool to push the
    forward-pass result into and (b) draw a replacement from the same source
    on rejection. Resume reproduces the exact ordering by seeding from
    (seed, cycle_idx) per cycle.
    """
    cycle = start_cycle
    while True:
        for src_idx in _cycle_order(seed, cycle):
            yield src_idx, next(packers[src_idx])
        cycle += 1


def _read_resume_state(out_dir: Path) -> tuple[list[int], int]:
    """Read (per-source docs_consumed, next cycle_idx) from last shard's metadata.

    Requires the cadence-60 metadata keys; old shards (from the pre-stratified
    cache_teacher.py) are not supported — regenerate the cache.
    """
    from safetensors import safe_open
    shards = sorted(out_dir.glob("shard_?????.safetensors"))
    with safe_open(str(shards[-1]), framework="pt") as f:
        meta = f.metadata() or {}
    required = ("docs_consumed_per_source", "next_cycle_idx", "cadence")
    if not all(k in meta for k in required):
        raise SystemExit(
            f"[append] existing shards in {out_dir} predate the cadence-60 "
            f"layout (missing keys: {[k for k in required if k not in meta]}). "
            "Delete the directory and re-run, or point --out to a fresh path.")
    per_src = [int(x) for x in meta["docs_consumed_per_source"].split(",")]
    if len(per_src) != _N_SOURCES:
        raise SystemExit(
            f"[append] docs_consumed_per_source has {len(per_src)} entries, "
            f"expected {_N_SOURCES}")
    cycle_idx = int(meta["next_cycle_idx"])
    print(f"[append] resuming: docs_consumed={per_src}  next_cycle={cycle_idx:,}")
    return per_src, cycle_idx


@torch.no_grad()
def run_corpus(model, tok, args, start_shard: int) -> None:
    if args.shard_seqs % args.batch_size != 0:
        raise SystemExit(
            f"--shard-seqs ({args.shard_seqs}) must be divisible by "
            f"--batch-size ({args.batch_size})")
    if args.shard_seqs % _CADENCE != 0:
        raise SystemExit(
            f"--shard-seqs ({args.shard_seqs}) must be divisible by the "
            f"interleave cadence ({_CADENCE}) so shards align to whole 45/9/6 "
            f"cycles. Try {(args.shard_seqs // _CADENCE) * _CADENCE} or "
            f"{(args.shard_seqs // _CADENCE + 1) * _CADENCE}.")
    cycles_per_shard = args.shard_seqs // _CADENCE
    per_src_per_shard = [cycles_per_shard * n for _, n in _SOURCES]
    threshold = args.max_mean_teacher_ce  # None = no filtering
    filtering = threshold is not None

    streams = _make_streams(seed=args.seed)
    docs_counters = [[0] for _ in range(_N_SOURCES)]
    cycle_idx = 0

    if start_shard > 0:
        per_src, cycle_idx = _read_resume_state(args.out)
        for i, n in enumerate(per_src):
            print(f"[append] stream {i} ({_SOURCES[i][0]}): skip({n:,})",
                  flush=True)
            streams[i] = streams[i].skip(n)
            docs_counters[i][0] = n

    packers = [
        _pack_one_source(streams[i], tok, args.seq_len, docs_counters[i])
        for i in range(_N_SOURCES)
    ]
    seq_iter = _stratified_seq_iter(packers, seed=args.seed,
                                    start_cycle=cycle_idx)

    # Per-source FIFO pools of accepted sequences (cpu tensors).
    # Each entry: (tokens [T], topk_idx [T,K], topk_prob [T,K], rest [T]).
    pool: list[list[tuple] ] = [[] for _ in range(_N_SOURCES)]
    pulled = [0] * _N_SOURCES   # total drawn from each source's packer
    rejected = [0] * _N_SOURCES  # total dropped by the CE filter

    shard_idx = start_shard
    tokens_per_shard = args.shard_seqs * args.seq_len
    n_shards_target = math.ceil(args.tokens / tokens_per_shard)
    print(f"[corpus] {n_shards_target} shards × {tokens_per_shard:,} tok "
          f"= {n_shards_target * tokens_per_shard:,} tok target "
          f"(cadence={_CADENCE}, {cycles_per_shard} cycles/shard, "
          f"per-shard targets {per_src_per_shard}, "
          f"starting at shard {start_shard:05d}, cycle {cycle_idx:,})")
    if filtering:
        print(f"[corpus] filter: drop sequences with mean teacher CE > "
              f"{threshold:.3f}")
    pbar = tqdm(total=n_shards_target * tokens_per_shard, unit="tok",
                desc="cache", unit_scale=True)

    pending_src: list[int] = []
    pending_chunks: list[list[int]] = []

    def have_full_shard() -> bool:
        return all(len(pool[i]) >= per_src_per_shard[i]
                   for i in range(_N_SOURCES))

    def need_more() -> bool:
        # Pull more if any source's pool is still below its per-shard quota
        # (we may overshoot one source while another lags; that's fine,
        # the surplus carries over to the next shard).
        return any(len(pool[i]) < per_src_per_shard[i]
                   for i in range(_N_SOURCES))

    def flush_batch() -> None:
        """Forward-pass the current pending batch, score, and push accepted
        sequences into their source pools."""
        if not pending_chunks:
            return
        ids = torch.tensor(pending_chunks, dtype=torch.long,
                           device=args.device)
        logits = model(ids).logits.float()
        probs = torch.softmax(logits, dim=-1)
        topk_p, topk_i = probs.topk(args.top_k, dim=-1)
        rest = (1.0 - topk_p.sum(dim=-1)).clamp_min(0.0)
        topk_p[:, -1, :] = 0.0
        rest[:, -1] = 0.0

        if filtering:
            # Teacher CE on the actual next token. At position t the teacher
            # predicts ids[:, t+1]; the final position has no target. Average
            # over the valid positions only.
            log_probs = torch.log_softmax(logits, dim=-1)
            target = ids[:, 1:].unsqueeze(-1)                          # [B,T-1,1]
            nll = -log_probs[:, :-1, :].gather(-1, target).squeeze(-1)  # [B,T-1]
            mean_ce = nll.mean(dim=-1).cpu()                            # [B]
        else:
            mean_ce = None

        ids_cpu = ids.cpu()
        topk_i_cpu = topk_i.cpu()
        topk_p_cpu = topk_p.cpu()
        rest_cpu = rest.cpu()

        for j, src in enumerate(pending_src):
            if mean_ce is not None and mean_ce[j].item() > threshold:
                rejected[src] += 1
                continue
            pool[src].append((
                ids_cpu[j], topk_i_cpu[j], topk_p_cpu[j], rest_cpu[j],
            ))
        pending_src.clear()
        pending_chunks.clear()

    def assemble_and_write_shard() -> None:
        nonlocal shard_idx
        # Pop seqs from per-source pools in cycle order for this shard's
        # cycles_per_shard cycles, starting from cycle_idx + (shard_idx-start)*cps.
        first_cycle = cycle_idx + cycles_per_shard * (shard_idx - start_shard)
        cursors = [0, 0, 0]
        out_tokens, out_kidx, out_kprob, out_rest = [], [], [], []
        for c in range(cycles_per_shard):
            for src in _cycle_order(args.seed, first_cycle + c):
                t, ki, kp, rm = pool[src][cursors[src]]
                cursors[src] += 1
                out_tokens.append(t)
                out_kidx.append(ki)
                out_kprob.append(kp)
                out_rest.append(rm)
        # Drop consumed entries from each pool (keep tail = surplus).
        for i in range(_N_SOURCES):
            del pool[i][:cursors[i]]

        next_cycle = first_cycle + cycles_per_shard
        meta = {
            "top_k": str(args.top_k), "seed_len": "0",
            "source": "smollm-corpus stratified 75/15/10 (cadence 60)",
            "cadence": str(_CADENCE),
            "per_cycle": ",".join(str(n) for _, n in _SOURCES),
            "sources": ",".join(name for name, _ in _SOURCES),
            "docs_consumed_per_source": ",".join(
                str(c[0]) for c in docs_counters),
            "next_cycle_idx": str(next_cycle),
            "pulled_per_source": ",".join(str(x) for x in pulled),
            "rejected_per_source": ",".join(str(x) for x in rejected),
        }
        if filtering:
            meta["max_mean_teacher_ce"] = f"{threshold:.6f}"
        _save_shard(
            args.out, shard_idx,
            torch.stack(out_tokens, dim=0),
            torch.stack(out_kidx, dim=0),
            torch.stack(out_kprob, dim=0),
            torch.stack(out_rest, dim=0),
            metadata=meta,
        )
        shard_idx += 1

    while (shard_idx - start_shard) < n_shards_target:
        while need_more():
            src, chunk = next(seq_iter)
            pending_src.append(src)
            pending_chunks.append(chunk)
            pulled[src] += 1
            if len(pending_chunks) == args.batch_size:
                flush_batch()
                pbar.update(args.batch_size * args.seq_len)
        # Out of pending pulls. If batch_buf still partial (impossible here
        # since need_more() controls), flush. Then assemble shard.
        if pending_chunks:
            flush_batch()
        assemble_and_write_shard()

    pbar.close()
    if filtering:
        tot_pulled = sum(pulled)
        tot_rej = sum(rejected)
        rej_pct = 100.0 * tot_rej / max(1, tot_pulled)
        per_src_pct = [
            100.0 * rejected[i] / max(1, pulled[i]) for i in range(_N_SOURCES)
        ]
        print(f"[filter] rejected {tot_rej:,}/{tot_pulled:,} = {rej_pct:.2f}% "
              f"(per source: " + ", ".join(
                  f"{_SOURCES[i][0]}={per_src_pct[i]:.2f}%"
                  for i in range(_N_SOURCES)) + ")")
    print(f"[done] wrote shards {start_shard:05d}–{shard_idx - 1:05d} "
          f"({n_shards_target * tokens_per_shard:,} tokens) to {args.out}")


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
    # corpus-only filtering
    ap.add_argument("--max-mean-teacher-ce", type=float, default=None,
                    help="[corpus] Drop any sequence whose mean teacher CE "
                         "on the actual next token exceeds this threshold "
                         "(nats). Cheap noise filter for garbled / OCR-broken "
                         "docs. Pick the threshold by running once unfiltered "
                         "and inspecting the per-seq CE distribution (e.g. "
                         "95th percentile). Off by default.")
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
        # corpus default = 240 = 4 × cadence(60), also divisible by batch_size 8
        args.shard_seqs = 128 if is_rollout else 240
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
