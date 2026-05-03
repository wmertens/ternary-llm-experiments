"""Cache teacher continuations + top-K logits for distillation.

Generates self-text from the FP teacher with min-P sampling.
`model.generate(output_logits=True)` returns the raw next-token logits
that the teacher used at every sampling step, so we extract top-K + an
explicit `rest_mass` scalar inline -- no separate forward pass.

Each shard is a safetensors file with:
  tokens     : int32  [S, T]            input token ids the student sees
  topk_idx   : int32  [S, T, K]         teacher's top-K vocab indices
  topk_prob  : fp16   [S, T, K]         teacher probs at those indices
  rest_mass  : fp16   [S, T]            1 - sum(topk_prob) at each position

The first `seed_len` positions and the final position have no teacher
target (we don't have the teacher's distribution for them); their entries
are zero, which contributes 0 to the KL loss (no gradient).  `seed_len`
is recorded in the safetensors metadata.
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
def generate_batch_with_topk(
    model, tokenizer, batch_size: int, seq_len: int, device: str,
    rng: random.Random, temperature: float, min_p: float,
    seed_token_ids: list[int], top_k: int, chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Generate one batch and extract top-K from the sampling-time logits.

    Returns (tokens [B,T] int64, topk_idx [B,T,K] int32,
             topk_prob [B,T,K] fp16, rest_mass [B,T] fp16, seed_len int).
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
    attn = torch.ones_like(ids)

    result = model.generate(
        input_ids=ids,
        attention_mask=attn,
        max_new_tokens=seq_len - seed_len,
        do_sample=True,
        temperature=temperature,
        min_p=min_p,
        pad_token_id=pad,
        return_dict_in_generate=True,
        output_logits=True,
    )

    seqs = result.sequences  # [B, gen_len]
    if seqs.shape[1] < seq_len:
        pad_block = torch.full((batch_size, seq_len - seqs.shape[1]), pad,
                               dtype=seqs.dtype, device=seqs.device)
        seqs = torch.cat([seqs, pad_block], dim=1)
    else:
        seqs = seqs[:, :seq_len]

    out_idx = torch.zeros((batch_size, seq_len, top_k), dtype=torch.int32)
    out_prob = torch.zeros((batch_size, seq_len, top_k), dtype=torch.float16)
    out_rest = torch.zeros((batch_size, seq_len), dtype=torch.float16)

    # result.logits is a tuple of [B, V] tensors, length == #generated.
    # logits[i] is the distribution that produced the token at absolute
    # position seed_len + i, i.e. it represents the teacher's prediction
    # at sequence position seed_len + i - 1.  Skip pos 0..seed_len-2 and
    # pos T-1 (no teacher info available without an extra forward pass).
    logits_list: list[torch.Tensor | None] = list(result.logits)
    n_gen = len(logits_list)
    write_start = seed_len - 1
    for s in range(0, n_gen, chunk):
        e = min(s + chunk, n_gen)
        ch = torch.stack(logits_list[s:e], dim=1).float()      # [B, ch, V]
        probs = torch.softmax(ch, dim=-1)
        vals, idx = torch.topk(probs, k=top_k, dim=-1)
        rest = (1.0 - vals.sum(dim=-1)).clamp_min(0.0)
        out_idx[:, write_start + s : write_start + e] = idx.to(torch.int32).cpu()
        out_prob[:, write_start + s : write_start + e] = vals.to(torch.float16).cpu()
        out_rest[:, write_start + s : write_start + e] = rest.to(torch.float16).cpu()
        for i in range(s, e):
            logits_list[i] = None
        del ch, probs, vals, idx, rest

    return seqs.cpu(), out_idx, out_prob, out_rest, seed_len


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache teacher top-K logits.")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--total-tokens", type=int, default=10_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--gen-batch", type=int, default=16,
                    help="Sequences generated in parallel.")
    ap.add_argument("--logit-chunk", type=int, default=64,
                    help="Positions per softmax/topk chunk on GPU.")
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--shard-seqs", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device)
    model.eval()

    seed_ids = _seed_token_ids(tok)
    n_seqs = math.ceil(args.total_tokens / args.seq_len)
    n_shards = math.ceil(n_seqs / args.shard_seqs)

    pbar = tqdm(total=n_seqs, desc="cache", unit="seq")
    seqs_done = 0
    seed_len_meta = 0
    for shard_idx in range(n_shards):
        in_shard = min(args.shard_seqs, n_seqs - seqs_done)
        token_chunks: list[torch.Tensor] = []
        idx_chunks: list[torch.Tensor] = []
        prob_chunks: list[torch.Tensor] = []
        rest_chunks: list[torch.Tensor] = []
        gathered = 0
        while gathered < in_shard:
            bs = min(args.gen_batch, in_shard - gathered)
            t, i, p, r, seed_len_meta = generate_batch_with_topk(
                model, tok, bs, args.seq_len, args.device, rng,
                args.temperature, args.min_p, seed_ids,
                args.top_k, args.logit_chunk,
            )
            token_chunks.append(t)
            idx_chunks.append(i)
            prob_chunks.append(p)
            rest_chunks.append(r)
            gathered += bs
            pbar.update(bs)
        save_file({
            "tokens": torch.cat(token_chunks, dim=0).to(torch.int32).contiguous(),
            "topk_idx": torch.cat(idx_chunks, dim=0).contiguous(),
            "topk_prob": torch.cat(prob_chunks, dim=0).contiguous(),
            "rest_mass": torch.cat(rest_chunks, dim=0).contiguous(),
        }, str(args.out / f"shard_{shard_idx:05d}.safetensors"),
            metadata={"seed_len": str(seed_len_meta), "top_k": str(args.top_k)})
        seqs_done += in_shard
    pbar.close()


if __name__ == "__main__":
    main()
