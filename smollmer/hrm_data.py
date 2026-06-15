"""hrm_data — weighted streaming mix DataLoader for hrm_bop.

Three HF datasets (fineweb-edu, cosmopedia-v2, OpenMathInstruct-2) streamed
in parallel; each batch position draws a dataset by weight then pulls the
next packed seq_len chunk from that source's iterator. Per-dataset
iterators are sharded by `(worker_id, num_workers)` so DataLoader workers
don't duplicate data.

python-edu was dropped from the spec mix — it ships as metadata-only on HF
(no `text` field), confirmed by a streaming probe at impl time. The
fallback mix recorded in the spec (70/25/5) is what we use.

Held-out validation: a separate `eval_iter` over fineweb-edu sample-100BT
that's never touched by training (a different sub-shard).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch.utils.data import IterableDataset

# A "formatter" turns one dataset example into one raw text string.
Formatter = Callable[[dict], str]


def _fmt_text(ex: dict) -> str:
    t = ex.get("text", "")
    return t if isinstance(t, str) else ""


def _fmt_openmath(ex: dict) -> str:
    p = ex.get("problem", "") or ""
    s = ex.get("generated_solution", "") or ""
    if not p or not s:
        return ""
    return f"Problem: {p}\n\nSolution: {s}"


@dataclass
class MixComponent:
    name: str
    dataset_name: str
    dataset_config: str | None
    split: str
    weight: float
    formatter: Formatter


def _mk_fineweb(weight: float) -> MixComponent:
    return MixComponent("fineweb-edu", "HuggingFaceFW/fineweb-edu",
                        "sample-10BT", "train", weight, _fmt_text)


def _mk_cosmo(weight: float) -> MixComponent:
    return MixComponent("cosmopedia-v2", "HuggingFaceTB/smollm-corpus",
                        "cosmopedia-v2", "train", weight, _fmt_text)


def _mk_openmath(weight: float) -> MixComponent:
    return MixComponent("openmath", "nvidia/OpenMathInstruct-2",
                        None, "train", weight, _fmt_openmath)


_SOURCE_FACTORIES = {
    "fineweb": _mk_fineweb,
    "cosmopedia": _mk_cosmo,
    "openmath": _mk_openmath,
}


def build_mix(weights: dict[str, float]) -> list[MixComponent]:
    """Build a mix from a {source_name: weight} dict. Zero-weighted sources
    are dropped so we don't open useless streams. Unknown source names raise.
    """
    out: list[MixComponent] = []
    for name, w in weights.items():
        if name not in _SOURCE_FACTORIES:
            raise ValueError(f"unknown data source {name!r}; "
                             f"known: {list(_SOURCE_FACTORIES)}")
        if w > 0:
            out.append(_SOURCE_FACTORIES[name](float(w)))
    if not out:
        raise ValueError("build_mix produced an empty mix")
    return out


DEFAULT_MIX: list[MixComponent] = build_mix(
    {"fineweb": 0.70, "cosmopedia": 0.25, "openmath": 0.05})


def _packed_chunks(text_iter: Iterator[str], tokenizer, seq_len: int,
                   eos_token_id: int, min_chars: int = 50) -> Iterator[list[int]]:
    """Tokenize → eos-separate → pack into fixed `seq_len` chunks. Discards
    the trailing tail of each accumulation pass — packing is per-call, not
    cross-call, to keep the iterator stateless from the caller's view.
    """
    buf: list[int] = []
    BATCH_TARGET = seq_len * 16          # tokenize in modest batches
    while True:
        # Refill the int buffer until we have at least one chunk's worth.
        while len(buf) < seq_len:
            try:
                text = next(text_iter)
            except StopIteration:
                # Caller restarts via _build_iter on the outer loop.
                return
            if not text or len(text) < min_chars:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos_token_id)
            if len(buf) > BATCH_TARGET:
                break
        # Yield as many full chunks as we have, keep the remainder.
        n_full = len(buf) // seq_len
        for i in range(n_full):
            yield buf[i * seq_len:(i + 1) * seq_len]
        buf = buf[n_full * seq_len:]


class MixedStream(IterableDataset):
    """Weighted multi-source streaming mix.

    Each `__iter__` call (one per worker) builds:
      - one streaming iterator per component, sharded by (wid, nworkers)
      - one packed-chunk iterator per component
      - a per-worker rng so weighted draws are deterministic given (seed, wid)

    Yields dicts: {"input_ids": LongTensor[seq_len]}.

    `start_skip` advances the per-component packed-chunk iterators by their
    expected share of the global skip, so resume produces (approximately)
    the same data the run would have seen without interruption. "Approximately"
    because weighted-random draws make exact replay impossible without
    storing the per-source skip counts — the spec accepts this drift as
    "good enough for streaming pretraining".
    """

    def __init__(self, components: list[MixComponent], tokenizer,
                 seq_len: int, seed: int = 0,
                 start_skip: int = 0,
                 eos_token_id: int | None = None) -> None:
        self.components = components
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.start_skip = int(start_skip)
        if eos_token_id is None:
            eos_token_id = (tokenizer.eos_token_id
                            if tokenizer.eos_token_id is not None
                            else tokenizer.pad_token_id)
        if eos_token_id is None:
            raise ValueError("tokenizer has neither eos nor pad token")
        self.eos_token_id = int(eos_token_id)
        self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(c.weight for c in self.components)
        if total <= 0:
            raise ValueError("mix weights sum to zero")

    def _build_text_iter(self, comp: MixComponent, wid: int, nworkers: int):
        from datasets import load_dataset
        ds = load_dataset(comp.dataset_name, comp.dataset_config,
                          split=comp.split, streaming=True)
        if nworkers > 1:
            ds = ds.shard(num_shards=nworkers, index=wid, contiguous=True)
        fmt = comp.formatter
        # Wrap to a flat text iterator.
        for ex in ds:
            yield fmt(ex)

    def _build_text_iter_restart(self, comp, wid, nworkers):
        """Generator that restarts when the underlying stream finishes."""
        while True:
            yield from self._build_text_iter(comp, wid, nworkers)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1
        rng = random.Random(self.seed * 1_000_003 + wid * 7919)

        text_iters = [self._build_text_iter_restart(c, wid, nworkers)
                      for c in self.components]
        chunk_iters = [_packed_chunks(ti, self.tokenizer, self.seq_len,
                                      self.eos_token_id)
                       for ti in text_iters]

        weights = [c.weight for c in self.components]
        # Per-worker skip, split evenly across workers.
        my_skip = (self.start_skip // nworkers
                   + (1 if wid < (self.start_skip % nworkers) else 0))

        produced = 0
        while True:
            comp_idx = rng.choices(range(len(self.components)), weights)[0]
            try:
                ids = next(chunk_iters[comp_idx])
            except StopIteration:
                # Rebuild this component's iterators if exhausted.
                text_iters[comp_idx] = self._build_text_iter_restart(
                    self.components[comp_idx], wid, nworkers)
                chunk_iters[comp_idx] = _packed_chunks(
                    text_iters[comp_idx], self.tokenizer, self.seq_len,
                    self.eos_token_id)
                ids = next(chunk_iters[comp_idx])
            if produced < my_skip:
                produced += 1
                continue
            produced += 1
            yield {"input_ids": torch.tensor(ids, dtype=torch.long)}


def make_train_loader(tokenizer, seq_len: int, batch_size: int,
                      seed: int, num_workers: int,
                      start_skip: int = 0,
                      components: list[MixComponent] | None = None):
    """Returns a DataLoader yielding {"input_ids": LongTensor[B, seq_len]}."""
    from torch.utils.data import DataLoader
    comps = components if components is not None else DEFAULT_MIX
    ds = MixedStream(comps, tokenizer=tokenizer, seq_len=seq_len,
                     seed=seed, start_skip=start_skip)

    def _worker_init(_wid: int) -> None:
        import signal as _sig
        _sig.signal(_sig.SIGINT, _sig.SIG_IGN)

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=True, worker_init_fn=_worker_init)


def make_val_loader(tokenizer, seq_len: int, batch_size: int,
                    n_batches: int = 16,
                    source: str = "fineweb") -> list[dict]:
    """Eager-load `n_batches` of held-out batches into RAM, so val is a
    deterministic fixed sample across the entire run.

    `source` selects the validation distribution:
      - "fineweb": fineweb-edu `sample-100BT` (different sub-shard from
        training's `sample-10BT`) — clean held-out.
      - "cosmopedia" / "openmath": same underlying split as training, but
        different seed + start_skip=1000 to randomise the draw. Some
        sequence overlap with training is possible (no second sub-shard
        exists for these); acceptable for diagnostic val on a curriculum
        phase B.
    """
    if source == "fineweb":
        comp = MixComponent("val-fineweb-edu", "HuggingFaceFW/fineweb-edu",
                            "sample-100BT", "train", 1.0, _fmt_text)
    elif source == "cosmopedia":
        comp = MixComponent("val-cosmopedia-v2", "HuggingFaceTB/smollm-corpus",
                            "cosmopedia-v2", "train", 1.0, _fmt_text)
    elif source == "openmath":
        comp = MixComponent("val-openmath", "nvidia/OpenMathInstruct-2",
                            None, "train", 1.0, _fmt_openmath)
    else:
        raise ValueError(f"unknown val source {source!r}; "
                         f"known: fineweb, cosmopedia, openmath")
    ds = MixedStream([comp], tokenizer=tokenizer, seq_len=seq_len,
                     seed=999_983,    # different from train seed
                     start_skip=1000)
    batches: list[dict] = []
    cur: list[torch.Tensor] = []
    for sample in ds:
        cur.append(sample["input_ids"])
        if len(cur) == batch_size:
            batches.append({"input_ids": torch.stack(cur)})
            cur = []
            if len(batches) >= n_batches:
                break
    return batches
