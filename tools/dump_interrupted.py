"""dump_interrupted.py — extract weights from an interrupted.pt into a
safetensors file. Discards optimiser state and EMA tracker; keeps the
model state_dict (suitable for deployment / warm-restart with fresh opt).

Usage:
    python -m tools.dump_interrupted \\
        --in  experiments_gpt/g020.../interrupted.pt \\
        --out experiments_gpt/g020.../final.safetensors \\
        [--cfg hidden_size=512,num_layers=6,...]   # optional metadata

If --cfg is omitted the safetensors is written without the model-shape
metadata. Most consumers don't need it (they reconstruct cfg from CLI
flags), but include it for portability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=Path, required=True,
                    help="path to interrupted.pt")
    ap.add_argument("--out", dest="dst", type=Path, required=True,
                    help="output safetensors path")
    ap.add_argument("--cfg", type=str, default=None,
                    help="optional comma-separated key=value pairs to "
                         "embed in the safetensors metadata")
    args = ap.parse_args()

    payload = torch.load(args.src, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise SystemExit(
            f"{args.src} has no 'model' key; not a trainer interrupted.pt")
    model_state = payload["model"]
    sd: dict[str, torch.Tensor] = {}
    seen: dict[int, str] = {}
    for k, v in model_state.items():
        t = v.detach().cpu().contiguous()
        ptr = t.data_ptr()
        if ptr in seen:
            t = t.clone()
        else:
            seen[ptr] = k
        sd[k] = t

    meta = {"format": "dumped-from-interrupted"}
    if args.cfg:
        for piece in args.cfg.split(","):
            if not piece.strip():
                continue
            k, _, v = piece.partition("=")
            meta[k.strip()] = v.strip()
    if "next_step" in payload:
        meta["dumped_at_step"] = str(payload["next_step"])
    if "run_name" in payload and payload["run_name"]:
        meta["run_name"] = str(payload["run_name"])

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.dst.with_suffix(args.dst.suffix + ".tmp")
    save_file(sd, str(tmp), metadata=meta)
    tmp.replace(args.dst)
    print(f"wrote {args.dst} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
