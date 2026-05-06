"""Quantize a fp32 ONNX model with MatMulNBits weight-only quantization.

Mirrors what onnx-community / transformers.js use to produce `model_q4.onnx`,
`model_q2.onnx` etc. The output op (`MatMulNBits`) is supported by
onnxruntime-web on WebGPU.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="Input model.onnx (fp32 or fp16).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output model_qN.onnx path.")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 4, 8])
    ap.add_argument("--block-size", type=int, default=32,
                    help="Group size along the input dim. Smaller = better "
                    "accuracy, larger = smaller file.")
    ap.add_argument("--symmetric", action="store_true",
                    help="Symmetric quantization (recommended for ternary).")
    args = ap.parse_args()

    model = onnx.load(str(args.input), load_external_data=True)
    quant = MatMulNBitsQuantizer(
        model=model,
        bits=args.bits,
        block_size=args.block_size,
        is_symmetric=args.symmetric,
    )
    quant.process()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        quant.model.model,
        str(args.output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=args.output.name + "_data",
        size_threshold=1024,
    )
    print(f"[done] {args.output} (+{args.output.name}_data)")


if __name__ == "__main__":
    main()
