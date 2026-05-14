#!/usr/bin/env python3
"""Simple ONNX quantization helper.

Modes:
 - dynamic/int8 : dynamic INT8 quantization using onnxruntime.quantization.quantize_dynamic (no calibration needed)
 - fp16/float16 : convert float32 weights to float16 using onnxconverter_common (requires onnxconverter-common)

Examples:
  & .venv\Scripts\python.exe scripts/quantize_onnx.py --input models/best_model.onnx --output models/best_model.int8.onnx --mode dynamic
  & .venv\Scripts\python.exe scripts/quantize_onnx.py --input models/best_model.onnx --output models/best_model.fp16.onnx --mode fp16
"""

import argparse
from pathlib import Path
import sys
import os


def human_size(path: Path):
    try:
        s = path.stat().st_size
        for unit in ['B','KB','MB','GB']:
            if s < 1024.0:
                return f"{s:.1f}{unit}"
            s /= 1024.0
        return f"{s:.1f}TB"
    except Exception:
        return "?"


def quantize_dynamic_int8(inp: Path, out: Path, weight_type: str = 'qint8', per_channel: bool = False, reduce_range: bool = False):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        raise RuntimeError("onnxruntime.quantization not available. Install a recent onnxruntime: pip install onnxruntime") from e

    qt = QuantType.QInt8 if weight_type.lower() in ('qint8','int8') else QuantType.QUInt8
    print(f"Running dynamic quantization (weight_type={qt}, per_channel={per_channel}, reduce_range={reduce_range})")
    quantize_dynamic(str(inp), str(out), weight_type=qt, per_channel=per_channel, reduce_range=reduce_range)


def convert_fp16(inp: Path, out: Path, keep_io_types: bool = False):
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception as e:
        raise RuntimeError("FP16 conversion requires 'onnxconverter-common'. Install with: pip install onnxconverter-common") from e

    print("Loading ONNX model...")
    model = onnx.load(str(inp))
    print("Converting weights to float16 (this may lose small precision)...")
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=keep_io_types)
    onnx.save(model_fp16, str(out))


def main():
    p = argparse.ArgumentParser(description="Quantize ONNX model (dynamic INT8 or FP16).")
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['dynamic','int8','fp16','float16'], default='dynamic')
    p.add_argument('--weight-type', choices=['qint8','quint8'], default='qint8', help='Weight type for dynamic quantization')
    p.add_argument('--per-channel', action='store_true', help='Enable per-channel quantization (recommended for conv/linear layers when supported)')
    p.add_argument('--reduce-range', action='store_true', help='Use reduced range (uint8 -> int8 mapping)')
    p.add_argument('--keep-io-types', action='store_true', help='When converting to fp16, keep model I/O dtypes as float32')
    args = p.parse_args()

    inp = args.input
    out = args.output
    if not inp.exists():
        print(f"Input not found: {inp}")
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    before = human_size(inp)

    try:
        if args.mode in ('dynamic','int8'):
            quantize_dynamic_int8(inp, out, weight_type=args.weight_type, per_channel=args.per_channel, reduce_range=args.reduce_range)
        else:
            convert_fp16(inp, out, keep_io_types=args.keep_io_types)
    except Exception as e:
        print("Quantization/conversion failed:", e)
        sys.exit(2)

    after = human_size(out)
    print(f"Wrote {out} (size {after}, before {before})")


if __name__ == '__main__':
    main()
