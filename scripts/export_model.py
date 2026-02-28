#!/usr/bin/env python3
"""Export cat re-identification model to ONNX and optionally TensorRT."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging
from app.ml.training.export import export_to_onnx, export_to_tensorrt


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX/TensorRT")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint")
    parser.add_argument("--onnx-output", help="ONNX output path")
    parser.add_argument("--trt-output", help="TensorRT output path")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 for TensorRT")
    args = parser.parse_args()

    setup_logging()

    checkpoint = Path(args.checkpoint)
    onnx_path = args.onnx_output or str(checkpoint.with_suffix(".onnx"))

    print(f"Exporting {checkpoint} -> {onnx_path}")
    export_to_onnx(str(checkpoint), onnx_path)

    if args.trt_output:
        print(f"Exporting {onnx_path} -> {args.trt_output}")
        export_to_tensorrt(onnx_path, args.trt_output, fp16=not args.no_fp16)


if __name__ == "__main__":
    main()
