#!/usr/bin/env python3
"""Benchmark inference speed for detection and identification models."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_detector(model_path: str, iterations: int = 100):
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Warmup
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy, verbose=False)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"YOLO Detection ({model_path}):")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  FPS:    {1000 / times.mean():.1f}")


def benchmark_identifier(model_path: str, iterations: int = 100):
    import torch
    from app.ml.identifier import CatReIDModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatReIDModel()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()

    # Warmup
    dummy = torch.randn(1, 3, 256, 256).to(device)
    for _ in range(10):
        with torch.no_grad():
            model(dummy)

    # Benchmark
    times = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"\nCat Re-ID ({model_path}):")
    print(f"  Device: {device}")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  FPS:    {1000 / times.mean():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--detector", default="yolov8s.pt")
    parser.add_argument("--identifier", help="Path to cat re-id checkpoint")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    benchmark_detector(args.detector, args.iterations)
    if args.identifier:
        benchmark_identifier(args.identifier, args.iterations)
