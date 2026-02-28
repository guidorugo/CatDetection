#!/usr/bin/env python3
"""CLI launcher for cat re-identification model training."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import setup_logging
from app.ml.model_registry import ModelRegistry
from app.ml.training.trainer import CatReIDTrainer


def main():
    parser = argparse.ArgumentParser(description="Train cat re-identification model")
    parser.add_argument("--data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-p", type=int, default=3, help="Cats per batch")
    parser.add_argument("--batch-k", type=int, default=8, help="Images per cat per batch")
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.3)
    args = parser.parse_args()

    setup_logging()

    trainer = CatReIDTrainer(
        data_dir=args.data_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_p=args.batch_p,
        batch_k=args.batch_k,
        freeze_epochs=args.freeze_epochs,
        margin=args.margin,
    )

    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Version:    {results['version']}")
    print(f"  Model path: {results['model_path']}")
    print(f"  Best Rank-1: {results['best_rank1']:.4f}")
    print(f"  Cats:       {', '.join(results['cat_names'])}")

    # Register with model registry
    registry = ModelRegistry()
    registry.register_model(
        version=results["version"],
        path=results["model_path"],
        metrics={"rank1": results["best_rank1"]},
    )
    print(f"\nModel registered as version {results['version']}")


if __name__ == "__main__":
    main()
