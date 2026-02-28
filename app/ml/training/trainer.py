import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.identifier import CatReIDModel
from app.ml.training.augmentation import get_train_transforms, get_val_transforms
from app.ml.training.dataset import CatReIDDataset, TripletBatchSampler
from app.ml.training.evaluate import evaluate_model

logger = get_logger(__name__)


def online_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3
) -> torch.Tensor:
    """Batch-hard triplet loss with online mining."""
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    # For each anchor, find hardest positive
    mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos.fill_diagonal_(False)
    # Set non-positive distances to -inf for max
    pos_dists = pairwise_dist.clone()
    pos_dists[~mask_pos] = -1.0
    hardest_positive, _ = pos_dists.max(dim=1)

    # For each anchor, find hardest negative
    mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)
    neg_dists = pairwise_dist.clone()
    neg_dists[~mask_neg] = float("inf")
    hardest_negative, _ = neg_dists.min(dim=1)

    loss = torch.clamp(hardest_positive - hardest_negative + margin, min=0.0)
    return loss.mean()


class CatReIDTrainer:
    """Training loop for cat re-identification model."""

    def __init__(
        self,
        data_dir: str,
        epochs: int = 50,
        learning_rate: float = 0.001,
        batch_p: int = 3,
        batch_k: int = 8,
        freeze_epochs: int = 10,
        margin: float = 0.3,
    ):
        self.data_dir = data_dir
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_p = batch_p
        self.batch_k = batch_k
        self.freeze_epochs = freeze_epochs
        self.margin = margin
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cancel = False

    def cancel(self):
        """Request cancellation of the current training run."""
        self._cancel = True
        logger.info("Training cancellation requested")

    def train(self, progress_callback=None) -> dict:
        """Run the full training pipeline. Returns training results dict."""
        logger.info("Starting Cat Re-ID training on %s", self.device)

        # Load datasets
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()

        train_dataset, cat_names = CatReIDDataset.from_directory(
            str(Path(self.data_dir) / "train"), transform=train_transform
        )
        val_dataset, _ = CatReIDDataset.from_directory(
            str(Path(self.data_dir) / "val"), transform=val_transform
        )

        logger.info(
            "Training data: %d images, %d cats (%s)",
            len(train_dataset),
            len(cat_names),
            ", ".join(cat_names),
        )

        train_sampler = TripletBatchSampler(train_dataset.labels, p=self.batch_p, k=self.batch_k)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        # Model
        model = CatReIDModel(embedding_dim=512).to(self.device)

        # Load pretrained ResNet50 backbone
        from torchvision.models import ResNet50_Weights
        pretrained = torch.hub.load("pytorch/vision", "resnet50", weights=ResNet50_Weights.DEFAULT)
        backbone_state = {k: v for k, v in pretrained.state_dict().items() if not k.startswith("fc.")}
        missing, unexpected = model.features.load_state_dict(backbone_state, strict=False)
        logger.info("Loaded pretrained ResNet50 backbone (missing=%d, unexpected=%d)", len(missing), len(unexpected))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = GradScaler()

        best_metric = 0.0
        loss_history = []
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_dir = Path(settings.MODELS_DIR) / "identification"
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = str(save_dir / f"cat_reid_{version}_best.pth")

        # Phase 1: Freeze backbone
        self._freeze_backbone(model)

        for epoch in range(1, self.epochs + 1):
            if self._cancel:
                logger.info("Training cancelled at epoch %d", epoch)
                break

            # Phase 2: Unfreeze last 2 blocks after freeze_epochs
            if epoch == self.freeze_epochs + 1:
                self._unfreeze_last_blocks(model, num_blocks=2)
                logger.info("Unfroze last 2 ResNet blocks at epoch %d", epoch)

            # Train
            model.train()
            epoch_losses = []
            for batch_imgs, batch_labels in train_loader:
                batch_imgs = batch_imgs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                with autocast():
                    embeddings = model(batch_imgs)
                    loss = online_hard_triplet_loss(embeddings, batch_labels, self.margin)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            scheduler.step()

            # Evaluate
            rank1, mAP = evaluate_model(model, val_loader, self.device)
            loss_history.append({"epoch": epoch, "loss": avg_loss, "rank1": rank1, "mAP": mAP})

            logger.info(
                "Epoch %d/%d | Loss: %.4f | Rank-1: %.4f | mAP: %.4f",
                epoch, self.epochs, avg_loss, rank1, mAP,
            )

            # Save best
            if rank1 > best_metric:
                best_metric = rank1
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cat_names": cat_names,
                        "embedding_dim": 512,
                        "version": version,
                        "epoch": epoch,
                        "rank1": rank1,
                        "mAP": mAP,
                    },
                    best_path,
                )
                logger.info("Saved best model (Rank-1: %.4f)", rank1)

            if progress_callback:
                progress_callback(epoch, self.epochs, avg_loss, rank1, mAP)

        return {
            "version": version,
            "model_path": best_path,
            "best_rank1": best_metric,
            "loss_history": loss_history,
            "cat_names": cat_names,
        }

    def _freeze_backbone(self, model: CatReIDModel):
        """Freeze all backbone parameters."""
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for initial training phase")

    def _unfreeze_last_blocks(self, model: CatReIDModel, num_blocks: int = 2):
        """Unfreeze the last N blocks of the ResNet backbone."""
        children = list(model.features.children())
        # ResNet blocks are layer1-layer4 (indices 4-7 in Sequential)
        for child in children[-(num_blocks + 1):]:
            for param in child.parameters():
                param.requires_grad = True
