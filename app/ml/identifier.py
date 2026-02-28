import asyncio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from app.core.logging import get_logger

logger = get_logger(__name__)


class CatReIDModel(nn.Module):
    """ResNet50-based cat re-identification model producing 512-d L2-normalized embeddings."""

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        backbone = models.resnet50(weights=None)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.embed = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embed(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class CatIdentifier:
    """Wrapper for cat re-identification inference."""

    def __init__(self):
        self.model: CatReIDModel | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    async def load(self, model_path: str):
        """Load a trained model checkpoint."""
        logger.info("Loading Cat Re-ID model from %s", model_path)
        self.model = await asyncio.to_thread(self._load_model, model_path)
        logger.info("Cat Re-ID model loaded on %s", self.device)

    def _load_model(self, model_path: str) -> CatReIDModel:
        model = CatReIDModel()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    async def get_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Get 512-d embedding from a cropped cat image (BGR numpy array)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return await asyncio.to_thread(self._get_embedding, crop)

    def _get_embedding(self, crop: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        return embedding.cpu().numpy()[0]
