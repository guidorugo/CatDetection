import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler


class CatReIDDataset(Dataset):
    """Dataset for cat re-identification training.

    Expects directory structure:
        data/processed/{cat_name}/img_001.jpg
    """

    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        # Group indices by label for triplet sampling
        self.label_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        label = self.labels[idx]
        return img, label

    @classmethod
    def from_directory(cls, root_dir: str, transform=None, split_file: str | None = None):
        """Load dataset from directory structure."""
        root = Path(root_dir)
        cat_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

        image_paths = []
        labels = []
        cat_names = []

        for label_idx, cat_dir in enumerate(cat_dirs):
            cat_names.append(cat_dir.name)
            for img_path in sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.png")):
                image_paths.append(str(img_path))
                labels.append(label_idx)

        return cls(image_paths, labels, transform), cat_names


class TripletBatchSampler(Sampler):
    """Samples batches for triplet loss: P cats x K images each."""

    def __init__(self, labels: list[int], p: int = 3, k: int = 8):
        self.labels = labels
        self.p = p
        self.k = k
        self.label_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)
        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        # Shuffle labels each epoch
        labels = self.unique_labels.copy()
        random.shuffle(labels)

        # Number of batches
        num_batches = max(
            min(len(self.label_to_indices[l]) for l in labels) // self.k, 1
        ) * 2

        for _ in range(num_batches):
            batch = []
            selected_labels = random.sample(labels, min(self.p, len(labels)))
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    batch.extend(random.sample(indices, self.k))
                else:
                    batch.extend(random.choices(indices, k=self.k))
            yield batch

    def __len__(self):
        min_count = min(len(v) for v in self.label_to_indices.values())
        return max(min_count // self.k, 1) * 2
