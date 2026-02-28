import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingStore:
    """In-memory store for cat embeddings with cosine similarity matching."""

    def __init__(self):
        # cat_id -> list of embeddings (numpy arrays)
        self._embeddings: dict[int, list[np.ndarray]] = {}
        # cat_id -> cat_name
        self._names: dict[int, str] = {}

    def add_cat(self, cat_id: int, name: str, embeddings: list[np.ndarray]):
        """Register a cat with its embeddings."""
        self._embeddings[cat_id] = embeddings
        self._names[cat_id] = name
        logger.info("Added cat '%s' (id=%d) with %d embeddings", name, cat_id, len(embeddings))

    def remove_cat(self, cat_id: int):
        """Remove a cat from the store."""
        self._embeddings.pop(cat_id, None)
        self._names.pop(cat_id, None)

    def clear(self):
        """Remove all cats."""
        self._embeddings.clear()
        self._names.clear()

    def find_match(
        self, query_embedding: np.ndarray, threshold: float | None = None
    ) -> tuple[int | None, str | None, float]:
        """Find the best matching cat for a query embedding.

        Returns (cat_id, cat_name, similarity) or (None, None, 0.0) if no match above threshold.
        """
        if threshold is None:
            threshold = settings.IDENTIFICATION_THRESHOLD

        best_cat_id = None
        best_name = None
        best_similarity = 0.0

        query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        for cat_id, embeddings in self._embeddings.items():
            for emb in embeddings:
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                similarity = float(np.dot(query, emb_norm))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cat_id = cat_id
                    best_name = self._names.get(cat_id)

        if best_similarity >= threshold:
            return best_cat_id, best_name, best_similarity
        return None, None, best_similarity

    @property
    def cat_count(self) -> int:
        return len(self._embeddings)

    @property
    def total_embeddings(self) -> int:
        return sum(len(v) for v in self._embeddings.values())
