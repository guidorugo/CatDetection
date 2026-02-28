import numpy as np
import pytest

from app.ml.embeddings import EmbeddingStore


def test_add_and_find_match():
    store = EmbeddingStore()
    emb1 = np.random.randn(512).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)
    store.add_cat(1, "Mimir", [emb1])

    # Query with same embedding should match
    cat_id, name, sim = store.find_match(emb1, threshold=0.5)
    assert cat_id == 1
    assert name == "Mimir"
    assert sim > 0.99


def test_no_match_below_threshold():
    store = EmbeddingStore()
    emb1 = np.array([1.0] + [0.0] * 511, dtype=np.float32)
    store.add_cat(1, "Mimir", [emb1])

    # Query with orthogonal embedding
    query = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)
    cat_id, name, sim = store.find_match(query, threshold=0.5)
    assert cat_id is None
    assert name is None


def test_multiple_cats():
    store = EmbeddingStore()
    emb_mimir = np.random.randn(512).astype(np.float32)
    emb_mimir = emb_mimir / np.linalg.norm(emb_mimir)
    emb_rolo = np.random.randn(512).astype(np.float32)
    emb_rolo = emb_rolo / np.linalg.norm(emb_rolo)

    store.add_cat(1, "Mimir", [emb_mimir])
    store.add_cat(2, "Rolo", [emb_rolo])

    # Small perturbation to Mimir's embedding
    query = emb_mimir + np.random.randn(512).astype(np.float32) * 0.01
    cat_id, name, _ = store.find_match(query, threshold=0.5)
    assert name == "Mimir"


def test_remove_cat():
    store = EmbeddingStore()
    emb = np.ones(512, dtype=np.float32) / np.sqrt(512)
    store.add_cat(1, "Mimir", [emb])
    assert store.cat_count == 1

    store.remove_cat(1)
    assert store.cat_count == 0


def test_clear():
    store = EmbeddingStore()
    store.add_cat(1, "A", [np.zeros(512, dtype=np.float32)])
    store.add_cat(2, "B", [np.zeros(512, dtype=np.float32)])
    store.clear()
    assert store.cat_count == 0
    assert store.total_embeddings == 0
