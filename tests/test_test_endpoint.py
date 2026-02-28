from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from httpx import AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_detect_requires_auth(client: AsyncClient):
    """Upload without auth should return 401."""
    resp = await client.post("/api/v1/test/detect", files={"file": ("cat.jpg", b"fake", "image/jpeg")})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_detect_no_pipeline(client: AsyncClient, auth_headers):
    """Upload with no pipeline initialized should return 503."""
    # Ensure no detector on app state
    app.state.detector = None
    app.state.embedding_store = None
    try:
        resp = await client.post(
            "/api/v1/test/detect",
            headers=auth_headers,
            files={"file": ("cat.jpg", b"fake", "image/jpeg")},
        )
        assert resp.status_code == 503
        assert "pipeline" in resp.json()["detail"].lower()
    finally:
        if hasattr(app.state, "detector"):
            del app.state.detector
        if hasattr(app.state, "embedding_store"):
            del app.state.embedding_store


@pytest.mark.asyncio
async def test_detect_with_mocked_pipeline(client: AsyncClient, auth_headers):
    """Upload with a valid image and mocked ML components should return detections."""
    # Create a small valid JPEG image
    import cv2

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [200, 150, 100]  # colored rectangle
    _, img_bytes = cv2.imencode(".jpg", img)

    # Mock detector
    mock_detector = AsyncMock()
    mock_detector.detect.return_value = [
        {"bbox": (10, 10, 50, 50), "confidence": 0.95, "class_id": 15},
        {"bbox": (60, 10, 30, 30), "confidence": 0.88, "class_id": 15},
    ]

    # Mock identifier
    mock_identifier = AsyncMock()
    mock_identifier.model = True  # truthy = model loaded
    embedding_1 = np.random.randn(512).astype(np.float32)
    embedding_2 = np.random.randn(512).astype(np.float32)
    mock_identifier.get_embedding.side_effect = [embedding_1, embedding_2]

    # Mock embedding store
    mock_embedding_store = MagicMock()
    mock_embedding_store.find_match.side_effect = [
        (1, "Whiskers", 0.87),
        (None, None, 0.32),
    ]

    app.state.detector = mock_detector
    app.state.identifier = mock_identifier
    app.state.embedding_store = mock_embedding_store

    try:
        resp = await client.post(
            "/api/v1/test/detect",
            headers=auth_headers,
            files={"file": ("cat.jpg", img_bytes.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert len(data["detections"]) == 2

        # First detection: matched
        d0 = data["detections"][0]
        assert d0["cat_id"] == 1
        assert d0["cat_name"] == "Whiskers"
        assert d0["confidence"] == 0.95
        assert d0["similarity"] == 0.87
        assert d0["bbox"] == [10, 10, 50, 50]

        # Second detection: unmatched
        d1 = data["detections"][1]
        assert d1["cat_id"] is None
        assert d1["cat_name"] is None
        assert d1["similarity"] == 0.32
    finally:
        del app.state.detector
        del app.state.identifier
        del app.state.embedding_store


@pytest.mark.asyncio
async def test_detect_invalid_image(client: AsyncClient, auth_headers):
    """Upload an invalid file should return 400."""
    mock_detector = AsyncMock()
    mock_embedding_store = MagicMock()
    app.state.detector = mock_detector
    app.state.embedding_store = mock_embedding_store

    try:
        resp = await client.post(
            "/api/v1/test/detect",
            headers=auth_headers,
            files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "invalid" in resp.json()["detail"].lower()
    finally:
        del app.state.detector
        del app.state.embedding_store


@pytest.mark.asyncio
async def test_detect_no_cats_found(client: AsyncClient, auth_headers):
    """Upload an image where detector finds no cats should return empty list."""
    import cv2

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode(".jpg", img)

    mock_detector = AsyncMock()
    mock_detector.detect.return_value = []
    mock_embedding_store = MagicMock()

    app.state.detector = mock_detector
    app.state.embedding_store = mock_embedding_store

    try:
        resp = await client.post(
            "/api/v1/test/detect",
            headers=auth_headers,
            files={"file": ("empty.jpg", img_bytes.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["detections"] == []
    finally:
        del app.state.detector
        del app.state.embedding_store
