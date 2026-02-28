import json
import tempfile
from pathlib import Path

import pytest

from app.ml.model_registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a ModelRegistry with a temp directory."""
    from unittest.mock import patch
    from app.core.config import Settings

    settings = Settings(MODELS_DIR=str(tmp_path / "models"))
    with patch("app.ml.model_registry.settings", settings):
        reg = ModelRegistry()
        yield reg


def test_register_and_get_model(registry):
    registry.register_model("v1", "/path/to/model.pth", {"rank1": 0.95})
    assert registry.get_active_model_path() == "/path/to/model.pth"
    assert registry.get_active_version() == "v1"


def test_register_multiple_models(registry):
    registry.register_model("v1", "/path/v1.pth")
    registry.register_model("v2", "/path/v2.pth")

    # First model should still be active
    assert registry.get_active_version() == "v1"

    # Activate v2
    path = registry.activate_model("v2")
    assert path == "/path/v2.pth"
    assert registry.get_active_version() == "v2"


def test_activate_nonexistent_model(registry):
    with pytest.raises(ValueError, match="not found"):
        registry.activate_model("nonexistent")


def test_no_active_model_initially(registry):
    assert registry.get_active_model_path() is None
    assert registry.get_active_version() is None


def test_list_models(registry):
    registry.register_model("v1", "/path/v1.pth", {"rank1": 0.9})
    registry.register_model("v2", "/path/v2.pth", {"rank1": 0.95})

    models = registry.list_models()
    assert "v1" in models["models"]
    assert "v2" in models["models"]
    assert models["active"] == "v1"
