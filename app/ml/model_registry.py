import json
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

REGISTRY_FILE = "registry.json"


class ModelRegistry:
    """Tracks model versions and provides loading paths."""

    def __init__(self):
        self.base_dir = Path(settings.MODELS_DIR) / "identification"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.base_dir / REGISTRY_FILE

    def _load_registry(self) -> dict:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text())
        return {"models": {}, "active": None}

    def _save_registry(self, data: dict):
        self._registry_path.write_text(json.dumps(data, indent=2))

    def register_model(self, version: str, path: str, metrics: dict | None = None):
        """Register a new model version."""
        registry = self._load_registry()
        registry["models"][version] = {
            "path": path,
            "metrics": metrics or {},
        }
        # Auto-activate if first model
        if registry["active"] is None:
            registry["active"] = version
        self._save_registry(registry)
        logger.info("Registered model version %s at %s", version, path)

    def activate_model(self, version: str) -> str:
        """Set the active model version. Returns model path."""
        registry = self._load_registry()
        if version not in registry["models"]:
            raise ValueError(f"Model version '{version}' not found")
        registry["active"] = version
        self._save_registry(registry)
        logger.info("Activated model version %s", version)
        return registry["models"][version]["path"]

    def get_active_model_path(self) -> str | None:
        """Get the path of the active model."""
        registry = self._load_registry()
        active = registry.get("active")
        if active and active in registry["models"]:
            return registry["models"][active]["path"]
        return None

    def get_active_version(self) -> str | None:
        """Get the active model version string."""
        registry = self._load_registry()
        return registry.get("active")

    def list_models(self) -> dict:
        """List all registered models."""
        return self._load_registry()
