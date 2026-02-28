from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    APP_NAME: str = "CatDetect"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-to-a-random-secret-key"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./catdetect.db"

    # Auth
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # ML
    YOLO_MODEL: str = "yolov8s.pt"
    DETECTION_CONFIDENCE: float = 0.5
    IDENTIFICATION_THRESHOLD: float = 0.6
    DETECTION_FPS: int = 5

    # Cameras
    MAX_CAMERAS: int = 3

    # Recording
    RECORDING_PRE_ROLL_SECONDS: int = 5
    RECORDING_POST_ROLL_SECONDS: int = 10
    RECORDING_MAX_DURATION: int = 120
    RECORDING_MAX_CONCURRENT: int = 2
    RECORDING_CRF: int = 23
    RECORDING_RESOLUTION: str = "1280x720"
    RECORDING_FPS: int = 30

    # Paths
    MODELS_DIR: Path = Path("./models")
    RECORDINGS_DIR: Path = Path("./recordings")
    THUMBNAILS_DIR: Path = Path("./thumbnails")
    DATA_DIR: Path = Path("./data")

    # Admin
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "changeme"
    ADMIN_EMAIL: str = "admin@localhost"


settings = Settings()
