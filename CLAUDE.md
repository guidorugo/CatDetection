# Cat Detection Project

## Quick Reference

- **Stack**: FastAPI + SQLAlchemy 2.0 (async) + SQLite + YOLOv8 + ResNet50 + Jinja2
- **Python**: 3.10 (Jetson) / 3.12 (server)
- **Target hardware**: Nvidia Jetson Orin Nano Super (8GB, no NVENC)
- **Dev setup**: `source venv/bin/activate` then `uvicorn app.main:app --reload`
- **Tests**: `pytest tests/`
- **DB migrations**: `alembic revision --autogenerate -m "description"` then `alembic upgrade head`

## Project Structure

- `app/` — Main application (FastAPI)
  - `api/` — HTTP routes and WebSocket endpoints
  - `core/` — Config, database, security, logging
  - `models/` — SQLAlchemy ORM models
  - `schemas/` — Pydantic request/response schemas
  - `services/` — Business logic (stream, detection, recording)
  - `ml/` — ML models (YOLO detector, ResNet identifier, training)
  - `templates/` — Jinja2 HTML templates
  - `static/` — CSS/JS assets
- `scripts/` — CLI tools (setup, training, data prep, admin)
- `docker/` — Docker/compose files
- `tests/` — pytest test suite

## Key Conventions

- Always use virtual environment (`venv/`)
- Database operations are async (aiosqlite)
- ML inference runs via `asyncio.to_thread()` to avoid blocking
- Frame grabbing is threaded (one thread per camera)
- Recording uses FFmpeg subprocess with libx264 (no hardware encoder)
- YOLO class 15 = cat
- Cat Re-ID produces 512-d L2-normalized embeddings
- Stop detection pipeline during training to avoid GPU OOM

## Docker

- **Base image**: `dustynv/l4t-pytorch:r36.4.0` (NOT `nvcr.io/nvidia/l4t-pytorch`)
- Build requires `network: host` in docker-compose build section (Jetson iptables issue)
- Dockerfile uses `--index-url https://pypi.org/simple/` to override the base image's broken Jetson pip indexes (torch is already in the base image, no need for Jetson-specific pip sources)
- `numpy` must be pinned to `<2` — base image PyTorch is compiled against numpy 1.x
- `bcrypt` pinned to `<5.0.0` for passlib compatibility
- ML imports in `app/main.py` are lazy (inside lifespan) to allow API-only mode when torch is unavailable

## Remote Training (Client-Server)

- Server runs `scripts/training_server.py` (standalone FastAPI daemon, no `app.*` imports)
- Jetson runs `scripts/training_client.sh` to push data, trigger training, poll, and pull the model
- Server handles both `prepare_data.py` (YOLO cropping) and `train_identifier.py` as subprocesses
- Auth via `X-API-Key` header (shared `TRAINING_API_KEY` env var)
- PyTorch `.pth` files are portable between x86_64 ↔ ARM64 (`map_location=device`)
- TensorRT `.engine` files are NOT portable (device-specific), but the app falls back to `.pth`
- `POST /api/v1/training/reload-model` hot-swaps the identifier model without restarting the app
- SQLite stays on Jetson only; the server never touches the database
- `scripts/sync_and_train.sh` is deprecated (kept for reference)

## Environment Variables

See `.env.example` for all available settings.
