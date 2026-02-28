# CatDetect — Cat Detection & Identification Service

A web-based service that ingests live video from IoT cameras (ESP32/RTSP), detects cats using YOLO, identifies which specific cat is present using a trained Re-ID model, records event clips, and provides a real-time dashboard.

## Target Hardware

- **Primary**: Nvidia Jetson Orin Nano Super (8GB, JetPack 6.2)
- **Secondary**: Server with NVIDIA GPU

## Features

- Real-time cat detection (YOLOv8s, ~5ms/frame on Jetson)
- Cat re-identification using ResNet50 embeddings
- Multi-camera support (RTSP, MJPEG, HTTP)
- Event-triggered 720p@30fps video recording
- Web dashboard with live camera feeds
- Camera health monitoring with auto-reconnection
- JWT authentication
- Training pipeline for custom cat identification

## Quick Start (Local Development)

```bash
# Setup
./scripts/setup_dev.sh

# Activate venv
source venv/bin/activate

# Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Default login: `admin` / `changeme` (change via `.env`)

## Docker (Jetson)

Base image: `dustynv/l4t-pytorch:r36.4.0` (includes CUDA 12.6, TensorRT, PyTorch 2.5).

```bash
cd docker
docker compose up --build
```

The entrypoint automatically runs database migrations, creates an admin user, and downloads the YOLOv8s model on first start. The container uses `network_mode: host` and requires `nvidia` runtime.

Volumes mount `data/`, `models/`, `recordings/`, and `thumbnails/` from the host. The database lives in a named Docker volume (`catdetect-db`).

## Data Preparation & Training

```bash
# 1. Crop cats from raw images
python scripts/prepare_data.py --data-dir data --output-dir data/processed

# 2. Train identification model
python scripts/train_identifier.py --epochs 50

# 3. Export to ONNX (optional TensorRT)
python scripts/export_model.py models/identification/cat_reid_*.pth
```

## Two-Machine Training (Server + Jetson)

If you have a separate server with a more powerful GPU (e.g., RTX 2080Ti), you can train on the server while the Jetson keeps running detection. PyTorch `.pth` models are portable between x86_64 and ARM64.

### Server Setup (one-time)

```bash
# Clone the repo on the server
git clone <repo-url> && cd cat-detection-project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set up SSH key auth to Jetson
ssh-copy-id user@jetson-ip
```

### Workflow

```bash
# On the Jetson: prepare training data (YOLO crops raw images)
python scripts/prepare_data.py --data-dir data --output-dir data/processed

# On the server: sync data, train, deploy
export JETSON_HOST=user@jetson-ip
export AUTH_TOKEN=<your-jwt-token>
./scripts/sync_and_train.sh
```

The script:
1. Pulls `data/processed/` from Jetson (cropped 256x256 JPEGs, not raw images)
2. Trains the model on the server GPU
3. Pushes the model + registry back to Jetson
4. Calls `POST /api/v1/training/reload-model` to hot-swap the model

Detection continues running on the Jetson throughout. The reload pauses the pipeline for ~2 seconds while loading the new `.pth` file.

## Architecture

```
Camera (RTSP/MJPEG) → FrameGrabber (threaded) → DetectionPipeline (async)
  ├── YOLO detect (cat class 15)
  ├── ResNet50 Re-ID embedding → cosine similarity matching
  ├── Event persistence + thumbnail
  ├── FFmpeg recording (libx264, pre-roll + post-roll)
  └── WebSocket broadcast → Dashboard
```

## API

- `POST /api/v1/auth/login` — JWT authentication
- `GET/POST /api/v1/cameras` — Camera CRUD
- `GET/POST /api/v1/cats` — Cat profile management
- `GET /api/v1/events` — Detection events (filterable, paginated)
- `GET /api/v1/recordings` — Recording management
- `POST /api/v1/training/start` — Start model training
- `POST /api/v1/training/reload-model` — Hot-reload identifier model from disk
- `WS /api/v1/ws/live/{camera_id}` — Live camera stream
- `WS /api/v1/ws/events` — Real-time event notifications

## Tests

```bash
pytest tests/
```
