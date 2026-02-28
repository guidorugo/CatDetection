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
- Training pipeline for custom cat identification (local or remote GPU server)
- Training job cancellation and live progress tracking

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

## Remote Training (Server + Jetson)

Train on a server with a more powerful GPU (e.g., RTX 2080Ti) while the Jetson keeps running detection. The server runs a persistent training daemon; the Jetson pushes raw images, triggers training (including YOLO data prep), polls progress, and pulls the model back.

```
Jetson (Client)                          Server (GPU)
+-----------------------+                +---------------------------+
| CatDetect app :8000   |                | training_server.py :8001  |
|                       |   rsync        |                           |
| data/{cat_name}/*.jpg | ============> | data/{cat_name}/*.jpg     |
|                       |                |                           |
| training_client.sh    | -- POST -----> | /prepare-and-train        |
|                       |                |   → prepare_data.py       |
|                       | -- GET ------> | /status (poll progress)   |
|                       |                |   → train_identifier.py   |
|                       | <-- GET ------ | /model/latest (.pth)      |
|                       | <-- GET ------ | /model/registry (.json)   |
|                       |                |                           |
| POST /reload-model    |                +---------------------------+
+-----------------------+
```

### Server Setup (one-time)

```bash
# Clone the repo on the server
git clone <repo-url> && cd cat-detection-project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Start the training daemon:

```bash
TRAINING_API_KEY=your-secret python scripts/training_server.py
```

Optional systemd service (`/etc/systemd/system/catdetect-training.service`):

```ini
[Unit]
Description=CatDetect Training Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/cat-detection-project
Environment=TRAINING_API_KEY=your-secret
ExecStart=/path/to/cat-detection-project/venv/bin/python scripts/training_server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Training from the Web UI

When `TRAINING_SERVER_SSH` and `TRAINING_API_KEY` are configured in `.env`, the training page shows a "Train on remote server" option. Selecting it will:
1. Rsync raw `data/` to the server automatically
2. Trigger training on the remote GPU server
3. Poll progress and show a live progress bar in the UI
4. Download the trained model and hot-reload the identifier

Training jobs can be cancelled from the UI at any time. The progress bar auto-polls every 5s while a job is running.

### Training from the CLI (Jetson)

```bash
# Set up SSH key auth to the server (for rsync)
ssh-copy-id user@server-ip

# Configure .env with remote training settings (TRAINING_SERVER_SSH,
# TRAINING_API_KEY — see .env.example)

# Run training (default 50 epochs)
./scripts/training_client.sh --epochs 50
```

The client script reads all configuration from `.env` (same file the app uses) and auto-logs in using `ADMIN_USERNAME`/`ADMIN_PASSWORD` for the model reload step. It:
1. Rsyncs raw `data/` (excluding `processed/`) to the server
2. Triggers `/prepare-and-train` (server runs YOLO cropping + model training)
3. Polls `/status` every 30s showing progress
4. Downloads the trained `.pth` model and registry
5. Auto-logs in and calls `POST /api/v1/training/reload-model` to hot-swap the model

Detection continues running on the Jetson throughout. PyTorch `.pth` models are portable between x86_64 and ARM64.

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
- `POST /api/v1/training/start` — Start model training (local or remote)
- `POST /api/v1/training/jobs/{id}/cancel` — Cancel a running training job
- `POST /api/v1/training/reload-model` — Hot-reload identifier model from disk
- `WS /api/v1/ws/live/{camera_id}` — Live camera stream
- `WS /api/v1/ws/events` — Real-time event notifications

## Tests

```bash
pytest tests/
```
