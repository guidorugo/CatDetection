#!/usr/bin/env bash
set -e

echo "=== CatDetect Entrypoint ==="

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Create admin user if needed
echo "Ensuring admin user exists..."
python3 scripts/create_admin.py || true

# Download YOLO model if needed
if [ ! -f "yolov8s.pt" ]; then
    echo "Downloading YOLOv8s model..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" || echo "YOLO download will happen on first use"
fi

echo "Starting application..."
exec "$@"
