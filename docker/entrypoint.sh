#!/usr/bin/env bash
set -e

echo "=== CatDetect Entrypoint ==="

# Set up SSH key with correct permissions (bind mounts lose file modes)
if [ -f /root/.ssh-mount/id_ed25519 ]; then
    mkdir -p /root/.ssh
    cp /root/.ssh-mount/id_ed25519 /root/.ssh/id_ed25519
    chmod 600 /root/.ssh/id_ed25519
    [ -f /root/.ssh-mount/known_hosts ] && cp /root/.ssh-mount/known_hosts /root/.ssh/known_hosts
    echo "SSH key configured"
fi

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
