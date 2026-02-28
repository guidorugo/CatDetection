#!/usr/bin/env bash
set -e

echo "=== CatDetect Entrypoint ==="

# Set up SSH key with correct permissions (bind mounts lose file modes)
SSH_KEY_CONFIGURED=false
mkdir -p /root/.ssh
for keyfile in id_rsa id_ed25519 id_ecdsa; do
    if [ -f "/root/.ssh-mount/${keyfile}" ]; then
        cp "/root/.ssh-mount/${keyfile}" "/root/.ssh/${keyfile}"
        chmod 600 "/root/.ssh/${keyfile}"
        SSH_KEY_CONFIGURED=true
    fi
done
if [ "$SSH_KEY_CONFIGURED" = true ]; then
    [ -f /root/.ssh-mount/known_hosts ] && cp /root/.ssh-mount/known_hosts /root/.ssh/known_hosts
    echo -e "Host *\n  StrictHostKeyChecking accept-new" > /root/.ssh/config
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
