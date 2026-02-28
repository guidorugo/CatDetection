#!/usr/bin/env bash
# training_client.sh — Run on the JETSON to trigger remote training on the GPU server.
#
# Workflow:
#   1. rsync raw training data to server
#   2. POST /prepare-and-train (server runs YOLO cropping + training)
#   3. Poll /status until complete/error
#   4. Download trained model + registry from server
#   5. POST /api/v1/training/reload-model on local Jetson app
#
# Prerequisites:
#   - SSH key auth to server (ssh-copy-id user@server)
#   - Server is running training_server.py
#   - Jetson CatDetect app is running (for model reload at the end)
#
# Usage:
#   ./scripts/training_client.sh [--epochs 50]
#
# Configuration is read from .env (same file the app uses). Relevant vars:
#   TRAINING_SERVER_SSH   — SSH user@host for rsync and API (e.g. user@192.168.1.200)
#   TRAINING_SERVER_PORT  — Server API port (default: 8001)
#   TRAINING_API_KEY      — Shared secret (same value on server and client)
#   TRAINING_SERVER_DIR   — Project dir on server (default: ~/cat-detection-project)
#
# The script auto-obtains a JWT from the local app using ADMIN_USERNAME/ADMIN_PASSWORD
# from .env, so no separate token management is needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Load .env ---
if [ -f "${PROJECT_DIR}/.env" ]; then
    # Export vars from .env, skipping comments and blank lines
    set -a
    # shellcheck disable=SC1091
    source <(grep -v '^\s*#' "${PROJECT_DIR}/.env" | grep -v '^\s*$')
    set +a
fi

# --- Parse arguments ---
EPOCHS=50
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Configuration ---
TRAINING_SERVER_SSH="${TRAINING_SERVER_SSH:?Set TRAINING_SERVER_SSH in .env (e.g., user@192.168.1.200)}"
TRAINING_SERVER_PORT="${TRAINING_SERVER_PORT:-8001}"
TRAINING_API_KEY="${TRAINING_API_KEY:?Set TRAINING_API_KEY in .env}"
TRAINING_SERVER_DIR="${TRAINING_SERVER_DIR:-~/cat-detection-project}"
CATDETECT_URL="${CATDETECT_URL:-http://localhost:8000}"

# Derive hostname from SSH target (strip user@ prefix)
TRAINING_SERVER_HOST="${TRAINING_SERVER_SSH#*@}"
SERVER_API="http://${TRAINING_SERVER_HOST}:${TRAINING_SERVER_PORT}"

echo "=== CatDetect: Remote Training ==="
echo "Server: ${TRAINING_SERVER_SSH} (API: ${SERVER_API})"
echo "Epochs: ${EPOCHS}"
echo ""

# --- Step 1: rsync raw data to server ---
echo "[1/5] Syncing raw training data to server..."
rsync -avz --progress \
    --exclude='processed/' \
    --exclude='*.zip' \
    "${PROJECT_DIR}/data/" \
    "${TRAINING_SERVER_SSH}:${TRAINING_SERVER_DIR}/data/"
echo "Data sync complete."
echo ""

# --- Step 2: Trigger prepare-and-train ---
echo "[2/5] Triggering training pipeline on server..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    "${SERVER_API}/prepare-and-train?epochs=${EPOCHS}" \
    -H "X-API-Key: ${TRAINING_API_KEY}")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "Training pipeline started."
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
elif [ "$HTTP_CODE" = "409" ]; then
    echo "Training already in progress on server. Joining poll loop..."
else
    echo "ERROR: Failed to start training (HTTP ${HTTP_CODE})"
    echo "$BODY"
    exit 1
fi
echo ""

# --- Step 3: Poll status ---
echo "[3/5] Polling training status..."
POLL_INTERVAL=30

while true; do
    RESPONSE=$(curl -s -w "\n%{http_code}" \
        "${SERVER_API}/status" \
        -H "X-API-Key: ${TRAINING_API_KEY}")

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | head -n -1)

    if [ "$HTTP_CODE" != "200" ]; then
        echo "WARNING: Status check failed (HTTP ${HTTP_CODE}), retrying..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    STATUS=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")
    PHASE=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('phase') or '')" 2>/dev/null || echo "")
    PROGRESS=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('progress') or '')" 2>/dev/null || echo "")

    TIMESTAMP=$(date '+%H:%M:%S')

    case "$STATUS" in
        complete)
            MODEL_VERSION=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_version') or 'unknown')" 2>/dev/null)
            echo "[${TIMESTAMP}] Training complete! Model version: ${MODEL_VERSION}"
            break
            ;;
        error)
            ERROR=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error') or 'unknown')" 2>/dev/null)
            echo "[${TIMESTAMP}] ERROR: Training failed"
            echo "  ${ERROR}"
            exit 1
            ;;
        preparing)
            echo "[${TIMESTAMP}] Preparing data (YOLO cropping)..."
            ;;
        training)
            echo "[${TIMESTAMP}] Training... ${PROGRESS}"
            ;;
        *)
            echo "[${TIMESTAMP}] Status: ${STATUS}"
            ;;
    esac

    sleep "$POLL_INTERVAL"
done
echo ""

# --- Step 4: Download model + registry ---
echo "[4/5] Downloading trained model..."
mkdir -p "${PROJECT_DIR}/models/identification"

# Download model .pth file
curl -s -f -o /tmp/catdetect_model_latest.pth \
    -H "X-API-Key: ${TRAINING_API_KEY}" \
    "${SERVER_API}/model/latest"

# Download registry
curl -s -f -o /tmp/catdetect_registry.json \
    -H "X-API-Key: ${TRAINING_API_KEY}" \
    "${SERVER_API}/model/registry"

# Fix registry paths to be relative (server may have absolute paths)
python3 -c "
import json, re, sys

reg = json.load(open('/tmp/catdetect_registry.json'))
for ver, info in reg.get('models', {}).items():
    path = info.get('path', '')
    # Extract relative path from models/ onward
    match = re.search(r'(models/identification/.+)', path)
    if match:
        info['path'] = match.group(1)
json.dump(reg, open('/tmp/catdetect_registry.json', 'w'), indent=2)
print('Registry paths fixed.')
"

# Get active model filename from registry
MODEL_FILENAME=$(python3 -c "
import json, os
reg = json.load(open('/tmp/catdetect_registry.json'))
active = reg.get('active')
if active and active in reg.get('models', {}):
    print(os.path.basename(reg['models'][active]['path']))
else:
    print('')
")

if [ -z "$MODEL_FILENAME" ]; then
    echo "ERROR: Could not determine model filename from registry"
    exit 1
fi

# Move files into place
mv /tmp/catdetect_model_latest.pth "${PROJECT_DIR}/models/identification/${MODEL_FILENAME}"
mv /tmp/catdetect_registry.json "${PROJECT_DIR}/models/identification/registry.json"
echo "Model saved: models/identification/${MODEL_FILENAME}"
echo ""

# --- Step 5: Reload model on Jetson app ---
echo "[5/5] Reloading model on local CatDetect app..."

# Auto-login using admin credentials from .env
ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:?Set ADMIN_PASSWORD in .env}"

LOGIN_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    "${CATDETECT_URL}/api/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\": \"${ADMIN_USERNAME}\", \"password\": \"${ADMIN_PASSWORD}\"}")

LOGIN_CODE=$(echo "$LOGIN_RESPONSE" | tail -1)
LOGIN_BODY=$(echo "$LOGIN_RESPONSE" | head -n -1)

if [ "$LOGIN_CODE" != "200" ]; then
    echo "ERROR: Login failed (HTTP ${LOGIN_CODE})"
    echo "$LOGIN_BODY"
    exit 1
fi

CATDETECT_TOKEN=$(echo "$LOGIN_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    "${CATDETECT_URL}/api/v1/training/reload-model" \
    -H "Authorization: Bearer ${CATDETECT_TOKEN}" \
    -H "Content-Type: application/json")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "Model reloaded successfully!"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "ERROR: Reload failed (HTTP ${HTTP_CODE})"
    echo "$BODY"
    exit 1
fi

echo ""
echo "=== Done! New model is live. ==="
