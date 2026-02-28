#!/usr/bin/env bash
# ============================================================================
# DEPRECATED — Use the new client-server training architecture instead:
#
#   Server:  TRAINING_API_KEY=secret python scripts/training_server.py
#   Jetson:  ./scripts/training_client.sh --epochs 50
#
# The new approach runs a persistent daemon on the server and lets the Jetson
# push data, trigger training (including YOLO data prep), poll progress, and
# pull the model back — all automated. See README.md for details.
# ============================================================================
#
# sync_and_train.sh — (Legacy) Run on the SERVER to train a cat re-ID model.
#
# Workflow:
#   1. rsync processed training data from Jetson → server
#   2. Train the model locally (uses server GPU)
#   3. rsync the trained model + registry back to Jetson
#   4. Call the reload endpoint on the Jetson to hot-swap the model
#
# Prerequisites:
#   - SSH key auth to Jetson (ssh-copy-id user@jetson)
#   - Server has the repo cloned with venv set up (pip install -r requirements.txt)
#   - Jetson is running the CatDetect app with a valid auth token
#
# Usage:
#   ./scripts/sync_and_train.sh
#
# Environment variables (set these or edit defaults below):
#   JETSON_HOST     — Jetson SSH host (e.g., user@192.168.1.100)
#   JETSON_DIR      — Project directory on Jetson (default: /ssd/projects/cat-detection-project)
#   JETSON_URL      — Jetson app base URL (default: http://<jetson_ip>:8000)
#   AUTH_TOKEN      — JWT token for the reload endpoint
#   EPOCHS          — Training epochs (default: 50)

set -euo pipefail

# --- Configuration ---
JETSON_HOST="${JETSON_HOST:?Set JETSON_HOST (e.g., user@192.168.1.100)}"
JETSON_DIR="${JETSON_DIR:-/ssd/projects/cat-detection-project}"
JETSON_IP="${JETSON_HOST#*@}"
JETSON_URL="${JETSON_URL:-http://${JETSON_IP}:8000}"
AUTH_TOKEN="${AUTH_TOKEN:?Set AUTH_TOKEN (JWT token for /api/v1/training/reload-model)}"
EPOCHS="${EPOCHS:-50}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== CatDetect: Sync & Train ==="
echo "Jetson: ${JETSON_HOST}:${JETSON_DIR}"
echo "Server project: ${PROJECT_DIR}"
echo "Epochs: ${EPOCHS}"
echo ""

# --- Step 1: Pull training data from Jetson ---
echo "[1/4] Syncing training data from Jetson..."
mkdir -p "${PROJECT_DIR}/data/processed"
rsync -avz --progress \
    "${JETSON_HOST}:${JETSON_DIR}/data/processed/" \
    "${PROJECT_DIR}/data/processed/"
echo "Data sync complete."
echo ""

# --- Step 2: Train the model ---
echo "[2/4] Training model (${EPOCHS} epochs)..."
cd "${PROJECT_DIR}"
source venv/bin/activate
python scripts/train_identifier.py \
    --data-dir data/processed \
    --epochs "${EPOCHS}"
echo "Training complete."
echo ""

# --- Step 3: Push model + registry back to Jetson ---
echo "[3/4] Syncing model to Jetson..."
rsync -avz --progress \
    "${PROJECT_DIR}/models/identification/" \
    "${JETSON_HOST}:${JETSON_DIR}/models/identification/"
echo "Model sync complete."
echo ""

# --- Step 4: Trigger model reload on Jetson ---
echo "[4/4] Reloading model on Jetson..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    "${JETSON_URL}/api/v1/training/reload-model" \
    -H "Authorization: Bearer ${AUTH_TOKEN}" \
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
echo "=== Done! New model is live on Jetson. ==="
