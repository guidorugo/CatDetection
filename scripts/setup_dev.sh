#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Cat Detection Project - Dev Setup ==="

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    # Generate random secret key
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i "s/change-me-to-a-random-secret-key/$SECRET_KEY/" .env
fi

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Create directories
mkdir -p models/{detection,identification} recordings thumbnails data/processed

# Create admin user
echo "Creating admin user..."
python3 scripts/create_admin.py || echo "Admin user may already exist"

echo ""
echo "=== Setup complete ==="
echo "Activate venv:  source venv/bin/activate"
echo "Run server:     uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
