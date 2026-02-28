#!/usr/bin/env python3
"""Training server daemon — runs on the GPU server to handle remote training requests.

Provides a REST API for the Jetson client to trigger data preparation + training,
poll progress, and download the resulting model.

Usage:
    TRAINING_API_KEY=your-secret python scripts/training_server.py

Environment variables:
    TRAINING_API_KEY  — Required. Shared secret for X-API-Key auth.
    TRAINING_PORT     — Server port (default: 8001).
    TRAINING_HOST     — Bind address (default: 0.0.0.0).
"""
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training_server")

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models" / "identification"
REGISTRY_FILE = MODELS_DIR / "registry.json"

API_KEY = os.environ.get("TRAINING_API_KEY", "")

app = FastAPI(title="CatDetect Training Server")

# --- Training state (single-job, in-memory) ---

state = {
    "status": "idle",       # idle | preparing | training | complete | error | cancelled
    "phase": None,          # "prepare_data" | "train" | None
    "progress": None,       # e.g. "12/50 epochs"
    "error": None,
    "model_version": None,
    "started_at": None,
    "completed_at": None,
    "_proc": None,          # subprocess reference for cancellation
    "_cancel": False,       # cancel flag
}
state_lock = threading.Lock()


def reset_state():
    state.update({
        "status": "idle",
        "phase": None,
        "progress": None,
        "error": None,
        "model_version": None,
        "started_at": None,
        "completed_at": None,
        "_proc": None,
        "_cancel": False,
    })


# --- Auth ---

def require_api_key(x_api_key: str = Header(None)):
    if not API_KEY:
        raise HTTPException(500, "TRAINING_API_KEY not configured on server")
    if x_api_key != API_KEY:
        raise HTTPException(403, "Invalid or missing API key")


# --- Background training ---

def _run_pipeline(epochs: int):
    """Run prepare_data.py then train_identifier.py sequentially."""
    try:
        venv_python = sys.executable
        logger.info("Pipeline started (epochs=%d)", epochs)

        # Phase 1: Prepare data
        with state_lock:
            state["status"] = "preparing"
            state["phase"] = "prepare_data"
            state["progress"] = None

        cmd_prepare = [
            venv_python, str(PROJECT_DIR / "scripts" / "prepare_data.py"),
            "--data-dir", str(PROJECT_DIR / "data"),
            "--output-dir", str(PROJECT_DIR / "data" / "processed"),
        ]
        prep_proc = subprocess.Popen(
            cmd_prepare, cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        with state_lock:
            state["_proc"] = prep_proc

        stdout, stderr = prep_proc.communicate(timeout=1800)

        with state_lock:
            state["_proc"] = None
            if state["_cancel"]:
                state["status"] = "cancelled"
                logger.info("Pipeline cancelled during data preparation")
                return

        if prep_proc.returncode != 0:
            logger.error("prepare_data.py failed (exit code %d)", prep_proc.returncode)
            with state_lock:
                state["status"] = "error"
                state["error"] = f"prepare_data.py failed:\n{stderr[-500:]}"
            return

        logger.info("Data preparation complete")

        # Phase 2: Train
        with state_lock:
            if state["_cancel"]:
                state["status"] = "cancelled"
                logger.info("Pipeline cancelled before training")
                return
            state["status"] = "training"
            state["phase"] = "train"
            state["progress"] = "0/{} epochs".format(epochs)

        cmd_train = [
            venv_python, str(PROJECT_DIR / "scripts" / "train_identifier.py"),
            "--data-dir", str(PROJECT_DIR / "data" / "processed"),
            "--epochs", str(epochs),
        ]
        proc = subprocess.Popen(
            cmd_train, cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        with state_lock:
            state["_proc"] = proc

        epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            m = epoch_re.search(line)
            if m:
                with state_lock:
                    state["progress"] = f"{m.group(1)}/{m.group(2)} epochs"
            with state_lock:
                if state["_cancel"]:
                    break

        proc.wait(timeout=7200)

        with state_lock:
            state["_proc"] = None
            if state["_cancel"]:
                state["status"] = "cancelled"
                logger.info("Pipeline cancelled during training")
                return

        if proc.returncode != 0:
            logger.error("train_identifier.py failed (exit code %d)", proc.returncode)
            with state_lock:
                state["status"] = "error"
                state["error"] = "train_identifier.py failed (exit code {})".format(proc.returncode)
            return

        # Read model version from registry
        model_version = None
        if REGISTRY_FILE.exists():
            try:
                reg = json.loads(REGISTRY_FILE.read_text())
                model_version = reg.get("active")
            except Exception:
                pass

        with state_lock:
            state["status"] = "complete"
            state["phase"] = None
            state["model_version"] = model_version
            state["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info("Pipeline complete — model version: %s", model_version)

    except subprocess.TimeoutExpired:
        with state_lock:
            state["status"] = "error"
            state["error"] = "Pipeline timed out"
    except Exception as e:
        with state_lock:
            state["status"] = "error"
            state["error"] = str(e)


# --- Endpoints ---

@app.get("/health")
async def health(request: Request):
    logger.info("Health check from %s", request.client.host)
    return {"status": "ok"}


@app.post("/prepare-and-train")
async def prepare_and_train(
    request: Request,
    epochs: int = Query(default=50, ge=1, le=500),
    x_api_key: str = Header(None),
):
    require_api_key(x_api_key)
    logger.info("Training requested from %s (epochs=%d)", request.client.host, epochs)

    with state_lock:
        if state["status"] in ("preparing", "training"):
            logger.warning("Rejected: training already in progress")
            raise HTTPException(409, "Training already in progress")
        reset_state()
        state["status"] = "preparing"
        state["started_at"] = datetime.now(timezone.utc).isoformat()

    thread = threading.Thread(target=_run_pipeline, args=(epochs,), daemon=True)
    thread.start()

    return {"message": "Training pipeline started", "epochs": epochs}


@app.get("/status")
async def status(request: Request, x_api_key: str = Header(None)):
    require_api_key(x_api_key)
    with state_lock:
        current = dict(state)
    logger.info("Status poll from %s — %s", request.client.host, current["status"])
    return current


@app.post("/cancel")
async def cancel(request: Request, x_api_key: str = Header(None)):
    require_api_key(x_api_key)
    with state_lock:
        if state["status"] not in ("preparing", "training"):
            raise HTTPException(400, f"No active job to cancel (status: {state['status']})")
        state["_cancel"] = True
        proc = state.get("_proc")
        if proc and proc.poll() is None:
            logger.info("Terminating subprocess (pid=%d)", proc.pid)
            proc.terminate()
    logger.info("Cancel requested from %s", request.client.host)
    return {"message": "Cancel requested"}


@app.get("/model/latest")
async def model_latest(request: Request, x_api_key: str = Header(None)):
    require_api_key(x_api_key)

    if not REGISTRY_FILE.exists():
        raise HTTPException(404, "No model registry found")

    reg = json.loads(REGISTRY_FILE.read_text())
    active = reg.get("active")
    if not active or active not in reg.get("models", {}):
        raise HTTPException(404, "No active model")

    model_path = Path(reg["models"][active]["path"])
    # Handle relative and absolute paths
    if not model_path.is_absolute():
        model_path = PROJECT_DIR / model_path

    if not model_path.exists():
        raise HTTPException(404, f"Model file not found: {model_path}")

    logger.info("Model download from %s — %s", request.client.host, model_path.name)
    return FileResponse(
        str(model_path),
        media_type="application/octet-stream",
        filename=model_path.name,
    )


@app.get("/model/registry")
async def model_registry(request: Request, x_api_key: str = Header(None)):
    require_api_key(x_api_key)

    if not REGISTRY_FILE.exists():
        raise HTTPException(404, "No model registry found")

    logger.info("Registry download from %s", request.client.host)
    return JSONResponse(json.loads(REGISTRY_FILE.read_text()))


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: Set TRAINING_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    port = int(os.environ.get("TRAINING_PORT", "8001"))
    host = os.environ.get("TRAINING_HOST", "0.0.0.0")

    print(f"Starting training server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
