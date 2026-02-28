import json

from fastapi import WebSocket

from app.core.logging import get_logger

logger = get_logger(__name__)


class NotificationService:
    """WebSocket broadcast hub for live feeds, events, and status updates."""

    def __init__(self):
        # camera_id -> list of websockets
        self._live_connections: dict[int, list[WebSocket]] = {}
        self._event_connections: list[WebSocket] = []
        self._status_connections: list[WebSocket] = []

    def register_live(self, camera_id: int, ws: WebSocket):
        self._live_connections.setdefault(camera_id, []).append(ws)

    def unregister_live(self, camera_id: int, ws: WebSocket):
        conns = self._live_connections.get(camera_id, [])
        if ws in conns:
            conns.remove(ws)

    def register_events(self, ws: WebSocket):
        self._event_connections.append(ws)

    def unregister_events(self, ws: WebSocket):
        if ws in self._event_connections:
            self._event_connections.remove(ws)

    def register_status(self, ws: WebSocket):
        self._status_connections.append(ws)

    def unregister_status(self, ws: WebSocket):
        if ws in self._status_connections:
            self._status_connections.remove(ws)

    async def broadcast_frame(self, camera_id: int, jpeg_bytes: bytes):
        """Send JPEG frame to all live viewers of a camera."""
        conns = self._live_connections.get(camera_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_bytes(jpeg_bytes)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.remove(ws)

    async def broadcast_event(self, event_data: dict):
        """Send detection event to all event subscribers."""
        msg = json.dumps(event_data)
        dead = []
        for ws in self._event_connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._event_connections.remove(ws)

    async def broadcast_status(self, status_data: dict):
        """Send status update to all status subscribers."""
        msg = json.dumps(status_data)
        dead = []
        for ws in self._status_connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._status_connections.remove(ws)
