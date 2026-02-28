from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _get_notification_service(websocket: WebSocket):
    """Get NotificationService from app state, or None if not initialized."""
    return getattr(websocket.app.state, "notification_service", None)


@router.websocket("/live/{camera_id}")
async def ws_live_feed(websocket: WebSocket, camera_id: int):
    await websocket.accept()
    ns = _get_notification_service(websocket)
    if ns:
        ns.register_live(camera_id, websocket)
    logger.info("Live feed WebSocket connected for camera %d", camera_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if ns:
            ns.unregister_live(camera_id, websocket)
        logger.info("Live feed WebSocket disconnected for camera %d", camera_id)


@router.websocket("/events")
async def ws_events(websocket: WebSocket):
    await websocket.accept()
    ns = _get_notification_service(websocket)
    if ns:
        ns.register_events(websocket)
    logger.info("Events WebSocket connected")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if ns:
            ns.unregister_events(websocket)
        logger.info("Events WebSocket disconnected")


@router.websocket("/status")
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    ns = _get_notification_service(websocket)
    if ns:
        ns.register_status(websocket)
    logger.info("Status WebSocket connected")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if ns:
            ns.unregister_status(websocket)
        logger.info("Status WebSocket disconnected")
