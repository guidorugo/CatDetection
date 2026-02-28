import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.notification_service import NotificationService


@pytest.mark.asyncio
async def test_register_and_unregister_events():
    ns = NotificationService()
    ws = MagicMock()
    ns.register_events(ws)
    assert ws in ns._event_connections
    ns.unregister_events(ws)
    assert ws not in ns._event_connections


@pytest.mark.asyncio
async def test_register_and_unregister_status():
    ns = NotificationService()
    ws = MagicMock()
    ns.register_status(ws)
    assert ws in ns._status_connections
    ns.unregister_status(ws)
    assert ws not in ns._status_connections


@pytest.mark.asyncio
async def test_register_and_unregister_live():
    ns = NotificationService()
    ws = MagicMock()
    ns.register_live(1, ws)
    assert ws in ns._live_connections[1]
    ns.unregister_live(1, ws)
    assert ws not in ns._live_connections[1]


@pytest.mark.asyncio
async def test_broadcast_event():
    ns = NotificationService()
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    ns.register_events(ws1)
    ns.register_events(ws2)

    await ns.broadcast_event({"type": "detection", "cat_name": "Mimir"})

    ws1.send_text.assert_called_once()
    ws2.send_text.assert_called_once()
    # Verify JSON content
    import json
    sent = json.loads(ws1.send_text.call_args[0][0])
    assert sent["cat_name"] == "Mimir"


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections():
    ns = NotificationService()
    good_ws = AsyncMock()
    bad_ws = AsyncMock()
    bad_ws.send_text.side_effect = Exception("Connection closed")

    ns.register_events(good_ws)
    ns.register_events(bad_ws)

    await ns.broadcast_event({"type": "test"})

    # bad_ws should have been removed
    assert bad_ws not in ns._event_connections
    assert good_ws in ns._event_connections


@pytest.mark.asyncio
async def test_broadcast_frame():
    ns = NotificationService()
    ws = AsyncMock()
    ns.register_live(1, ws)

    await ns.broadcast_frame(1, b"fake-jpeg-data")
    ws.send_bytes.assert_called_once_with(b"fake-jpeg-data")
