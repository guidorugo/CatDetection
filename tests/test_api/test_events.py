import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.camera import Camera
from app.models.cat import Cat
from app.models.event import DetectionEvent


@pytest.fixture
async def camera_and_cat(db_session: AsyncSession):
    camera = Camera(name="TestCam", source_url="rtsp://test", source_type="rtsp")
    cat = Cat(name="Mimir", description="Grey tabby")
    db_session.add_all([camera, cat])
    await db_session.commit()
    await db_session.refresh(camera)
    await db_session.refresh(cat)
    return camera, cat


@pytest.fixture
async def sample_events(db_session: AsyncSession, camera_and_cat):
    camera, cat = camera_and_cat
    events = []
    for i in range(5):
        event = DetectionEvent(
            camera_id=camera.id,
            cat_id=cat.id,
            detection_confidence=0.95 - i * 0.05,
            identification_confidence=0.85,
            bbox_x=100,
            bbox_y=100,
            bbox_w=200,
            bbox_h=200,
        )
        db_session.add(event)
        events.append(event)
    # One event with unknown cat
    unknown_event = DetectionEvent(
        camera_id=camera.id,
        cat_id=None,
        detection_confidence=0.7,
        bbox_x=50,
        bbox_y=50,
        bbox_w=100,
        bbox_h=100,
    )
    db_session.add(unknown_event)
    events.append(unknown_event)
    await db_session.commit()
    for e in events:
        await db_session.refresh(e)
    return events


@pytest.mark.asyncio
async def test_list_events(client: AsyncClient, auth_headers, sample_events):
    resp = await client.get("/api/v1/events/", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 6
    assert len(data["items"]) == 6


@pytest.mark.asyncio
async def test_list_events_filter_by_camera(
    client: AsyncClient, auth_headers, sample_events, camera_and_cat
):
    camera, _ = camera_and_cat
    resp = await client.get(
        f"/api/v1/events/?camera_id={camera.id}", headers=auth_headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 6
    assert len(data["items"]) == 6
    assert all(e["camera_id"] == camera.id for e in data["items"])


@pytest.mark.asyncio
async def test_list_events_filter_by_cat(
    client: AsyncClient, auth_headers, sample_events, camera_and_cat
):
    _, cat = camera_and_cat
    resp = await client.get(
        f"/api/v1/events/?cat_id={cat.id}", headers=auth_headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["items"]) == 5  # excludes the unknown cat event


@pytest.mark.asyncio
async def test_list_events_pagination(client: AsyncClient, auth_headers, sample_events):
    resp = await client.get("/api/v1/events/?limit=3&offset=0", headers=auth_headers)
    assert resp.status_code == 200
    page1 = resp.json()
    assert page1["total"] == 6
    assert len(page1["items"]) == 3

    resp = await client.get("/api/v1/events/?limit=3&offset=3", headers=auth_headers)
    page2 = resp.json()
    assert len(page2["items"]) == 3

    # Different events on each page
    ids1 = {e["id"] for e in page1["items"]}
    ids2 = {e["id"] for e in page2["items"]}
    assert ids1.isdisjoint(ids2)


@pytest.mark.asyncio
async def test_get_single_event(client: AsyncClient, auth_headers, sample_events):
    event_id = sample_events[0].id
    resp = await client.get(f"/api/v1/events/{event_id}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == event_id
    assert data["cat_name"] == "Mimir"
    assert data["camera_name"] == "TestCam"


@pytest.mark.asyncio
async def test_get_event_not_found(client: AsyncClient, auth_headers):
    resp = await client.get("/api/v1/events/9999", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_event_stats(client: AsyncClient, auth_headers, sample_events):
    resp = await client.get("/api/v1/events/stats", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_events"] == 6
    assert data["events_today"] == 6
    assert "Mimir" in data["cats_detected"]
    assert data["cats_detected"]["Mimir"] == 5
