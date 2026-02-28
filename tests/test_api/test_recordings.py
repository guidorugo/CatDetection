import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recording import Recording
from datetime import datetime, timezone


@pytest.fixture
async def sample_recording(db_session: AsyncSession):
    recording = Recording(
        camera_id=1,
        file_path="/tmp/nonexistent_test_recording.mp4",
        start_time=datetime.now(timezone.utc),
        status="completed",
        duration=15.5,
        file_size=1024000,
        resolution="1280x720",
        fps=30,
        codec="h264",
    )
    db_session.add(recording)
    await db_session.commit()
    await db_session.refresh(recording)
    return recording


@pytest.mark.asyncio
async def test_list_recordings(client: AsyncClient, auth_headers, sample_recording):
    resp = await client.get("/api/v1/recordings/", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["status"] == "completed"
    assert data[0]["codec"] == "h264"


@pytest.mark.asyncio
async def test_list_recordings_filter_by_camera(
    client: AsyncClient, auth_headers, sample_recording
):
    resp = await client.get("/api/v1/recordings/?camera_id=1", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1

    resp = await client.get("/api/v1/recordings/?camera_id=999", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 0


@pytest.mark.asyncio
async def test_get_recording(client: AsyncClient, auth_headers, sample_recording):
    resp = await client.get(
        f"/api/v1/recordings/{sample_recording.id}", headers=auth_headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == sample_recording.id
    assert data["duration"] == 15.5


@pytest.mark.asyncio
async def test_get_recording_not_found(client: AsyncClient, auth_headers):
    resp = await client.get("/api/v1/recordings/9999", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_recording_file_not_found(
    client: AsyncClient, auth_headers, sample_recording
):
    resp = await client.get(
        f"/api/v1/recordings/{sample_recording.id}/stream", headers=auth_headers
    )
    assert resp.status_code == 404  # file doesn't exist on disk


@pytest.mark.asyncio
async def test_delete_recording(client: AsyncClient, auth_headers, sample_recording):
    resp = await client.delete(
        f"/api/v1/recordings/{sample_recording.id}", headers=auth_headers
    )
    assert resp.status_code == 204

    resp = await client.get(
        f"/api/v1/recordings/{sample_recording.id}", headers=auth_headers
    )
    assert resp.status_code == 404
