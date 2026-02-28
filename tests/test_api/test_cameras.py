import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_camera(client: AsyncClient, auth_headers):
    resp = await client.post(
        "/api/v1/cameras",
        headers=auth_headers,
        json={
            "name": "Front Door",
            "source_url": "rtsp://192.168.1.100:554/stream",
            "source_type": "rtsp",
            "location": "Front Porch",
            "expected_fps": 30,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Front Door"
    assert data["is_enabled"] is True


@pytest.mark.asyncio
async def test_list_cameras(client: AsyncClient, auth_headers):
    # Create a camera first
    await client.post(
        "/api/v1/cameras",
        headers=auth_headers,
        json={"name": "Test Cam", "source_url": "rtsp://test"},
    )

    resp = await client.get("/api/v1/cameras", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_update_camera(client: AsyncClient, auth_headers):
    # Create
    create_resp = await client.post(
        "/api/v1/cameras",
        headers=auth_headers,
        json={"name": "Update Me", "source_url": "rtsp://test"},
    )
    cam_id = create_resp.json()["id"]

    # Update
    resp = await client.put(
        f"/api/v1/cameras/{cam_id}",
        headers=auth_headers,
        json={"name": "Updated Camera", "location": "Backyard"},
    )
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated Camera"
    assert resp.json()["location"] == "Backyard"


@pytest.mark.asyncio
async def test_delete_camera(client: AsyncClient, auth_headers):
    create_resp = await client.post(
        "/api/v1/cameras",
        headers=auth_headers,
        json={"name": "Delete Me", "source_url": "rtsp://test"},
    )
    cam_id = create_resp.json()["id"]

    resp = await client.delete(f"/api/v1/cameras/{cam_id}", headers=auth_headers)
    assert resp.status_code == 204

    resp = await client.get(f"/api/v1/cameras/{cam_id}", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_enable_disable_camera(client: AsyncClient, auth_headers):
    create_resp = await client.post(
        "/api/v1/cameras",
        headers=auth_headers,
        json={"name": "Toggle Cam", "source_url": "rtsp://test"},
    )
    cam_id = create_resp.json()["id"]

    # Disable
    resp = await client.post(f"/api/v1/cameras/{cam_id}/disable", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["is_enabled"] is False

    # Enable
    resp = await client.post(f"/api/v1/cameras/{cam_id}/enable", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["is_enabled"] is True
