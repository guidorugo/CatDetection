import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_root_redirects_to_dashboard(client: AsyncClient):
    resp = await client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/dashboard"


@pytest.mark.asyncio
async def test_login_page_renders(client: AsyncClient):
    resp = await client.get("/login")
    assert resp.status_code == 200
    assert "CatDetect" in resp.text
    assert "loginForm" in resp.text


@pytest.mark.asyncio
async def test_dashboard_page_renders(client: AsyncClient):
    resp = await client.get("/dashboard")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


@pytest.mark.asyncio
async def test_cameras_page_renders(client: AsyncClient):
    resp = await client.get("/cameras")
    assert resp.status_code == 200
    assert "Camera Management" in resp.text


@pytest.mark.asyncio
async def test_cats_page_renders(client: AsyncClient):
    resp = await client.get("/cats")
    assert resp.status_code == 200
    assert "Cat Profiles" in resp.text


@pytest.mark.asyncio
async def test_events_page_renders(client: AsyncClient):
    resp = await client.get("/events")
    assert resp.status_code == 200
    assert "Detection Events" in resp.text


@pytest.mark.asyncio
async def test_recordings_page_renders(client: AsyncClient):
    resp = await client.get("/recordings")
    assert resp.status_code == 200
    assert "Recordings" in resp.text


@pytest.mark.asyncio
async def test_training_page_renders(client: AsyncClient):
    resp = await client.get("/training")
    assert resp.status_code == 200
    assert "Model Training" in resp.text
