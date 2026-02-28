import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register(client: AsyncClient):
    resp = await client.post(
        "/api/v1/auth/register",
        json={"username": "newuser", "email": "new@test.com", "password": "pass123"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "newuser"
    assert data["email"] == "new@test.com"


@pytest.mark.asyncio
async def test_register_duplicate(client: AsyncClient, admin_user):
    resp = await client.post(
        "/api/v1/auth/register",
        json={"username": "testadmin", "email": "other@test.com", "password": "pass"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_login(client: AsyncClient, admin_user):
    resp = await client.post(
        "/api/v1/auth/login",
        json={"username": "testadmin", "password": "testpass"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid(client: AsyncClient, admin_user):
    resp = await client.post(
        "/api/v1/auth/login",
        json={"username": "testadmin", "password": "wrong"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me(client: AsyncClient, auth_headers):
    resp = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "testadmin"
    assert data["is_admin"] is True


@pytest.mark.asyncio
async def test_refresh_token(client: AsyncClient, admin_user):
    # Login first
    login_resp = await client.post(
        "/api/v1/auth/login",
        json={"username": "testadmin", "password": "testpass"},
    )
    refresh_token = login_resp.json()["refresh_token"]

    # Refresh
    resp = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_unauthorized(client: AsyncClient):
    resp = await client.get("/api/v1/auth/me")
    assert resp.status_code == 401
