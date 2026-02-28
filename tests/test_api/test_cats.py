import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_cat(client: AsyncClient, auth_headers):
    resp = await client.post(
        "/api/v1/cats",
        headers=auth_headers,
        json={"name": "Mimir", "description": "Grey tabby"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Mimir"
    assert data["description"] == "Grey tabby"


@pytest.mark.asyncio
async def test_create_duplicate_cat(client: AsyncClient, auth_headers):
    await client.post(
        "/api/v1/cats", headers=auth_headers, json={"name": "DupCat"}
    )
    resp = await client.post(
        "/api/v1/cats", headers=auth_headers, json={"name": "DupCat"}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_cats(client: AsyncClient, auth_headers):
    await client.post(
        "/api/v1/cats", headers=auth_headers, json={"name": "Iorek"}
    )
    resp = await client.get("/api/v1/cats", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


@pytest.mark.asyncio
async def test_update_cat(client: AsyncClient, auth_headers):
    create_resp = await client.post(
        "/api/v1/cats", headers=auth_headers, json={"name": "Rolo"}
    )
    cat_id = create_resp.json()["id"]

    resp = await client.put(
        f"/api/v1/cats/{cat_id}",
        headers=auth_headers,
        json={"description": "Orange fluffball"},
    )
    assert resp.status_code == 200
    assert resp.json()["description"] == "Orange fluffball"


@pytest.mark.asyncio
async def test_delete_cat(client: AsyncClient, auth_headers):
    create_resp = await client.post(
        "/api/v1/cats", headers=auth_headers, json={"name": "DeleteMe"}
    )
    cat_id = create_resp.json()["id"]

    resp = await client.delete(f"/api/v1/cats/{cat_id}", headers=auth_headers)
    assert resp.status_code == 204

    resp = await client.get(f"/api/v1/cats/{cat_id}", headers=auth_headers)
    assert resp.status_code == 404
