import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.training_job import TrainingJob


@pytest.mark.asyncio
async def test_start_training(client: AsyncClient, auth_headers):
    resp = await client.post(
        "/api/v1/training/start",
        headers=auth_headers,
        json={"model_type": "cat_reid", "epochs": 10, "learning_rate": 0.001},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "pending"
    assert data["model_type"] == "cat_reid"
    assert data["epochs_total"] == 10


@pytest.mark.asyncio
async def test_start_training_duplicate(client: AsyncClient, auth_headers):
    # Start first job
    await client.post(
        "/api/v1/training/start",
        headers=auth_headers,
        json={"model_type": "cat_reid", "epochs": 5},
    )
    # Try to start another — should fail
    resp = await client.post(
        "/api/v1/training/start",
        headers=auth_headers,
        json={"model_type": "cat_reid", "epochs": 5},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_training_jobs(client: AsyncClient, auth_headers, db_session):
    job = TrainingJob(
        model_type="cat_reid",
        status="completed",
        epochs_total=50,
        epochs_completed=50,
        best_metric=0.95,
        model_version="20260101_120000",
    )
    db_session.add(job)
    await db_session.commit()

    resp = await client.get("/api/v1/training/jobs", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_get_training_job(client: AsyncClient, auth_headers, db_session):
    job = TrainingJob(
        model_type="cat_reid",
        status="running",
        epochs_total=50,
        epochs_completed=25,
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)

    resp = await client.get(f"/api/v1/training/jobs/{job.id}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["epochs_completed"] == 25


@pytest.mark.asyncio
async def test_get_training_job_not_found(client: AsyncClient, auth_headers):
    resp = await client.get("/api/v1/training/jobs/9999", headers=auth_headers)
    assert resp.status_code == 404
