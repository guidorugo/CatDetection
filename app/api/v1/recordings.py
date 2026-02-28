from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.recording import Recording
from app.models.user import User
from app.schemas.recording import RecordingResponse

router = APIRouter()


@router.get("/", response_model=list[RecordingResponse])
async def list_recordings(
    camera_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    query = select(Recording)
    if camera_id is not None:
        query = query.where(Recording.camera_id == camera_id)
    query = query.order_by(Recording.start_time.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{recording_id}", response_model=RecordingResponse)
async def get_recording(
    recording_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Recording).where(Recording.id == recording_id))
    recording = result.scalar_one_or_none()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    return recording


@router.get("/{recording_id}/stream")
async def stream_recording(
    recording_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Recording).where(Recording.id == recording_id))
    recording = result.scalar_one_or_none()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    file_path = Path(recording.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Recording file not found")
    return FileResponse(
        str(file_path), media_type="video/mp4", headers={"Accept-Ranges": "bytes"}
    )


@router.get("/{recording_id}/download")
async def download_recording(
    recording_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Recording).where(Recording.id == recording_id))
    recording = result.scalar_one_or_none()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    file_path = Path(recording.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Recording file not found")
    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        filename=file_path.name,
    )


@router.delete("/{recording_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_recording(
    recording_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Recording).where(Recording.id == recording_id))
    recording = result.scalar_one_or_none()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    file_path = Path(recording.file_path)
    if file_path.exists():
        file_path.unlink()
    await db.delete(recording)
    await db.commit()
