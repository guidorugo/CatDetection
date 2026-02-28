from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.api.deps import get_current_user, get_db
from app.models.cat import Cat
from app.models.event import DetectionEvent
from app.models.user import User
from app.schemas.event import EventResponse, EventStats

router = APIRouter()


@router.get("/", response_model=list[EventResponse])
async def list_events(
    camera_id: int | None = None,
    cat_id: int | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(default=50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    query = select(DetectionEvent).options(
        joinedload(DetectionEvent.cat),
        joinedload(DetectionEvent.camera),
    )

    if camera_id is not None:
        query = query.where(DetectionEvent.camera_id == camera_id)
    if cat_id is not None:
        query = query.where(DetectionEvent.cat_id == cat_id)
    if start_time is not None:
        query = query.where(DetectionEvent.timestamp >= start_time)
    if end_time is not None:
        query = query.where(DetectionEvent.timestamp <= end_time)

    query = query.order_by(DetectionEvent.timestamp.desc()).limit(limit).offset(offset)
    result = await db.execute(query)
    events = result.unique().scalars().all()

    # Populate cat_name and camera_name
    responses = []
    for event in events:
        resp = EventResponse.model_validate(event)
        if event.cat:
            resp.cat_name = event.cat.name
        if event.camera:
            resp.camera_name = event.camera.name
        responses.append(resp)
    return responses


@router.get("/stats", response_model=EventStats)
async def get_event_stats(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    # Total events
    result = await db.execute(select(func.count(DetectionEvent.id)))
    total = result.scalar() or 0

    # Events today
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.count(DetectionEvent.id)).where(DetectionEvent.timestamp >= today_start)
    )
    today = result.scalar() or 0

    # Per-cat counts
    result = await db.execute(
        select(Cat.name, func.count(DetectionEvent.id))
        .join(DetectionEvent, DetectionEvent.cat_id == Cat.id)
        .group_by(Cat.name)
    )
    cats_detected = dict(result.all())

    return EventStats(total_events=total, events_today=today, cats_detected=cats_detected)


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(
        select(DetectionEvent)
        .options(joinedload(DetectionEvent.cat), joinedload(DetectionEvent.camera))
        .where(DetectionEvent.id == event_id)
    )
    event = result.unique().scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    resp = EventResponse.model_validate(event)
    if event.cat:
        resp.cat_name = event.cat.name
    if event.camera:
        resp.camera_name = event.camera.name
    return resp


@router.get("/{event_id}/thumbnail")
async def get_event_thumbnail(
    event_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(
        select(DetectionEvent).where(DetectionEvent.id == event_id)
    )
    event = result.scalar_one_or_none()
    if not event or not event.thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(event.thumbnail_path)
