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


@router.get("/")
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
    # Build filter conditions
    conditions = []
    if camera_id is not None:
        conditions.append(DetectionEvent.camera_id == camera_id)
    if cat_id is not None:
        conditions.append(DetectionEvent.cat_id == cat_id)
    if start_time is not None:
        conditions.append(DetectionEvent.timestamp >= start_time)
    if end_time is not None:
        conditions.append(DetectionEvent.timestamp <= end_time)

    # Count total matching
    count_query = select(func.count(DetectionEvent.id))
    for cond in conditions:
        count_query = count_query.where(cond)
    total = (await db.execute(count_query)).scalar()

    # Fetch page
    query = (
        select(DetectionEvent)
        .options(joinedload(DetectionEvent.cat), joinedload(DetectionEvent.camera))
    )
    for cond in conditions:
        query = query.where(cond)
    query = query.order_by(DetectionEvent.timestamp.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    events = result.unique().scalars().all()

    # Populate cat_name and camera_name
    items = []
    for event in events:
        resp = EventResponse.model_validate(event)
        if event.cat:
            resp.cat_name = event.cat.name
        if event.camera:
            resp.camera_name = event.camera.name
        items.append(resp)

    return {"items": items, "total": total, "limit": limit, "offset": offset}


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
