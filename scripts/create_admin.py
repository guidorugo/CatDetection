#!/usr/bin/env python3
"""Create initial admin user."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session, engine
from app.core.database import Base
from app.core.security import hash_password
from app.models.user import User


async def create_admin():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as db:
        result = await db.execute(select(User).where(User.username == settings.ADMIN_USERNAME))
        if result.scalar_one_or_none():
            print(f"Admin user '{settings.ADMIN_USERNAME}' already exists")
            return

        user = User(
            username=settings.ADMIN_USERNAME,
            email=settings.ADMIN_EMAIL,
            hashed_password=hash_password(settings.ADMIN_PASSWORD),
            is_admin=True,
        )
        db.add(user)
        await db.commit()
        print(f"Created admin user: {settings.ADMIN_USERNAME}")


if __name__ == "__main__":
    asyncio.run(create_admin())
