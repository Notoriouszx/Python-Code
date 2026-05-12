from typing import Optional

import asyncpg

from app.config import settings

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> Optional[asyncpg.Pool]:
    global _pool
    if not settings.DATABASE_URL:
        return None
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=1, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_schema() -> None:
    pool = await get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS biometric_users (
                user_id SERIAL PRIMARY KEY,
                display_name TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )


async def insert_user(display_name: Optional[str]) -> int:
    pool = await get_pool()
    if pool is None:
        raise RuntimeError("DATABASE_URL is not configured")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO biometric_users (display_name) VALUES ($1) RETURNING user_id",
            display_name,
        )
        assert row is not None
        return int(row["user_id"])


async def user_exists(user_id: int) -> bool:
    pool = await get_pool()
    if pool is None:
        return False
    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT COUNT(*) FROM biometric_users WHERE user_id = $1", user_id
        )
        return int(n or 0) > 0


def is_configured() -> bool:
    return bool(settings.DATABASE_URL.strip())
