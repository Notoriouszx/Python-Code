# app/services/database.py (add these to init_schema)
from typing import Any, Optional
from urllib.parse import urlparse

import asyncpg

from app.config import settings

_pool: Optional[asyncpg.Pool] = None


def _pool_connect_kwargs(url: str) -> dict[str, Any]:
    """Build asyncpg pool kwargs. SSL is explicit unless sslmode= is already in the DSN."""
    kwargs: dict[str, Any] = {"min_size": 1, "max_size": 10}
    if "sslmode=" in url.lower():
        return kwargs
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        kwargs["ssl"] = True
        return kwargs
    # Local / docker-compose postgres (see docker-compose.yml service name `db`) without TLS
    if host in ("localhost", "127.0.0.1", "::1", "db"):
        kwargs["ssl"] = False
    else:
        kwargs["ssl"] = True
    return kwargs


async def get_pool() -> Optional[asyncpg.Pool]:
    global _pool
    if not settings.DATABASE_URL:
        return None
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.DATABASE_URL, **_pool_connect_kwargs(settings.DATABASE_URL))
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
        # Existing users table
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS biometric_users (
                user_id SERIAL PRIMARY KEY,
                display_name TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        
        # NEW: Table for storing biometric templates/embeddings
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS biometric_sample (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES biometric_users(user_id) ON DELETE CASCADE,
                modality VARCHAR(50) NOT NULL,
                embedding JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(user_id, modality)
            );
            """
        )
        
        # NEW: Table for logging verification attempts
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS biometric_attempt (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES biometric_users(user_id) ON DELETE CASCADE,
                verified BOOLEAN NOT NULL,
                confidence FLOAT NOT NULL,
                scores JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        
        # Create indexes for better performance
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_biometric_sample_user_id ON biometric_sample(user_id);
            CREATE INDEX IF NOT EXISTS idx_biometric_attempt_user_id ON biometric_attempt(user_id);
            CREATE INDEX IF NOT EXISTS idx_biometric_attempt_created_at ON biometric_attempt(created_at);
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


async def save_biometric_template(user_id: int, modality: str, embedding: np.ndarray) -> None:
    """Save or update a biometric template for a user"""
    pool = await get_pool()
    if pool is None:
        raise RuntimeError("DATABASE_URL is not configured")
    
    # Convert numpy array to list for JSON storage
    embedding_list = embedding.tolist()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO biometric_sample (user_id, modality, embedding, created_at, updated_at)
            VALUES ($1, $2, $3, NOW(), NOW())
            ON CONFLICT (user_id, modality) 
            DO UPDATE SET embedding = $3, updated_at = NOW()
            """,
            user_id, modality, embedding_list
        )


async def get_biometric_templates(user_id: int) -> Dict[str, np.ndarray]:
    """Retrieve all biometric templates for a user"""
    pool = await get_pool()
    if pool is None:
        return {}
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT modality, embedding 
            FROM biometric_sample 
            WHERE user_id = $1
            """,
            user_id
        )
    
    templates = {}
    for row in rows:
        modality = row['modality']
        embedding_data = row['embedding']
        # Convert JSON/list back to numpy array
        if isinstance(embedding_data, str):
            import json
            embedding_data = json.loads(embedding_data)
        templates[modality] = np.array(embedding_data)
    
    return templates


async def log_verification_attempt(user_id: int, verified: bool, confidence: float, scores: Dict[str, float]) -> None:
    """Log a verification attempt for auditing"""
    pool = await get_pool()
    if pool is None:
        return
    
    import json
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO biometric_attempt (user_id, verified, confidence, scores, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            """,
            user_id, verified, confidence, json.dumps(scores)
        )


def is_configured() -> bool:
    return bool(settings.DATABASE_URL.strip())
