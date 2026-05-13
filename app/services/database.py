# app/services/database.py
from typing import Any, Optional, Dict
from urllib.parse import urlparse
import json
import uuid

import asyncpg
import numpy as np

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
    """Initialize schema - only create tables if they don't exist"""
    pool = await get_pool()
    if pool is None:
        print("No DATABASE_URL configured, skipping schema initialization")
        return
    
    async with pool.acquire() as conn:
        # Check if biometric_users exists
        try:
            # Create biometric_users if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS biometric_users (
                    user_id SERIAL PRIMARY KEY,
                    display_name TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            print("✓ biometric_users table ready")
            
            # Create biometric_sample if not exists (matching your schema)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS biometric_sample (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    embedding JSONB NOT NULL,
                    quality DOUBLE PRECISION,
                    "isActive" BOOLEAN DEFAULT true,
                    "capturedAt" TIMESTAMPTZ DEFAULT NOW(),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            print("✓ biometric_sample table ready")
            
            # Create biometric_attempt if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS biometric_attempt (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    verified BOOLEAN NOT NULL,
                    confidence FLOAT NOT NULL,
                    scores JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            print("✓ biometric_attempt table ready")
            
            # Create indexes if they don't exist
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_biometric_sample_user_id ON biometric_sample(user_id);
                CREATE INDEX IF NOT EXISTS idx_biometric_attempt_user_id ON biometric_attempt(user_id);
            """)
            
        except Exception as e:
            print(f"Warning: Schema initialization issue: {e}")
            # Don't fail - just continue


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


async def save_biometric_template(user_id: int, modality: str, embedding: np.ndarray, quality: float = 0.0) -> None:
    """Save or update a biometric template for a user"""
    pool = await get_pool()
    if pool is None:
        raise RuntimeError("DATABASE_URL is not configured")
    
    # Convert numpy array to list for JSON storage
    embedding_list = embedding.tolist()
    template_id = str(uuid.uuid4())
    
    async with pool.acquire() as conn:
        # Check if template exists for this user and modality
        existing = await conn.fetchrow(
            "SELECT id FROM biometric_sample WHERE user_id = $1 AND modality = $2",
            str(user_id), modality
        )
        
        if existing:
            # Update existing template
            await conn.execute(
                """
                UPDATE biometric_sample 
                SET embedding = $3, quality = $4, updated_at = NOW()
                WHERE user_id = $1 AND modality = $2
                """,
                str(user_id), modality, json.dumps(embedding_list), quality
            )
        else:
            # Insert new template
            await conn.execute(
                """
                INSERT INTO biometric_sample (id, user_id, modality, embedding, quality, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                """,
                template_id, str(user_id), modality, json.dumps(embedding_list), quality
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
            WHERE user_id = $1 AND "isActive" = true
            """,
            str(user_id)
        )
    
    templates = {}
    for row in rows:
        modality = row['modality']
        embedding_data = row['embedding']
        # Convert JSON/list back to numpy array
        if isinstance(embedding_data, str):
            embedding_data = json.loads(embedding_data)
        elif isinstance(embedding_data, dict):
            embedding_data = list(embedding_data.values()) if hasattr(embedding_data, 'values') else embedding_data
        templates[modality] = np.array(embedding_data)
    
    return templates


async def log_verification_attempt(user_id: int, verified: bool, confidence: float, scores: Dict[str, float]) -> None:
    """Log a verification attempt for auditing"""
    pool = await get_pool()
    if pool is None:
        return
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO biometric_attempt (user_id, verified, confidence, scores, created_at)
            VALUES ($1, $2, $3, $4, NOW())
            """,
            user_id, verified, confidence, json.dumps(scores)
        )


def is_configured() -> bool:
    return bool(settings.DATABASE_URL and settings.DATABASE_URL.strip())
