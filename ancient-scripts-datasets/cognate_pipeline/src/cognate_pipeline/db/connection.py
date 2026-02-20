"""Database engine and session factory."""

from __future__ import annotations

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

from cognate_pipeline.config.schema import DatabaseConfig

_engine_cache: dict[str, Engine] = {}


def get_engine(config: DatabaseConfig) -> Engine:
    """Get or create a SQLAlchemy engine for the given config."""
    url = config.url
    if url not in _engine_cache:
        _engine_cache[url] = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine_cache[url]


def get_session(engine: Engine) -> Session:
    """Create a new session bound to the given engine."""
    factory = sessionmaker(bind=engine)
    return factory()
