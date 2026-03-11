"""Database helpers."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.runtime_security import ensure_private_file
from app.models.history import Base


default_db_path = Path(settings.DATABASE_PATH).resolve()
DATABASE_URL = settings.DATABASE_URL or f"sqlite:///{default_db_path.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.DATABASE_ECHO,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    ensure_private_file(default_db_path)


def get_db() -> Session:
    """Return a database session."""
    init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
