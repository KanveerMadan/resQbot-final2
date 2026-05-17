from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, create_engine, Session
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./resqbot.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class User(SQLModel, table=True):
    """One row per WhatsApp subscriber."""

    id: Optional[int] = Field(default=None, primary_key=True)
    phone: str = Field(unique=True, index=True)       # e.g. +919876543210
    latitude: float
    longitude: float
    radius_km: int = Field(default=300)               # 100 / 300 / 500
    timezone_offset: int = Field(default=0)           # hours, inferred from longitude
    near_fault: bool = Field(default=False)           # within 200 km of plate boundary
    active: bool = Field(default=True)                # False after STOP command
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Onboarding state — tracks where the user is in the setup flow
    # Values: "awaiting_location" | "awaiting_radius" | "active"
    onboarding_state: str = Field(default="awaiting_location")


class AlertLog(SQLModel, table=True):
    """One row per alert message sent to a user."""

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    usgs_event_id: str = Field(index=True)            # deduplication key
    tier: str                                         # GREEN / YELLOW / ORANGE / RED
    mag: float
    depth: float
    distance_km: float                                # km from user at time of alert
    sent_at: datetime = Field(default_factory=datetime.utcnow)

    # Are-you-safe check-in fields (only populated for RED alerts)
    checkin_sent: bool = Field(default=False)
    checkin_sent_at: Optional[datetime] = None
    checkin_response: Optional[str] = None            # "SAFE" | "HELP" | None
    checkin_responded_at: Optional[datetime] = None
    escalated: bool = Field(default=False)            # True if no reply after 60 min


class EventLog(SQLModel, table=True):
    """Every USGS event processed by the scheduler, regardless of alerting."""

    id: Optional[int] = Field(default=None, primary_key=True)
    usgs_event_id: str = Field(unique=True, index=True)
    mag: float
    depth: float
    latitude: float
    longitude: float
    gap: float
    lr_score: float                                   # raw LR probability
    rf_score: float                                   # raw RF probability
    confidence: int                                   # 0 / 1 / 2 models agreed ≥ 0.5
    tier: str                                         # GREEN / YELLOW / ORANGE / RED / BELOW_THRESHOLD
    processed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_db_and_tables() -> None:
    """Create all tables if they don't exist. Call once at app startup."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """
    Yield a database session. Intended for use as a FastAPI dependency
    or as a plain context manager in background tasks.

    Usage (FastAPI):
        @app.get("/...")
        def route(session: Session = Depends(get_session)):
            ...

    Usage (background task):
        with get_session() as session:
            ...
    """
    with Session(engine) as session:
        yield session