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

class User(SQLModel, table=True):
    """One row per WhatsApp subscriber."""

    id: Optional[int] = Field(default=None, primary_key=True)
    phone: str = Field(unique=True, index=True)       
    latitude: float
    longitude: float
    radius_km: int = Field(default=300)               
    timezone_offset: int = Field(default=0)           
    near_fault: bool = Field(default=False)           
    active: bool = Field(default=True)                
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    
    onboarding_state: str = Field(default="awaiting_location")


class AlertLog(SQLModel, table=True):
    """One row per alert message sent to a user."""

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    usgs_event_id: str = Field(index=True)            
    tier: str                                         
    mag: float
    depth: float
    distance_km: float                                
    sent_at: datetime = Field(default_factory=datetime.utcnow)

    
    checkin_sent: bool = Field(default=False)
    checkin_sent_at: Optional[datetime] = None
    checkin_response: Optional[str] = None            
    checkin_responded_at: Optional[datetime] = None
    escalated: bool = Field(default=False)            


class EventLog(SQLModel, table=True):
    """Every USGS event processed by the scheduler, regardless of alerting."""

    id: Optional[int] = Field(default=None, primary_key=True)
    usgs_event_id: str = Field(unique=True, index=True)
    mag: float
    depth: float
    latitude: float
    longitude: float
    gap: float
    lr_score: float                                   
    rf_score: float                                   
    confidence: int                                   
    tier: str                                         
    processed_at: datetime = Field(default_factory=datetime.utcnow)



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