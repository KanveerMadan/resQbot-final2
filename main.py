"""
main.py — FastAPI application entry point for ResQbot.

Assembles all routers, initialises the database, starts the scheduler,
and exposes the handful of utility endpoints needed for operations.

Endpoints registered here:
  GET  /          — redirect to /admin
  GET  /health    — Render keepalive + basic liveness probe
  POST /predict   — manual prediction test endpoint (dev/debug only)

All WhatsApp webhook routes live in webhook.py (prefix: /webhook).
All admin dashboard routes live in dashboard.py (prefix: /admin).
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from models import create_db_and_tables
from prediction import classify_tier, predict
from scheduler import start_scheduler, stop_scheduler
import webhook
import dashboard

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.INFO)
logging.getLogger("anthropic").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Everything before `yield` runs at startup; everything after at shutdown.
    """
    logger.info("ResQbot starting up…")

    
    try:
        create_db_and_tables()
        logger.info("Database initialised")
    except Exception as exc:
        logger.critical("Database init failed — cannot start: %s", exc)
        raise

    
    try:
        from prediction import _load_bundle
        bundle = _load_bundle()
        if bundle is None:
            logger.warning(
                "Model bundle could not be loaded at startup — "
                "prediction will retry on first use"
            )
        else:
            logger.info("Model bundle pre-loaded successfully")
    except Exception as exc:
        logger.warning("Model pre-load failed (non-fatal): %s", exc)

    
    try:
        start_scheduler()
        logger.info("Background scheduler started")
    except Exception as exc:
        logger.critical("Scheduler failed to start: %s", exc)
        raise

    logger.info("ResQbot is ready")
    yield

    
    logger.info("ResQbot shutting down…")
    try:
        stop_scheduler()
    except Exception as exc:
        logger.warning("Scheduler shutdown error (non-fatal): %s", exc)
    logger.info("Shutdown complete")


app = FastAPI(
    title="ResQbot",
    description="WhatsApp-based earthquake alert system",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.include_router(webhook.router)
app.include_router(dashboard.router)

@app.get("/", include_in_schema=False)
def root():
    """Redirect naked domain hits to the admin dashboard."""
    return RedirectResponse("/admin", status_code=302)


@app.get("/health")
def health():
    """
    Liveness probe — used by:
      - Render's health check to decide whether to route traffic
      - scheduler.job_keepalive to prevent free-tier spin-down
    Returns 200 as long as the process is alive.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


class PredictRequest(BaseModel):
    latitude:  float
    longitude: float
    depth:     float
    mag:       float
    gap:       float = 180.0
    place:     str   = "Manual test"


@app.post("/predict")
def manual_predict(req: PredictRequest):
    """
    Manual prediction endpoint for testing the ML pipeline without needing
    a live USGS event. Disabled in production via ENABLE_PREDICT env var.

    Usage:
      curl -X POST https://<host>/predict \\
        -H 'Content-Type: application/json' \\
        -d '{"latitude":28.6,"longitude":77.2,"depth":10,"mag":5.5}'
    """
    if os.getenv("ENABLE_PREDICT", "false").lower() != "true":
        raise HTTPException(
            status_code=403,
            detail="Prediction endpoint is disabled. Set ENABLE_PREDICT=true to enable.",
        )

    event = {
        "usgs_id":     "manual_test",
        "latitude":    req.latitude,
        "longitude":   req.longitude,
        "depth":       req.depth,
        "mag":         req.mag,
        "gap":         req.gap,
        "place":       req.place,
        "event_time":  None,
        "distance_km": 0.0,
    }

    try:
        result = predict(event)
    except Exception as exc:
        logger.exception("Manual predict failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "usgs_id":          result.usgs_id,
        "mag":              result.mag,
        "depth":            result.depth,
        "latitude":         result.latitude,
        "longitude":        result.longitude,
        "gap":              result.gap,
        "lr_score":         result.lr_score,
        "rf_score":         result.rf_score,
        "confidence":       result.confidence,
        "earthquake_likely": result.earthquake_likely,
        "tier":             result.tier,
        "alertable":        result.tier not in ("BELOW_THRESHOLD",),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level=LOG_LEVEL.lower(),
        
        workers=1,
    )