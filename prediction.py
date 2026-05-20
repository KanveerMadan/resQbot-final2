"""
prediction.py — ML inference and urgency classification for ResQbot.

Responsibilities:
  - Lazy-load combined_model.pkl once on first use (avoids 20–40s cold starts)
  - Run 2-model majority vote (LR + RF) on a normalised USGS event dict
  - Classify result into 4-tier urgency system (GREEN / YELLOW / ORANGE / RED)
  - Return a PredictionResult dataclass consumed by scheduler.py and webhook.py
  - Never raise — log and return a safe BELOW_THRESHOLD result on any failure
"""

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "combined_model.pkl")


TIER_GREEN = "GREEN"
TIER_YELLOW = "YELLOW"
TIER_ORANGE = "ORANGE"
TIER_RED = "RED"
TIER_BELOW = "BELOW_THRESHOLD"


VOTE_THRESHOLD = 0.5

# Urgency rules (mag + depth)
# RED:    mag >= 6.5 AND depth < 70 km
# ORANGE: mag >= 5.5 OR (depth < 30 km AND mag >= 5.0)
# YELLOW: mag >= 4.5
# GREEN:  mag >= 4.0  (floor — anything below never reaches classify_tier)
_RED_MAG = 6.5
_RED_DEPTH = 70.0
_ORANGE_MAG = 5.5
_ORANGE_SHALLOW_MAG = 5.0
_ORANGE_SHALLOW_DEPTH = 30.0
_YELLOW_MAG = 4.5
_GREEN_MAG = 4.0


@dataclass
class PredictionResult:
    """
    Everything downstream (scheduler, whatsapp, eventlog) needs in one place.

    Fields:
        usgs_id         — USGS event ID, passed through for deduplication
        mag             — earthquake magnitude
        depth           — depth in km
        latitude        — epicentre latitude
        longitude       — epicentre longitude
        gap             — azimuthal gap used as model input
        lr_score        — raw LR probability (0.0–1.0)
        rf_score        — raw RF probability (0.0–1.0)
        confidence      — number of models that voted ≥ 0.5  (0 / 1 / 2)
        earthquake_likely — True if at least 1 of 2 models voted yes
        tier            — GREEN / YELLOW / ORANGE / RED / BELOW_THRESHOLD
        place           — human-readable place string from USGS
        event_time      — UTC datetime of the seismic event
        distance_km     — great-circle km from the user that triggered fetch
    """
    usgs_id: str
    mag: float
    depth: float
    latitude: float
    longitude: float
    gap: float
    lr_score: float
    rf_score: float
    confidence: int
    earthquake_likely: bool
    tier: str
    place: str
    event_time: Optional[object]   
    distance_km: float


_bundle: Optional[dict] = None
_bundle_lock = Lock()


def _load_bundle() -> Optional[dict]:
    """
    Load combined_model.pkl exactly once across all threads.
    Returns None if the file is missing or corrupt — callers handle gracefully.
    """
    global _bundle
    if _bundle is not None:
        return _bundle

    with _bundle_lock:
        
        if _bundle is not None:
            return _bundle
        try:
            loaded = joblib.load(MODEL_PATH)
            
            required = {"lr_model", "rf_model", "scaler"}
            missing = required - set(loaded.keys())
            if missing:
                logger.error("combined_model.pkl is missing keys: %s", missing)
                return None
            _bundle = loaded
            logger.info("Model bundle loaded from %s", MODEL_PATH)
            return _bundle
        except FileNotFoundError:
            logger.error("Model file not found: %s", MODEL_PATH)
        except Exception as exc:
            logger.exception("Failed to load model bundle: %s", exc)
        return None


def _get_models():
    """Unpack bundle into (lr_model, rf_model, scaler) or raise RuntimeError."""
    bundle = _load_bundle()
    if bundle is None:
        raise RuntimeError("Model bundle unavailable — check MODEL_PATH and logs")
    return bundle["lr_model"], bundle["rf_model"], bundle["scaler"]


def predict(event: dict) -> PredictionResult:
    """
    Run ML inference and urgency classification on a single normalised event dict
    (as returned by usgs._normalise_feature).

    Always returns a PredictionResult — never raises.  On model failure, returns
    a BELOW_THRESHOLD result so the scheduler can still log and continue.
    """
    usgs_id      = event.get("usgs_id", "unknown")
    mag          = float(event.get("mag", 0.0))
    depth        = float(event.get("depth", 0.0))
    latitude     = float(event.get("latitude", 0.0))
    longitude    = float(event.get("longitude", 0.0))
    gap          = float(event.get("gap", 180.0))
    place        = event.get("place", "Unknown location")
    event_time   = event.get("event_time")
    distance_km  = float(event.get("distance_km", 0.0))

    
    features = np.array([[latitude, longitude, depth, mag, gap]], dtype=float)

    lr_score = 0.0
    rf_score = 0.0
    confidence = 0
    earthquake_likely = False

    try:
        lr_model, rf_model, scaler = _get_models()
        scaled = scaler.transform(features)

        lr_prob = float(lr_model.predict_proba(scaled)[0, 1])
        rf_prob = float(rf_model.predict_proba(scaled)[0, 1])

        lr_vote = int(lr_prob >= VOTE_THRESHOLD)
        rf_vote = int(rf_prob >= VOTE_THRESHOLD)

        lr_score = round(lr_prob, 4)
        rf_score = round(rf_prob, 4)
        confidence = lr_vote + rf_vote
        earthquake_likely = confidence >= 1

        logger.debug(
            "Event %s | mag=%.1f depth=%.1f | lr=%.3f rf=%.3f confidence=%d",
            usgs_id, mag, depth, lr_prob, rf_prob, confidence,
        )

    except RuntimeError as exc:
        
        logger.error("Prediction skipped for %s: %s", usgs_id, exc)

    except Exception as exc:
        logger.exception("Unexpected prediction error for %s: %s", usgs_id, exc)

    tier = classify_tier(mag, depth, earthquake_likely)

    return PredictionResult(
        usgs_id=usgs_id,
        mag=mag,
        depth=depth,
        latitude=latitude,
        longitude=longitude,
        gap=gap,
        lr_score=lr_score,
        rf_score=rf_score,
        confidence=confidence,
        earthquake_likely=earthquake_likely,
        tier=tier,
        place=place,
        event_time=event_time,
        distance_km=distance_km,
    )


def classify_tier(mag: float, depth: float, earthquake_likely: bool) -> str:
    """
    Map magnitude + depth + model vote to a 4-tier urgency string.

    Rules (evaluated top-down, first match wins):
      BELOW_THRESHOLD — model said no (confidence == 0) AND mag < GREEN floor
      RED             — mag >= 6.5 AND depth < 70 km
      ORANGE          — mag >= 5.5  OR  (depth < 30 km AND mag >= 5.0)
      YELLOW          — mag >= 4.5
      GREEN           — mag >= 4.0

    NOTE: Tier is determined by physical parameters even when both models voted
    no.  If magnitude is high enough (RED/ORANGE territory), we alert regardless
    — the ML vote informs confidence score, not the alert gate at these extremes.
    This is an intentional safety-over-precision choice.
    """
    
    if not earthquake_likely and mag < _GREEN_MAG:
        return TIER_BELOW

    if mag >= _RED_MAG and depth < _RED_DEPTH:
        return TIER_RED

    if mag >= _ORANGE_MAG or (depth < _ORANGE_SHALLOW_DEPTH and mag >= _ORANGE_SHALLOW_MAG):
        return TIER_ORANGE

    if mag >= _YELLOW_MAG:
        return TIER_YELLOW

    if mag >= _GREEN_MAG:
        return TIER_GREEN

    return TIER_BELOW


def is_alertable(result: PredictionResult) -> bool:
    """
    Return True if the result should trigger a WhatsApp alert.
    BELOW_THRESHOLD events are logged to EventLog but never messaged.
    """
    return result.tier != TIER_BELOW


def bypasses_quiet_hours(result: PredictionResult) -> bool:
    """RED alerts go out regardless of quiet hours (10pm–7am local)."""
    return result.tier == TIER_RED


def predict_batch(events: list[dict]) -> list[PredictionResult]:
    """
    Run predict() over a list of normalised event dicts.
    Results are returned in the same order as input.
    Failures on individual events are logged and skipped.
    """
    results = []
    for event in events:
        try:
            results.append(predict(event))
        except Exception as exc:
            logger.exception("predict_batch: skipping event due to error: %s", exc)
    return results