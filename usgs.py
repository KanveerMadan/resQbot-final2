"""
usgs.py — USGS Earthquake Catalog fetcher for ResQbot.

Responsibilities:
  - Fetch recent earthquake events around a given lat/lon/radius
  - Fetch 5-year historical summary for onboarding report
  - Normalize raw GeoJSON into clean dicts ready for prediction.py
  - Handle all null fields defensively (USGS data is frequently incomplete)
  - Never raise — log and return empty on any network/parse failure
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Minimum magnitude for real-time polling (lowered to 3.5 for fault-zone users;
# the caller passes whichever threshold is appropriate).
DEFAULT_MIN_MAG = 4.0
FAULT_ZONE_MIN_MAG = 3.5

# How far back to look on each polling cycle (slightly more than the scheduler
# interval to avoid missing events at boundary).
POLL_LOOKBACK_MINUTES = 12

# Haversine Earth radius
_EARTH_RADIUS_KM = 6371.0

# Request timeout — Render free tier; be generous but don't hang forever.
_TIMEOUT_SECONDS = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_recent_events(
    latitude: float,
    longitude: float,
    radius_km: int,
    near_fault: bool = False,
    lookback_minutes: int = POLL_LOOKBACK_MINUTES,
) -> list[dict]:
    """
    Fetch earthquake events within `radius_km` of the given coordinates
    that occurred in the last `lookback_minutes` minutes.

    Returns a list of normalised event dicts (see _normalise_feature).
    Returns [] on any error — callers should treat an empty list as
    "nothing new" and continue normally.
    """
    min_mag = FAULT_ZONE_MIN_MAG if near_fault else DEFAULT_MIN_MAG
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=lookback_minutes)

    params = {
        "format": "geojson",
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "starttime": _fmt_time(start_time),
        "endtime": _fmt_time(end_time),
        "orderby": "time",
        "limit": 20,
    }

    features = _query_usgs(params)
    events = []
    for feature in features:
        event = _normalise_feature(feature, latitude, longitude)
        if event is not None:
            events.append(event)
    return events


def fetch_historical_summary(
    latitude: float,
    longitude: float,
    radius_km: int,
    years: int = 5,
) -> dict:
    """
    Query USGS for all M4.0+ events within `radius_km` over the past `years`.
    Returns a summary dict used in the onboarding historical report message.

    Return shape:
    {
        "total_events": int,
        "max_mag": float | None,
        "max_mag_date": str | None,   # "15 Mar 2022"
        "last_event_mag": float | None,
        "last_event_date": str | None,
        "last_event_distance_km": float | None,
    }
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=365 * years)

    params = {
        "format": "geojson",
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": radius_km,
        "minmagnitude": 4.0,
        "starttime": _fmt_time(start_time),
        "endtime": _fmt_time(end_time),
        "orderby": "time",
        "limit": 1000,
    }

    features = _query_usgs(params)
    if not features:
        return _empty_summary()

    total = len(features)
    max_mag: Optional[float] = None
    max_mag_date: Optional[str] = None
    last_event_mag: Optional[float] = None
    last_event_date: Optional[str] = None
    last_event_distance_km: Optional[float] = None

    # Features are ordered newest-first from USGS
    for i, feature in enumerate(features):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [])

        mag = _safe_float(props.get("mag"))
        event_time_ms = props.get("time")
        event_time = _parse_usgs_time(event_time_ms)

        if i == 0 and mag is not None:
            last_event_mag = mag
            last_event_date = _fmt_human(event_time) if event_time else None
            if len(coords) >= 2:
                last_event_distance_km = round(
                    haversine_km(latitude, longitude, coords[1], coords[0]), 1
                )

        if mag is not None and (max_mag is None or mag > max_mag):
            max_mag = mag
            max_mag_date = _fmt_human(event_time) if event_time else None

    return {
        "total_events": total,
        "max_mag": max_mag,
        "max_mag_date": max_mag_date,
        "last_event_mag": last_event_mag,
        "last_event_date": last_event_date,
        "last_event_distance_km": last_event_distance_km,
    }


def fetch_events_for_digest(
    latitude: float,
    longitude: float,
    radius_km: int,
    near_fault: bool = False,
) -> list[dict]:
    """
    Fetch the past 7 days of events for the weekly Sunday digest.
    Returns normalised event dicts, same shape as fetch_recent_events.
    """
    min_mag = FAULT_ZONE_MIN_MAG if near_fault else DEFAULT_MIN_MAG
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)

    params = {
        "format": "geojson",
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "starttime": _fmt_time(start_time),
        "endtime": _fmt_time(end_time),
        "orderby": "time",
        "limit": 200,
    }

    features = _query_usgs(params)
    events = []
    for feature in features:
        event = _normalise_feature(feature, latitude, longitude)
        if event is not None:
            events.append(event)
    return events


def fetch_events_for_aftershock_check(
    latitude: float,
    longitude: float,
    radius_km: float = 50.0,
    lookback_hours: int = 2,
    min_mag: float = 2.5,
) -> list[dict]:
    """
    Used by the aftershock cluster detector in scheduler.py.
    Fetches events near a mainshock epicenter over the past `lookback_hours`.
    Lower magnitude threshold since aftershocks can be smaller.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=lookback_hours)

    params = {
        "format": "geojson",
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": radius_km,
        "minmagnitude": min_mag,
        "starttime": _fmt_time(start_time),
        "endtime": _fmt_time(end_time),
        "orderby": "time",
        "limit": 50,
    }

    features = _query_usgs(params)
    events = []
    for feature in features:
        event = _normalise_feature(feature, latitude, longitude)
        if event is not None:
            events.append(event)
    return events


# ---------------------------------------------------------------------------
# Haversine distance (exposed so prediction.py / scheduler.py can import it)
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _query_usgs(params: dict) -> list:
    """
    Execute a single USGS FDSN query and return the feature list.
    Returns [] on any error without raising.
    """
    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.get(USGS_BASE, params=params)
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])
            logger.debug("USGS returned %d features for params %s", len(features), params)
            return features
    except httpx.TimeoutException:
        logger.warning("USGS request timed out (params=%s)", params)
    except httpx.HTTPStatusError as exc:
        logger.warning("USGS HTTP error %s (params=%s)", exc.response.status_code, params)
    except Exception as exc:
        logger.exception("Unexpected error fetching USGS data: %s", exc)
    return []


def _normalise_feature(
    feature: dict,
    user_lat: float,
    user_lon: float,
) -> Optional[dict]:
    """
    Convert a raw USGS GeoJSON feature into a flat dict ready for
    prediction.py and alert construction.

    Returns None if the feature is missing critical fields (mag, coords).

    Output shape:
    {
        "usgs_id":      str,
        "latitude":     float,
        "longitude":    float,
        "depth":        float,
        "mag":          float,
        "gap":          float,      # defaults to 180.0 if null
        "place":        str,        # human-readable place name from USGS
        "event_time":   datetime,   # UTC
        "distance_km":  float,      # great-circle km from user
    }
    """
    try:
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [])
        usgs_id = feature.get("id", "")

        if not coords or len(coords) < 3:
            logger.debug("Skipping feature %s — missing coordinates", usgs_id)
            return None

        lon = _safe_float(coords[0])
        lat = _safe_float(coords[1])
        depth = _safe_float(coords[2])
        mag = _safe_float(props.get("mag"))

        if lat is None or lon is None or depth is None or mag is None:
            logger.debug("Skipping feature %s — null critical field", usgs_id)
            return None

        # gap: USGS frequently returns null; default to 180.0 (model trained with this)
        gap = _safe_float(props.get("gap")) or 180.0

        place = props.get("place") or "Unknown location"
        event_time = _parse_usgs_time(props.get("time"))

        distance_km = round(haversine_km(user_lat, user_lon, lat, lon), 1)

        return {
            "usgs_id": usgs_id,
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
            "mag": mag,
            "gap": gap,
            "place": place,
            "event_time": event_time,
            "distance_km": distance_km,
        }

    except Exception as exc:
        logger.warning("Failed to normalise USGS feature: %s", exc)
        return None


def _safe_float(value) -> Optional[float]:
    """Cast to float, return None if null or non-numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_usgs_time(ms_timestamp) -> Optional[datetime]:
    """Convert USGS Unix epoch milliseconds to a UTC datetime."""
    if ms_timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(int(ms_timestamp) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _fmt_time(dt: datetime) -> str:
    """Format datetime as USGS-compatible ISO 8601 string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _fmt_human(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime as a readable string for WhatsApp messages."""
    if dt is None:
        return None
    return dt.strftime("%-d %b %Y")  # e.g. "3 Apr 2023"


def _empty_summary() -> dict:
    return {
        "total_events": 0,
        "max_mag": None,
        "max_mag_date": None,
        "last_event_mag": None,
        "last_event_date": None,
        "last_event_distance_km": None,
    }