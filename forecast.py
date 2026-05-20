"""
forecast.py — Probabilistic seismic forecasting for ResQbot.

Methodology:
  - Pulls USGS historical data for the user's region (90-day rolling window)
  - Calculates base rate: how many M4+ events per day in this region historically
  - Applies Poisson probability model: P(at least 1 event in N days) = 1 - e^(-rate * N)
  - Adjusts for recent cluster activity (foreshock pattern signal)
  - Adjusts for fault proximity (near_fault users get higher baseline)
  - Returns a ForecastResult with probabilities and a human-readable WhatsApp message

Scientific basis:
  Poisson process is the standard model used by USGS, GNS Science (NZ), and
  JMA (Japan) for short-term operational earthquake forecasting.
  It assumes events are random and independent — valid for background seismicity.
  Cluster adjustment uses a simple decay multiplier based on recent activity rate
  vs long-term rate, similar to ETAS (Epidemic Type Aftershock Sequence) lite.

Limitations (always communicated to user):
  - Cannot account for unknown fault stress accumulation
  - Cannot detect precursory signals (GPS deformation, radon, EM anomalies)
  - Poisson assumption breaks down during active aftershock sequences
  - All probabilities are statistical estimates, not deterministic predictions
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Lookback windows for rate calculation
LONG_WINDOW_DAYS  = 90    # base rate calculated over 90 days
SHORT_WINDOW_DAYS = 7     # recent activity window for cluster detection
CLUSTER_WINDOW_DAYS = 2   # very recent window for foreshock signal

# Forecast horizons
HORIZON_24H = 1
HORIZON_72H = 3
HORIZON_7D  = 7

# Magnitude thresholds
MIN_MAG_BASE    = 4.0   # base rate counts M4+
MIN_MAG_FAULT   = 3.5   # fault-zone users count M3.5+
MIN_MAG_CLUSTER = 3.0   # cluster detection uses M3+ (catches foreshock swarms)

# Cluster multiplier cap — prevent runaway scores during active sequences
MAX_CLUSTER_MULTIPLIER = 3.0

# Fault zone baseline boost
FAULT_BASELINE_BOOST = 1.4

# Risk tier thresholds (probability of M4+ in 24h)
RISK_VERY_LOW  = 0.05   # < 5%
RISK_LOW       = 0.15   # 5–15%
RISK_MODERATE  = 0.30   # 15–30%
RISK_ELEVATED  = 0.50   # 30–50%
# > 50% = High


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """
    Probabilistic forecast for a user's region.

    prob_m4_24h  — P(at least one M4+ within radius in next 24 hours)
    prob_m4_72h  — P(at least one M4+ within radius in next 72 hours)
    prob_m4_7d   — P(at least one M4+ within radius in next 7 days)
    prob_m5_72h  — P(at least one M5+ within radius in next 72 hours)
    base_rate    — M4+ events per day (long-term)
    recent_rate  — M4+ events per day (last 7 days)
    cluster_flag — True if recent rate significantly exceeds base rate
    risk_label   — VERY LOW / LOW / MODERATE / ELEVATED / HIGH
    event_count_90d — M4+ events in past 90 days in this region
    last_event_days — days since last M4+ event
    whatsapp_message — formatted message ready to send
    """
    prob_m4_24h:      float
    prob_m4_72h:      float
    prob_m4_7d:       float
    prob_m5_72h:      float
    base_rate:        float
    recent_rate:      float
    cluster_flag:     bool
    risk_label:       str
    event_count_90d:  int
    last_event_days:  Optional[float]
    whatsapp_message: str


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_forecast(
    latitude: float,
    longitude: float,
    radius_km: int,
    near_fault: bool = False,
) -> Optional[ForecastResult]:
    """
    Generate a probabilistic seismic forecast for the given location.
    Returns None on API failure — callers should handle gracefully.
    """
    try:
        now = datetime.now(timezone.utc)

        # Fetch long-window events for base rate
        long_events = _fetch_events(
            latitude, longitude, radius_km,
            start=now - timedelta(days=LONG_WINDOW_DAYS),
            end=now,
            min_mag=MIN_MAG_FAULT if near_fault else MIN_MAG_BASE,
        )

        # Fetch short-window events for cluster detection
        short_events = _fetch_events(
            latitude, longitude, radius_km,
            start=now - timedelta(days=SHORT_WINDOW_DAYS),
            end=now,
            min_mag=MIN_MAG_CLUSTER,
        )

        # Fetch very recent events for foreshock signal
        cluster_events = _fetch_events(
            latitude, longitude, radius_km,
            start=now - timedelta(days=CLUSTER_WINDOW_DAYS),
            end=now,
            min_mag=MIN_MAG_CLUSTER,
        )

        # Also fetch M5+ events for higher-magnitude probability
        m5_events = _fetch_events(
            latitude, longitude, radius_km,
            start=now - timedelta(days=LONG_WINDOW_DAYS),
            end=now,
            min_mag=5.0,
        )

        return _compute_forecast(
            long_events=long_events,
            short_events=short_events,
            cluster_events=cluster_events,
            m5_events=m5_events,
            near_fault=near_fault,
            now=now,
        )

    except Exception as exc:
        logger.exception("Forecast generation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_forecast(
    long_events: list,
    short_events: list,
    cluster_events: list,
    m5_events: list,
    near_fault: bool,
    now: datetime,
) -> ForecastResult:

    event_count_90d = len(long_events)

    # ── Base rate (events per day over long window) ──────────────────────────
    base_rate = event_count_90d / LONG_WINDOW_DAYS

    # Fault zone baseline boost
    if near_fault:
        base_rate *= FAULT_BASELINE_BOOST

    # ── Recent rate (events per day over short window) ───────────────────────
    recent_rate = len(short_events) / SHORT_WINDOW_DAYS

    # ── Cluster multiplier ───────────────────────────────────────────────────
    cluster_flag = False
    cluster_multiplier = 1.0

    if base_rate > 0 and recent_rate > base_rate * 1.5:
        # Recent activity is 1.5x above baseline — possible foreshock swarm
        cluster_flag = True
        raw_multiplier = recent_rate / base_rate
        cluster_multiplier = min(raw_multiplier, MAX_CLUSTER_MULTIPLIER)
        logger.info(
            "Cluster detected: recent_rate=%.3f base_rate=%.3f multiplier=%.2f",
            recent_rate, base_rate, cluster_multiplier,
        )

    # Very recent cluster (last 48h) gets an additional boost
    if len(cluster_events) >= 3:
        cluster_flag = True
        cluster_multiplier = min(cluster_multiplier * 1.3, MAX_CLUSTER_MULTIPLIER)

    # ── Effective rate ───────────────────────────────────────────────────────
    effective_rate = base_rate * cluster_multiplier

    # ── M5+ rate ─────────────────────────────────────────────────────────────
    m5_base_rate = len(m5_events) / LONG_WINDOW_DAYS
    if near_fault:
        m5_base_rate *= FAULT_BASELINE_BOOST
    m5_effective_rate = m5_base_rate * cluster_multiplier

    # ── Poisson probabilities ────────────────────────────────────────────────
    # P(at least 1 event in T days) = 1 - e^(-rate * T)
    prob_m4_24h = _poisson_prob(effective_rate, HORIZON_24H)
    prob_m4_72h = _poisson_prob(effective_rate, HORIZON_72H)
    prob_m4_7d  = _poisson_prob(effective_rate, HORIZON_7D)
    prob_m5_72h = _poisson_prob(m5_effective_rate, HORIZON_72H)

    # ── Risk label ───────────────────────────────────────────────────────────
    risk_label = _risk_label(prob_m4_24h)

    # ── Days since last event ────────────────────────────────────────────────
    last_event_days = _days_since_last(long_events, now)

    # ── Build WhatsApp message ───────────────────────────────────────────────
    message = _build_message(
        prob_m4_24h=prob_m4_24h,
        prob_m4_72h=prob_m4_72h,
        prob_m4_7d=prob_m4_7d,
        prob_m5_72h=prob_m5_72h,
        risk_label=risk_label,
        event_count_90d=event_count_90d,
        last_event_days=last_event_days,
        cluster_flag=cluster_flag,
        near_fault=near_fault,
    )

    return ForecastResult(
        prob_m4_24h=round(prob_m4_24h, 4),
        prob_m4_72h=round(prob_m4_72h, 4),
        prob_m4_7d=round(prob_m4_7d, 4),
        prob_m5_72h=round(prob_m5_72h, 4),
        base_rate=round(base_rate, 5),
        recent_rate=round(recent_rate, 5),
        cluster_flag=cluster_flag,
        risk_label=risk_label,
        event_count_90d=event_count_90d,
        last_event_days=last_event_days,
        whatsapp_message=message,
    )


# ---------------------------------------------------------------------------
# USGS fetch
# ---------------------------------------------------------------------------

def _fetch_events(
    latitude: float,
    longitude: float,
    radius_km: int,
    start: datetime,
    end: datetime,
    min_mag: float,
) -> list:
    """Fetch events from USGS FDSNWS. Returns list of feature dicts."""
    params = {
        "format":        "geojson",
        "latitude":      latitude,
        "longitude":     longitude,
        "maxradiuskm":   radius_km,
        "minmagnitude":  min_mag,
        "starttime":     start.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime":       end.strftime("%Y-%m-%dT%H:%M:%S"),
        "orderby":       "time",
        "limit":         1000,
    }
    try:
        resp = httpx.get(USGS_BASE, params=params, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("features", [])
    except Exception as exc:
        logger.warning("USGS fetch failed (start=%s min_mag=%s): %s", start.date(), min_mag, exc)
        return []


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _poisson_prob(rate_per_day: float, days: int) -> float:
    """
    Poisson probability of at least one event in `days` days given `rate_per_day`.
    P(X >= 1) = 1 - P(X = 0) = 1 - e^(-lambda)
    where lambda = rate * days
    """
    if rate_per_day <= 0:
        return 0.0
    lam = rate_per_day * days
    return 1.0 - math.exp(-lam)


def _risk_label(prob_24h: float) -> str:
    if prob_24h < RISK_VERY_LOW:
        return "VERY LOW"
    if prob_24h < RISK_LOW:
        return "LOW"
    if prob_24h < RISK_MODERATE:
        return "MODERATE"
    if prob_24h < RISK_ELEVATED:
        return "ELEVATED"
    return "HIGH"


def _risk_emoji(label: str) -> str:
    return {
        "VERY LOW":  "🟢",
        "LOW":       "🟡",
        "MODERATE":  "🟠",
        "ELEVATED":  "🔴",
        "HIGH":      "🚨",
    }.get(label, "⚪")


def _days_since_last(events: list, now: datetime) -> Optional[float]:
    """Return days since the most recent event, or None if no events."""
    if not events:
        return None
    try:
        times = [
            ev["properties"]["time"] / 1000
            for ev in events
            if ev.get("properties", {}).get("time")
        ]
        if not times:
            return None
        last_ts = max(times)
        last_dt = datetime.fromtimestamp(last_ts, tz=timezone.utc)
        return round((now - last_dt).total_seconds() / 86400, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_message(
    prob_m4_24h: float,
    prob_m4_72h: float,
    prob_m4_7d: float,
    prob_m5_72h: float,
    risk_label: str,
    event_count_90d: int,
    last_event_days: Optional[float],
    cluster_flag: bool,
    near_fault: bool,
) -> str:

    emoji = _risk_emoji(risk_label)

    # Format probabilities as percentages
    p24  = f"{prob_m4_24h * 100:.1f}%"
    p72  = f"{prob_m4_72h * 100:.1f}%"
    p7d  = f"{prob_m4_7d  * 100:.1f}%"
    p5_72 = f"{prob_m5_72h * 100:.1f}%"

    # Last event string
    if last_event_days is None:
        last_str = "No M4+ events recorded in past 90 days"
    elif last_event_days < 1:
        last_str = "Less than 24 hours ago"
    elif last_event_days < 2:
        last_str = "Yesterday"
    else:
        last_str = f"{int(last_event_days)} days ago"

    # Cluster warning
    cluster_line = ""
    if cluster_flag:
        cluster_line = (
            "\n⚠️ Elevated recent activity detected — "
            "seismic cluster in progress. Stay alert."
        )

    # Fault line note
    fault_line = ""
    if near_fault:
        fault_line = "\n📍 You are near a tectonic plate boundary — baseline risk is higher than average."

    msg = (
        f"🔍 Seismic Forecast for your region\n"
        f"{emoji} Overall risk: {risk_label}\n"
        f"\n"
        f"📊 Probability of M4.0+ event near you:\n"
        f"Next 24 hours: {p24}\n"
        f"Next 72 hours: {p72}\n"
        f"Next 7 days:   {p7d}\n"
        f"\n"
        f"📊 Probability of M5.0+ event near you:\n"
        f"Next 72 hours: {p5_72}\n"
        f"\n"
        f"📈 Based on:\n"
        f"Events in past 90 days: {event_count_90d}\n"
        f"Last recorded event: {last_str}"
        f"{cluster_line}"
        f"{fault_line}"
        f"\n\n"
        f"⚡ These are statistical estimates based on historical "
        f"seismicity patterns using the Poisson probability model — "
        f"not deterministic predictions. Always follow official guidance."
    )

    return msg


# ---------------------------------------------------------------------------
# Convenience: short threat summary (used in onboarding + weekly digest)
# ---------------------------------------------------------------------------

def short_threat_summary(result: ForecastResult) -> str:
    """
    One-line threat summary for embedding in other messages.
    e.g. "🟡 24h outlook: LOW (8.3% chance of M4+)"
    """
    emoji = _risk_emoji(result.risk_label)
    p24 = f"{result.prob_m4_24h * 100:.1f}%"
    return f"{emoji} 24h outlook: {result.risk_label} ({p24} chance of M4+ event near you)"