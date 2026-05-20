"""
scheduler.py — APScheduler background jobs for ResQbot.

Jobs:
  1. poll_seismic        — every 10 min: fetch USGS events per active user,
                           run prediction, send alerts, log to DB
  2. check_aftershocks   — every 10 min: detect clusters near recent M5.0+ events
  3. check_checkins      — every 5 min:  escalate unanswered RED check-ins after 60 min
  4. weekly_digest       — every Sunday 9am (per user's local tz): send digest
  5. keepalive           — every 14 min: hit /health so Render free tier doesn't sleep

All jobs are fire-and-forget. Any exception inside a job is caught and logged —
a failing job must never crash the scheduler or the FastAPI process.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlmodel import Session, select

from models import AlertLog, EventLog, User, engine
from prediction import PredictionResult, classify_tier, is_alertable, bypasses_quiet_hours, predict_batch
from usgs import (
    fetch_recent_events,
    fetch_events_for_aftershock_check,
    fetch_events_for_digest,
    haversine_km,
)
import whatsapp as wa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Aftershock cluster detection thresholds
CLUSTER_MAINSHOCK_MAG   = 5.0    # minimum mainshock magnitude to watch
CLUSTER_RADIUS_KM       = 50.0   # search radius around mainshock
CLUSTER_LOOKBACK_HOURS  = 2      # how far back to look for aftershocks
CLUSTER_MIN_COUNT       = 3      # minimum events to trigger cluster alert
CLUSTER_MIN_MAG         = 2.5    # minimum aftershock magnitude to count

# Check-in timing
CHECKIN_DELAY_MINUTES   = 30     # send check-in N minutes after RED alert
CHECKIN_ESCALATE_MINUTES = 90    # escalate N minutes after RED alert (30 + 60)

# Quiet hours (local time, inclusive)
QUIET_HOUR_START        = 22     # 10 PM
QUIET_HOUR_END          = 7      # 7 AM

# Keepalive
KEEPALIVE_URL           = "http://localhost:8000/health"
KEEPALIVE_INTERVAL_MIN  = 14


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------

_scheduler: Optional[BackgroundScheduler] = None


def start_scheduler() -> None:
    """
    Initialise and start the APScheduler instance.
    Called once from main.py on application startup.
    """
    global _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")

    # 1. Seismic polling — every 10 minutes
    _scheduler.add_job(
        job_poll_seismic,
        trigger=IntervalTrigger(minutes=10),
        id="poll_seismic",
        name="USGS seismic poll",
        max_instances=1,       # never overlap; USGS is slow sometimes
        misfire_grace_time=60,
    )

    # 2. Aftershock cluster check — every 10 minutes (offset by 2 min)
    _scheduler.add_job(
        job_check_aftershocks,
        trigger=IntervalTrigger(minutes=10, start_date=_offset_now(minutes=2)),
        id="check_aftershocks",
        name="Aftershock cluster detector",
        max_instances=1,
        misfire_grace_time=60,
    )

    # 3. Check-in escalation — every 5 minutes
    _scheduler.add_job(
        job_check_checkins,
        trigger=IntervalTrigger(minutes=5),
        id="check_checkins",
        name="RED alert check-in escalator",
        max_instances=1,
        misfire_grace_time=30,
    )

    # 4. Weekly digest — Sunday 9am UTC
    # Each user's local 9am is approximated by running hourly and filtering
    # by timezone offset inside the job (avoids 24 separate cron jobs).
    _scheduler.add_job(
        job_weekly_digest,
        trigger=CronTrigger(day_of_week="sun", hour="*", minute=0),
        id="weekly_digest",
        name="Sunday weekly digest",
        max_instances=1,
        misfire_grace_time=300,
    )

    # 5. Render keepalive — every 14 minutes
    _scheduler.add_job(
        job_keepalive,
        trigger=IntervalTrigger(minutes=KEEPALIVE_INTERVAL_MIN),
        id="keepalive",
        name="Render free tier keepalive",
        max_instances=1,
        misfire_grace_time=60,
    )

    _scheduler.start()
    logger.info("Scheduler started with %d jobs", len(_scheduler.get_jobs()))


def stop_scheduler() -> None:
    """Graceful shutdown — called from main.py on application shutdown."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


# ---------------------------------------------------------------------------
# Job 1 — Seismic poll
# ---------------------------------------------------------------------------

def job_poll_seismic() -> None:
    """
    For every active user: fetch recent USGS events, run ML prediction,
    deduplicate, apply quiet hours, send alerts, log everything to DB.
    """
    logger.debug("job_poll_seismic: starting")

    with Session(engine) as session:
        users = session.exec(select(User).where(User.active == True)).all()

    if not users:
        logger.debug("job_poll_seismic: no active users — skipping")
        return

    logger.info("job_poll_seismic: polling for %d active user(s)", len(users))

    for user in users:
        try:
            _poll_for_user(user)
        except Exception as exc:
            logger.exception("job_poll_seismic: error for user %s: %s", user.phone, exc)


def _poll_for_user(user: User) -> None:
    """Fetch, predict, deduplicate, and alert for a single user."""
    events = fetch_recent_events(
        user.latitude,
        user.longitude,
        user.radius_km,
        near_fault=user.near_fault,
    )

    if not events:
        return

    results = predict_batch(events)

    with Session(engine) as session:
        for result in results:
            _process_result(result, user, session)


def _process_result(result: PredictionResult, user: User, session: Session) -> None:
    """
    For one prediction result and one user:
      - Skip if below threshold
      - Skip if already alerted (dedup by usgs_event_id + user_id)
      - Skip if quiet hours (unless RED)
      - Send alert
      - Write AlertLog + EventLog
    """
    # Log every processed event regardless of alerting
    _upsert_event_log(result, session)

    if not is_alertable(result):
        return

    # Deduplication — have we already sent this event to this user?
    existing = session.exec(
        select(AlertLog)
        .where(AlertLog.user_id == user.id)
        .where(AlertLog.usgs_event_id == result.usgs_id)
    ).first()
    if existing:
        return

    # Quiet hours check
    if _is_quiet_hours(user) and not bypasses_quiet_hours(result):
        logger.debug(
            "Suppressing %s alert for %s (quiet hours)", result.tier, user.phone
        )
        return

    # Send the alert
    sent = wa.send_alert(user.phone, result)
    if not sent:
        logger.warning(
            "Failed to send %s alert to %s for event %s",
            result.tier, user.phone, result.usgs_id,
        )
        return

    # Persist AlertLog
    alert_log = AlertLog(
        user_id=user.id,
        usgs_event_id=result.usgs_id,
        tier=result.tier,
        mag=result.mag,
        depth=result.depth,
        distance_km=result.distance_km,
    )
    session.add(alert_log)
    session.commit()
    session.refresh(alert_log)

    logger.info(
        "Alert sent: %s M%.1f to %s (event %s)",
        result.tier, result.mag, user.phone, result.usgs_id,
    )

    # Schedule check-in for RED alerts
    if result.tier == "RED":
        _schedule_checkin(alert_log, user, session)


def _upsert_event_log(result: PredictionResult, session: Session) -> None:
    """
    Write to EventLog if this USGS event hasn't been logged yet.
    Multiple users can share the same event; we store it once.
    """
    existing = session.exec(
        select(EventLog).where(EventLog.usgs_event_id == result.usgs_id)
    ).first()
    if existing:
        return

    event_log = EventLog(
        usgs_event_id=result.usgs_id,
        mag=result.mag,
        depth=result.depth,
        latitude=result.latitude,
        longitude=result.longitude,
        gap=result.gap,
        lr_score=result.lr_score,
        rf_score=result.rf_score,
        confidence=result.confidence,
        tier=result.tier,
    )
    session.add(event_log)
    session.commit()


def _schedule_checkin(alert_log: AlertLog, user: User, session: Session) -> None:
    """
    Mark the AlertLog row so job_check_checkins knows to follow up.
    The actual send timing is handled by that job comparing timestamps.
    """
    alert_log.checkin_sent    = False   # not sent yet — job will send it
    alert_log.checkin_sent_at = None
    session.add(alert_log)
    session.commit()
    logger.info("Check-in scheduled for user %s alert_log_id=%d", user.phone, alert_log.id)


# ---------------------------------------------------------------------------
# Job 2 — Aftershock cluster detection
# ---------------------------------------------------------------------------

def job_check_aftershocks() -> None:
    """
    Find all M5.0+ events sent to any user in the last 2 hours.
    For each, check if 3+ smaller events have occurred within 50 km.
    Alert the relevant users once per mainshock cluster (dedup by usgs_event_id).
    """
    logger.debug("job_check_aftershocks: starting")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=CLUSTER_LOOKBACK_HOURS)

    with Session(engine) as session:
        # Find recent major events across all alert logs
        major_alerts = session.exec(
            select(AlertLog)
            .join(EventLog, AlertLog.usgs_event_id == EventLog.usgs_event_id)
            .where(EventLog.mag >= CLUSTER_MAINSHOCK_MAG)
            .where(AlertLog.sent_at >= cutoff)
        ).all()

        if not major_alerts:
            return

        # Deduplicate mainshocks — one cluster check per unique event
        seen_mainshocks: set[str] = set()

        for alert in major_alerts:
            if alert.usgs_event_id in seen_mainshocks:
                continue
            seen_mainshocks.add(alert.usgs_event_id)

            # Get mainshock epicentre from EventLog
            event_log = session.exec(
                select(EventLog).where(EventLog.usgs_event_id == alert.usgs_event_id)
            ).first()
            if not event_log:
                continue

            # Fetch aftershocks near the epicentre
            aftershocks = fetch_events_for_aftershock_check(
                event_log.latitude,
                event_log.longitude,
                radius_km=CLUSTER_RADIUS_KM,
                lookback_hours=CLUSTER_LOOKBACK_HOURS,
                min_mag=CLUSTER_MIN_MAG,
            )

            # Exclude the mainshock itself
            aftershocks = [
                e for e in aftershocks if e.get("usgs_id") != alert.usgs_event_id
            ]

            if len(aftershocks) < CLUSTER_MIN_COUNT:
                continue

            logger.info(
                "Cluster detected: %d aftershocks near mainshock %s",
                len(aftershocks), alert.usgs_event_id,
            )

            # Alert all users who received the original mainshock alert
            # and haven't received a cluster alert for this mainshock yet
            mainshock_user_ids = session.exec(
                select(AlertLog.user_id)
                .where(AlertLog.usgs_event_id == alert.usgs_event_id)
            ).all()

            for user_id in set(mainshock_user_ids):
                _send_cluster_alert_if_needed(
                    user_id, alert.usgs_event_id, event_log, len(aftershocks), session
                )


def _send_cluster_alert_if_needed(
    user_id: int,
    mainshock_id: str,
    event_log: EventLog,
    aftershock_count: int,
    session: Session,
) -> None:
    """Send a cluster alert to one user unless already sent for this mainshock."""
    user = session.get(User, user_id)
    if not user or not user.active:
        return

    # Dedup: use a synthetic event ID for the cluster alert
    cluster_event_id = f"cluster_{mainshock_id}"
    existing = session.exec(
        select(AlertLog)
        .where(AlertLog.user_id == user_id)
        .where(AlertLog.usgs_event_id == cluster_event_id)
    ).first()
    if existing:
        return

    if _is_quiet_hours(user):
        return

    sent = wa.send_cluster_alert(
        user.phone,
        aftershock_count,
        f"{event_log.latitude:.2f}°, {event_log.longitude:.2f}°",
        CLUSTER_RADIUS_KM,
    )
    if not sent:
        return

    cluster_log = AlertLog(
        user_id=user_id,
        usgs_event_id=cluster_event_id,
        tier="ORANGE",
        mag=event_log.mag,
        depth=event_log.depth,
        distance_km=haversine_km(
            user.latitude, user.longitude, event_log.latitude, event_log.longitude
        ),
    )
    session.add(cluster_log)
    session.commit()
    logger.info("Cluster alert sent to user %s for mainshock %s", user.phone, mainshock_id)


# ---------------------------------------------------------------------------
# Job 3 — Check-in escalation
# ---------------------------------------------------------------------------

def job_check_checkins() -> None:
    """
    1. Send the 30-min check-in message for RED alerts where checkin_sent=False
       and the alert was sent ≥ 30 minutes ago.
    2. Escalate (send emergency services message) for RED alerts where
       checkin_sent=True, no response, and sent_at ≥ 90 minutes ago.
    """
    logger.debug("job_check_checkins: starting")
    now = datetime.now(timezone.utc)

    with Session(engine) as session:
        _send_pending_checkins(now, session)
        _escalate_unanswered_checkins(now, session)


def _send_pending_checkins(now: datetime, session: Session) -> None:
    """Send the are-you-safe message for RED alerts older than 30 min."""
    checkin_threshold = now - timedelta(minutes=CHECKIN_DELAY_MINUTES)

    pending = session.exec(
        select(AlertLog)
        .where(AlertLog.tier == "RED")
        .where(AlertLog.checkin_sent == False)
        .where(AlertLog.sent_at <= checkin_threshold)
    ).all()

    for alert_log in pending:
        user = session.get(User, alert_log.user_id)
        if not user:
            continue

        sent = wa.send_checkin_prompt(user.phone)
        if sent:
            alert_log.checkin_sent    = True
            alert_log.checkin_sent_at = now
            session.add(alert_log)
            logger.info("Check-in prompt sent to %s (alert_log_id=%d)", user.phone, alert_log.id)

    session.commit()


def _escalate_unanswered_checkins(now: datetime, session: Session) -> None:
    """Escalate RED alerts with no check-in response after 90 minutes total."""
    escalate_threshold = now - timedelta(minutes=CHECKIN_ESCALATE_MINUTES)

    unanswered = session.exec(
        select(AlertLog)
        .where(AlertLog.tier == "RED")
        .where(AlertLog.checkin_sent == True)
        .where(AlertLog.checkin_response == None)
        .where(AlertLog.escalated == False)
        .where(AlertLog.sent_at <= escalate_threshold)
    ).all()

    for alert_log in unanswered:
        user = session.get(User, alert_log.user_id)
        if not user:
            continue

        sent = wa.send_checkin_escalation(user.phone)
        if sent:
            alert_log.escalated = True
            session.add(alert_log)
            logger.info(
                "Check-in escalated for %s (alert_log_id=%d)", user.phone, alert_log.id
            )

    session.commit()


# ---------------------------------------------------------------------------
# Job 4 — Weekly digest
# ---------------------------------------------------------------------------

def job_weekly_digest() -> None:
    """
    Run every Sunday, every hour UTC.
    Send the digest to users whose local time is currently 9am (±30 min).
    """
    logger.debug("job_weekly_digest: starting")
    now_utc = datetime.now(timezone.utc)

    with Session(engine) as session:
        active_users = session.exec(select(User).where(User.active == True)).all()

    for user in active_users:
        try:
            local_hour = (now_utc.hour + user.timezone_offset) % 24
            if local_hour != 9:
                continue

            _send_digest_to_user(user)
        except Exception as exc:
            logger.exception("job_weekly_digest: error for user %s: %s", user.phone, exc)


def _send_digest_to_user(user: User) -> None:
    events = fetch_events_for_digest(
        user.latitude, user.longitude, user.radius_km, user.near_fault
    )
    sent = wa.send_weekly_digest(user.phone, events, user.radius_km)
    if sent:
        logger.info("Weekly digest sent to %s (%d events)", user.phone, len(events))

    # Send updated forecast with the weekly digest
    try:
        from forecast import generate_forecast
        forecast = generate_forecast(
            user.latitude, user.longitude, user.radius_km, user.near_fault
        )
        if forecast:
            wa.send_message(user.phone, forecast.whatsapp_message)
    except Exception as exc:
        logger.warning("Weekly forecast failed for %s: %s", user.phone, exc)


# ---------------------------------------------------------------------------
# Job 5 — Render keepalive
# ---------------------------------------------------------------------------

def job_keepalive() -> None:
    """
    Ping /health every 14 minutes to prevent Render free tier from sleeping.
    Render spins down after 15 minutes of inactivity.
    """
    try:
        import httpx
        with httpx.Client(timeout=10) as client:
            resp = client.get(KEEPALIVE_URL)
            logger.debug("Keepalive ping: %s", resp.status_code)
    except Exception as exc:
        logger.debug("Keepalive ping failed (non-critical): %s", exc)


# ---------------------------------------------------------------------------
# Quiet hours helper
# ---------------------------------------------------------------------------

def _is_quiet_hours(user: User) -> bool:
    """
    Return True if the user's local time falls between QUIET_HOUR_START (22)
    and QUIET_HOUR_END (7) inclusive.
    """
    now_utc    = datetime.now(timezone.utc)
    local_hour = (now_utc.hour + user.timezone_offset) % 24

    if QUIET_HOUR_START <= QUIET_HOUR_END:
        # Same-day range (unusual — would be e.g. 2am–6am)
        return QUIET_HOUR_START <= local_hour < QUIET_HOUR_END
    else:
        # Overnight range: 22 → 7 wraps midnight
        return local_hour >= QUIET_HOUR_START or local_hour < QUIET_HOUR_END


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _offset_now(minutes: int = 0) -> datetime:
    """Return a UTC datetime offset by `minutes` from now — used for job staggering."""
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)