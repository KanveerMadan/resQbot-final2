"""
webhook.py — Meta WhatsApp Cloud API webhook router for ResQbot.

Responsibilities:
  - Handle GET /webhook  — one-time Meta verification handshake
  - Handle POST /webhook — inbound message routing
  - Parse Meta's nested webhook payload reliably
  - Route by message type: location pin, known command, free text
  - Manage onboarding state machine (awaiting_location → awaiting_radius → active)
  - Deduplicate replayed webhooks via message_id
  - Never raise a 5xx — always return 200 OK to Meta (otherwise Meta retries
    aggressively and floods the queue)

All business logic lives in the imported modules; this file is pure routing.
"""

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from sqlmodel import Session, select

from models import AlertLog, User, engine
from prediction import PredictionResult
from usgs import fetch_historical_summary, haversine_km
import whatsapp as wa

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VERIFY_TOKEN    = os.getenv("VERIFY_TOKEN", "")
APP_SECRET      = os.getenv("APP_SECRET", "")   # optional payload signature check

# Radius map — user replies "1" / "2" / "3"
RADIUS_MAP = {"1": 100, "2": 300, "3": 500}

# Tectonic plate boundary GeoJSON path (checked at registration)
PLATE_BOUNDARY_GEOJSON = os.getenv("PLATE_BOUNDARY_GEOJSON", "plate_boundaries.geojson")
FAULT_ZONE_RADIUS_KM   = 200.0

# Seen message IDs — in-memory dedup for the current process lifetime.
# On Render free tier a single worker handles all traffic; this is sufficient.
# For multi-worker deployments, move to Redis or a DB column.
_seen_message_ids: set[str] = set()
_MAX_SEEN = 10_000   # cap memory growth


# ---------------------------------------------------------------------------
# GET /webhook — Meta verification handshake
# ---------------------------------------------------------------------------

@router.get("/webhook")
def verify_webhook(
    hub_mode:       str = Query(default="", alias="hub.mode"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
    hub_challenge:  str = Query(default="", alias="hub.challenge"),
):
    """
    Meta calls this once when you register the webhook URL in the developer
    portal.  Must echo hub.challenge back with 200 or Meta rejects the URL.
    """
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return Response(content=hub_challenge, media_type="text/plain")

    logger.warning(
        "Webhook verification failed — mode=%s token_match=%s",
        hub_mode, hub_verify_token == VERIFY_TOKEN,
    )
    raise HTTPException(status_code=403, detail="Verification failed")


# ---------------------------------------------------------------------------
# POST /webhook — inbound message handler
# ---------------------------------------------------------------------------

@router.post("/webhook")
async def receive_webhook(request: Request) -> dict:
    """
    Entry point for all inbound WhatsApp messages.
    Always returns {"status": "ok"} with 200 — Meta requires this.
    """
    # Optional: verify X-Hub-Signature-256 if APP_SECRET is configured
    if APP_SECRET:
        await _verify_signature(request)

    try:
        body = await request.json()
    except Exception:
        logger.warning("Received non-JSON webhook payload")
        return {"status": "ok"}

    # Meta wraps everything in a deeply nested structure; extract messages safely
    messages = _extract_messages(body)
    if not messages:
        # Status updates (delivered, read receipts) arrive here — ignore silently
        return {"status": "ok"}

    for message, phone in messages:
        try:
            await _route_message(message, phone)
        except Exception as exc:
            logger.exception("Unhandled error routing message from %s: %s", phone, exc)
        # Always continue — one bad message must not block the rest

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Message router
# ---------------------------------------------------------------------------

async def _route_message(message: dict, phone: str) -> None:
    """Dispatch a single inbound message to the right handler."""
    message_id  = message.get("id", "")
    msg_type    = message.get("type", "")

    # Deduplication — Meta occasionally replays webhooks
    if _is_duplicate(message_id):
        logger.debug("Duplicate message_id %s — skipping", message_id)
        return

    # Mark as read immediately (gives user double blue ticks)
    if message_id:
        wa.mark_as_read(message_id)

    with Session(engine) as session:
        user = _get_user(session, phone)

        if msg_type == "location":
            await _handle_location(message, phone, user, session)

        elif msg_type == "text":
            text = message.get("text", {}).get("body", "").strip()
            await _handle_text(text, phone, user, session)

        else:
            # Stickers, images, audio, etc. — nudge the user gently
            if user and user.onboarding_state == "active":
                wa.send_message(
                    phone,
                    "I can only process text and location messages. "
                    "Reply *HELP* to see available commands.",
                )
            elif not user:
                wa.send_welcome(phone)


# ---------------------------------------------------------------------------
# Location handler
# ---------------------------------------------------------------------------

async def _handle_location(
    message: dict,
    phone: str,
    user: Optional[User],
    session: Session,
) -> None:
    """Process an inbound location pin — works for both new and existing users."""
    loc = message.get("location", {})
    lat = loc.get("latitude")
    lon = loc.get("longitude")

    if lat is None or lon is None:
        logger.warning("Location message from %s missing coordinates", phone)
        wa.send_message(phone, "I couldn't read that location. Please try sharing it again.")
        return

    lat, lon = float(lat), float(lon)

    # Tectonic proximity check
    near_fault = _check_near_fault(lat, lon)

    # Timezone offset from longitude (rough but free — no API needed)
    tz_offset = _longitude_to_tz_offset(lon)

    # Reverse-geocode place name — use USGS place from a quick query; fall back gracefully
    place_name = _rough_place_name(lat, lon)

    if user is None:
        # New user — create record
        user = User(
            phone=phone,
            latitude=lat,
            longitude=lon,
            near_fault=near_fault,
            timezone_offset=tz_offset,
            onboarding_state="awaiting_radius",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        logger.info("New user registered: %s near_fault=%s", phone, near_fault)
    else:
        # Existing user updating their location
        user.latitude        = lat
        user.longitude       = lon
        user.near_fault      = near_fault
        user.timezone_offset = tz_offset
        user.onboarding_state = "awaiting_radius"
        user.updated_at      = datetime.now(timezone.utc)
        session.add(user)
        session.commit()
        logger.info("User %s updated location to %.4f, %.4f", phone, lat, lon)

    wa.send_radius_prompt(phone, place_name, near_fault)


# ---------------------------------------------------------------------------
# Text handler
# ---------------------------------------------------------------------------

async def _handle_text(
    text: str,
    phone: str,
    user: Optional[User],
    session: Session,
) -> None:
    """Route a text message: onboarding radius reply, command, or free-text Q&A."""
    upper = text.upper().strip()

    # ── No user record yet ──────────────────────────────────────────────────
    if user is None:
        wa.send_welcome(phone)
        return

    # ── Onboarding: awaiting radius pick ────────────────────────────────────
    if user.onboarding_state == "awaiting_radius":
        if text.strip() in RADIUS_MAP:
            radius_km = RADIUS_MAP[text.strip()]
            user.radius_km         = radius_km
            user.onboarding_state  = "active"
            user.updated_at        = datetime.now(timezone.utc)
            session.add(user)
            session.commit()
            logger.info("User %s set radius=%d km", phone, radius_km)

            # Fetch 5-year historical summary (may take a moment — acceptable in webhook)
            historical = fetch_historical_summary(
                user.latitude, user.longitude, radius_km
            )
            wa.send_onboarding_complete(phone, radius_km, historical)

            # Send probabilistic forecast after onboarding
            try:
                from forecast import generate_forecast
                forecast = generate_forecast(
                    user.latitude, user.longitude, radius_km, user.near_fault
                )
                if forecast:
                    wa.send_message(phone, forecast.whatsapp_message)
            except Exception as exc:
                logger.warning("Forecast generation failed for %s: %s", phone, exc)

        else:
            wa.send_invalid_radius(phone)
        return

    # ── Awaiting location (shouldn't normally receive text here) ────────────
    if user.onboarding_state == "awaiting_location":
        wa.send_welcome(phone)
        return

    # ── Hard commands (always handled regardless of active/stopped state) ───
    if upper == "STOP":
        if not user.active:
            wa.send_already_stopped(phone)
        else:
            user.active     = False
            user.updated_at = datetime.now(timezone.utc)
            session.add(user)
            session.commit()
            wa.send_stop_confirmation(phone)
        return

    if upper == "START":
        if user.active:
            wa.send_already_active(phone)
        else:
            user.active     = True
            user.updated_at = datetime.now(timezone.utc)
            session.add(user)
            session.commit()
            wa.send_start_confirmation(phone)
        return

    if upper in ("UPDATE LOCATION", "UPDATELOCATION", "UPDATE"):
        wa.send_update_location_prompt(phone)
        return

    if upper == "HELP":
        wa.send_help_response(phone)
        return

    # ── Check-in responses (SAFE / HELP after RED alert) ────────────────────
    if upper == "SAFE":
        _handle_checkin_response(phone, "SAFE", session)
        wa.send_checkin_acknowledged_safe(phone)
        return

    # HELP is already caught above — also records check-in response
    # (handled there; the HELP message covers both check-in and command contexts)

    # ── HISTORY command ──────────────────────────────────────────────────────
    if upper == "HISTORY":
        from usgs import fetch_events_for_digest
        events = fetch_events_for_digest(user.latitude, user.longitude, user.radius_km, user.near_fault)
        wa.send_history(phone, events)
        return

    # ── Free-text → Claude Q&A ───────────────────────────────────────────────
    # Import here to keep startup fast (claude_qa loads the Anthropic client)
    from claude_qa import answer_question
    answer_question(phone, text, user)


# ---------------------------------------------------------------------------
# Check-in response recorder
# ---------------------------------------------------------------------------

def _handle_checkin_response(phone: str, response: str, session: Session) -> None:
    """
    Find the most recent unanswered RED alert check-in for this user and
    record the response. Scheduler.py uses this to cancel the 60-min escalation.
    """
    user = _get_user(session, phone)
    if not user:
        return

    stmt = (
        select(AlertLog)
        .where(AlertLog.user_id == user.id)
        .where(AlertLog.tier == "RED")
        .where(AlertLog.checkin_sent == True)
        .where(AlertLog.checkin_response == None)
        .order_by(AlertLog.sent_at.desc())
        .limit(1)
    )
    alert_log = session.exec(stmt).first()
    if alert_log:
        alert_log.checkin_response     = response
        alert_log.checkin_responded_at = datetime.now(timezone.utc)
        session.add(alert_log)
        session.commit()
        logger.info("Check-in response '%s' recorded for user %s", response, phone)


# ---------------------------------------------------------------------------
# Payload parsing helpers
# ---------------------------------------------------------------------------

def _extract_messages(body: dict) -> list[tuple[dict, str]]:
    """
    Walk Meta's nested webhook payload and return a flat list of
    (message_dict, phone_number) tuples.

    Meta payload structure:
    {
      "entry": [{
        "changes": [{
          "value": {
            "messages": [{ ...message... }],
            "contacts": [{ "wa_id": "919876543210" }]
          }
        }]
      }]
    }
    """
    results = []
    try:
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value    = change.get("value", {})
                messages = value.get("messages", [])
                contacts = value.get("contacts", [])

                # Build a wa_id → display phone map from contacts list
                contact_map = {}
                for c in contacts:
                    wa_id = c.get("wa_id", "")
                    # Normalise to E.164 with + prefix
                    phone = f"+{wa_id}" if not wa_id.startswith("+") else wa_id
                    contact_map[wa_id] = phone

                for msg in messages:
                    # Phone comes from the message's own "from" field
                    raw_from = msg.get("from", "")
                    phone = contact_map.get(raw_from) or (
                        f"+{raw_from}" if not raw_from.startswith("+") else raw_from
                    )
                    if phone:
                        results.append((msg, phone))
    except Exception as exc:
        logger.exception("Failed to parse webhook payload: %s", exc)
    return results


def _get_user(session: Session, phone: str) -> Optional[User]:
    """Fetch a User by phone number. Returns None if not found."""
    stmt = select(User).where(User.phone == phone)
    return session.exec(stmt).first()


def _is_duplicate(message_id: str) -> bool:
    """Return True if we've already processed this message_id."""
    if not message_id:
        return False
    if message_id in _seen_message_ids:
        return True
    # Trim set if it grows too large
    if len(_seen_message_ids) >= _MAX_SEEN:
        # Remove ~10% oldest entries (sets are unordered; just discard arbitrary slice)
        to_remove = list(_seen_message_ids)[:_MAX_SEEN // 10]
        for mid in to_remove:
            _seen_message_ids.discard(mid)
    _seen_message_ids.add(message_id)
    return False


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

async def _verify_signature(request: Request) -> None:
    """
    Verify X-Hub-Signature-256 header using APP_SECRET.
    Raises HTTP 403 if signature is missing or invalid.
    Only called when APP_SECRET is configured.
    """
    sig_header = request.headers.get("X-Hub-Signature-256", "")
    if not sig_header.startswith("sha256="):
        logger.warning("Missing or malformed X-Hub-Signature-256")
        raise HTTPException(status_code=403, detail="Missing signature")

    body_bytes = await request.body()
    expected   = hmac.new(
        APP_SECRET.encode(), body_bytes, hashlib.sha256
    ).hexdigest()
    received   = sig_header[len("sha256="):]

    if not hmac.compare_digest(expected, received):
        logger.warning("Webhook signature mismatch — possible spoofed request")
        raise HTTPException(status_code=403, detail="Invalid signature")


# ---------------------------------------------------------------------------
# Geospatial helpers
# ---------------------------------------------------------------------------

def _check_near_fault(lat: float, lon: float) -> bool:
    """
    Return True if the given coordinates are within FAULT_ZONE_RADIUS_KM of
    any tectonic plate boundary in the local GeoJSON file.

    Falls back to False if the file is missing — tectonic tagging is a
    nice-to-have; it must never block registration.
    """
    try:
        import json
        with open(PLATE_BOUNDARY_GEOJSON, "r") as f:
            geojson = json.load(f)

        for feature in geojson.get("features", []):
            geometry = feature.get("geometry", {})
            geo_type = geometry.get("type", "")
            coords   = geometry.get("coordinates", [])

            if geo_type == "LineString":
                coord_list = [coords]
            elif geo_type == "MultiLineString":
                coord_list = coords
            else:
                continue

            for line in coord_list:
                for point in line:
                    if len(point) < 2:
                        continue
                    dist = haversine_km(lat, lon, point[1], point[0])
                    if dist <= FAULT_ZONE_RADIUS_KM:
                        return True
    except FileNotFoundError:
        logger.debug("Plate boundary GeoJSON not found — skipping fault check")
    except Exception as exc:
        logger.warning("Fault zone check failed: %s", exc)
    return False


def _longitude_to_tz_offset(lon: float) -> int:
    """
    Approximate UTC offset in whole hours from longitude.
    Accurate to ±1h for most locations; good enough for quiet-hours gating.
    For precise offsets, integrate a timezone API or use the tzwhere library.
    """
    return round(lon / 15)


def _rough_place_name(lat: float, lon: float) -> str:
    """
    Get a human-readable place name for the registration message.
    Queries USGS for the nearest recent event's place string as a quick
    reverse-geocode proxy.  Falls back to coordinate string if unavailable.
    """
    try:
        from usgs import fetch_recent_events
        # Use a small radius and low mag to find any nearby reference point
        events = fetch_recent_events(lat, lon, radius_km=500, near_fault=False, lookback_minutes=60 * 24 * 30)
        if events:
            place = events[0].get("place", "")
            # USGS place strings are like "12km NNE of Dehradun, India"
            # Extract the city/country part after "of " if present
            if " of " in place:
                return place.split(" of ", 1)[1]
            return place
    except Exception as exc:
        logger.debug("Could not fetch place name: %s", exc)

    # Final fallback — coordinate string
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{lat_dir}, {abs(lon):.2f}°{lon_dir}"