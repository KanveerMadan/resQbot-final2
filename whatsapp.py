"""
whatsapp.py — Meta WhatsApp Cloud API messaging layer for ResQbot.

Responsibilities:
  - Send any text message to a WhatsApp phone number
  - Build every user-facing message string (alerts, onboarding, commands, digest)
  - Mark incoming messages as read (removes the clock icon on user's screen)
  - Never raise — log errors and return False so callers can handle gracefully

All message content lives here so copy changes never touch business logic.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from prediction import PredictionResult, TIER_RED, TIER_ORANGE, TIER_YELLOW, TIER_GREEN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID   = os.getenv("PHONE_NUMBER_ID", "")
API_VERSION       = "v19.0"
API_BASE          = f"https://graph.facebook.com/{API_VERSION}/{PHONE_NUMBER_ID}/messages"

_TIMEOUT_SECONDS  = 15


# ---------------------------------------------------------------------------
# Low-level send primitives
# ---------------------------------------------------------------------------

def send_message(to: str, text: str) -> bool:
    """
    Send a plain-text WhatsApp message to `to` (E.164 format, e.g. +919876543210).
    Returns True on success, False on any failure.
    """
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        logger.error("WHATSAPP_TOKEN or PHONE_NUMBER_ID not set — cannot send message")
        return False

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.post(API_BASE, json=payload, headers=headers)
            response.raise_for_status()
            logger.info("Message sent to %s (%.40s…)", to, text.replace("\n", " "))
            return True
    except httpx.HTTPStatusError as exc:
        logger.error(
            "WhatsApp API error %s sending to %s: %s",
            exc.response.status_code, to, exc.response.text,
        )
    except httpx.TimeoutException:
        logger.warning("WhatsApp API timeout sending to %s", to)
    except Exception as exc:
        logger.exception("Unexpected error sending WhatsApp message to %s: %s", to, exc)
    return False


def mark_as_read(message_id: str) -> bool:
    """
    Mark an incoming message as read so the user sees double blue ticks.
    Returns True on success, False on any failure.
    """
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        return False

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.post(API_BASE, json=payload, headers=headers)
            response.raise_for_status()
            return True
    except Exception as exc:
        logger.warning("Failed to mark message %s as read: %s", message_id, exc)
        return False


# ---------------------------------------------------------------------------
# Onboarding messages
# ---------------------------------------------------------------------------

def send_welcome(to: str) -> bool:
    """Sent on first contact or unrecognised opener before location is shared."""
    text = (
        "👋 Welcome to *ResQbot* — your personal earthquake alert system.\n\n"
        "To begin monitoring, share your location using WhatsApp's location "
        "feature:\n"
        "📎 Tap the paperclip → *Location* → *Send Your Current Location*\n\n"
        "ResQbot will watch for seismic activity near you and alert you if "
        "anything significant is detected."
    )
    return send_message(to, text)


def send_radius_prompt(to: str, place_name: str, near_fault: bool) -> bool:
    """
    Sent after a location pin is received. Asks the user to pick a watch radius.
    `place_name` is a human-readable string derived from reverse-geocoding or
    the USGS place field; falls back to "your area" if unavailable.
    """
    fault_note = (
        "\n\n⚠️ *Note:* You are within 200 km of a tectonic plate boundary — "
        "elevated baseline risk zone. I'll apply a lower alert threshold for you."
        if near_fault else ""
    )

    text = (
        f"📍 Got it! Monitoring seismic activity near *{place_name}*.{fault_note}\n\n"
        "How wide a radius should I watch?\n\n"
        "Reply with a number:\n"
        "1️⃣  *100 km* — Local only\n"
        "2️⃣  *300 km* — Regional (recommended)\n"
        "3️⃣  *500 km* — Wide area"
    )
    return send_message(to, text)


def send_onboarding_complete(
    to: str,
    radius_km: int,
    historical: Optional[dict] = None,
) -> bool:
    """
    Sent after the user picks a radius. Includes the 5-year historical summary
    if available, then the command reference.
    """
    history_block = _build_history_block(historical)

    text = (
        f"✅ All set! I'll alert you about earthquakes within *{radius_km} km* "
        f"of your location.\n"
        f"{history_block}\n"
        "────────────────────\n"
        "📋 *Commands you can use anytime:*\n"
        "• *STOP* — pause all alerts\n"
        "• *START* — resume alerts\n"
        "• *UPDATE LOCATION* — change your location\n"
        "• *HISTORY* — recent events near you\n"
        "• *HELP* — show this list\n\n"
        "You can also ask me anything — e.g. _\"What does a 5.5 earthquake feel like?\"_"
    )
    return send_message(to, text)


def send_invalid_radius(to: str) -> bool:
    """Sent when the user replies something other than 1/2/3 during onboarding."""
    text = (
        "Please reply with *1*, *2*, or *3* to set your alert radius:\n\n"
        "1️⃣  100 km\n"
        "2️⃣  300 km\n"
        "3️⃣  500 km"
    )
    return send_message(to, text)


# ---------------------------------------------------------------------------
# Alert messages
# ---------------------------------------------------------------------------

_TIER_META = {
    TIER_GREEN:  ("🟢", "GREEN WATCH"),
    TIER_YELLOW: ("🟡", "YELLOW ADVISORY"),
    TIER_ORANGE: ("🟠", "ORANGE WARNING"),
    TIER_RED:    ("🔴", "RED EMERGENCY"),
}

_TIER_BODY = {
    TIER_GREEN: (
        "Low-level activity detected. No immediate action needed — "
        "stay aware of your surroundings."
    ),
    TIER_YELLOW: (
        "Moderate seismic activity detected. Low immediate risk — "
        "stay informed and avoid damaged structures."
    ),
    TIER_ORANGE: (
        "Significant seismic activity detected. *Be prepared to take "
        "cover.* Move away from windows and heavy objects."
    ),
    TIER_RED: (
        "⚠️ *MAJOR EARTHQUAKE DETECTED.* Drop, cover, and hold on. "
        "After shaking stops, evacuate if structurally unsafe. "
        "Expect aftershocks."
    ),
}


def send_alert(to: str, result: PredictionResult) -> bool:
    """
    Send a tiered alert message for a PredictionResult.
    Called by scheduler.py after deduplication and quiet-hours checks.
    """
    emoji, label = _TIER_META.get(result.tier, ("⚪", result.tier))
    body = _TIER_BODY.get(result.tier, "")
    event_time_str = _fmt_event_time(result.event_time)
    confidence_str = f"{result.confidence}/2 models" if result.confidence else "physical parameters"

    text = (
        f"{emoji} *{label}*\n"
        f"────────────────────\n"
        f"📌 {result.place}\n"
        f"📏 *{result.distance_km} km* from your location\n"
        f"💥 Magnitude *{result.mag:.1f}* | Depth *{result.depth:.0f} km*\n"
        f"🕐 {event_time_str}\n"
        f"🤖 Confidence: {confidence_str}\n"
        f"────────────────────\n"
        f"{body}"
    )
    return send_message(to, text)


def send_cluster_alert(to: str, event_count: int, epicentre_place: str, radius_km: float) -> bool:
    """
    Sent when 3+ aftershocks are detected within 50 km in 2 hours after M5.0+.
    """
    text = (
        f"🟠 *AFTERSHOCK CLUSTER DETECTED*\n"
        f"────────────────────\n"
        f"📌 Near {epicentre_place}\n"
        f"⚡ *{event_count} events* within {radius_km:.0f} km in the last 2 hours\n"
        f"────────────────────\n"
        "Ongoing seismic sequence in progress. Avoid unstable structures. "
        "Aftershocks can sometimes exceed the mainshock — remain alert."
    )
    return send_message(to, text)


# ---------------------------------------------------------------------------
# Check-in messages (are-you-safe flow for RED alerts)
# ---------------------------------------------------------------------------

def send_checkin_prompt(to: str) -> bool:
    """Sent 30 minutes after a RED alert."""
    text = (
        "🔴 *Are you safe?*\n\n"
        "A major earthquake was detected near you 30 minutes ago.\n\n"
        "Please reply:\n"
        "✅ *SAFE* — I'm okay\n"
        "🆘 *HELP* — I need assistance"
    )
    return send_message(to, text)


def send_checkin_acknowledged_safe(to: str) -> bool:
    text = (
        "✅ Glad you're safe. Stay away from damaged structures and be "
        "alert for aftershocks. I'll continue monitoring your area."
    )
    return send_message(to, text)


def send_checkin_escalation(to: str) -> bool:
    """
    Sent 60 minutes after the check-in prompt if no reply received.
    Directs the user to emergency services — ResQbot cannot call anyone.
    """
    text = (
        "🆘 *No response received.*\n\n"
        "If you are in an emergency, please contact your local emergency "
        "services immediately:\n"
        "🇮🇳 India: *112*\n"
        "🌍 International emergency: *112* or local equivalent\n\n"
        "ResQbot will continue monitoring your area."
    )
    return send_message(to, text)


def send_help_response(to: str) -> bool:
    """Sent when user replies HELP to a check-in or sends HELP as a command."""
    text = (
        "🆘 *Emergency services:*\n"
        "🇮🇳 India: *112*\n"
        "🌍 International: *112* or local equivalent\n\n"
        "📋 *ResQbot commands:*\n"
        "• *STOP* — pause alerts\n"
        "• *START* — resume alerts\n"
        "• *UPDATE LOCATION* — change your pin\n"
        "• *HISTORY* — recent events near you\n"
        "• *HELP* — show this list\n\n"
        "Ask me anything about earthquakes and I'll do my best to answer."
    )
    return send_message(to, text)


# ---------------------------------------------------------------------------
# Command response messages
# ---------------------------------------------------------------------------

def send_stop_confirmation(to: str) -> bool:
    text = (
        "🔕 Alerts paused. You won't receive any earthquake notifications.\n\n"
        "Reply *START* to resume monitoring at any time."
    )
    return send_message(to, text)


def send_start_confirmation(to: str) -> bool:
    text = (
        "🔔 Monitoring resumed. You'll receive alerts for seismic activity "
        "near your saved location.\n\n"
        "Reply *STOP* to pause again."
    )
    return send_message(to, text)


def send_already_active(to: str) -> bool:
    text = "✅ You're already active — ResQbot is monitoring your area."
    return send_message(to, text)


def send_already_stopped(to: str) -> bool:
    text = "🔕 Alerts are already paused. Reply *START* to resume."
    return send_message(to, text)


def send_update_location_prompt(to: str) -> bool:
    text = (
        "📍 To update your location, share a new pin using WhatsApp's "
        "location feature:\n"
        "📎 Tap the paperclip → *Location* → *Send Your Current Location*"
    )
    return send_message(to, text)


def send_no_location_yet(to: str) -> bool:
    """Sent when a command arrives before the user has shared a location."""
    text = (
        "I don't have your location yet. Please share it first:\n"
        "📎 Tap the paperclip → *Location* → *Send Your Current Location*"
    )
    return send_message(to, text)


# ---------------------------------------------------------------------------
# History command
# ---------------------------------------------------------------------------

def send_history(to: str, events: list[dict]) -> bool:
    """
    Send the HISTORY command response — up to 5 recent events near the user.
    `events` is a list of normalised event dicts from usgs.fetch_recent_events
    or a cached EventLog query in scheduler.py.
    """
    if not events:
        text = (
            "📭 No significant seismic events detected near your location "
            "in the last 7 days."
        )
        return send_message(to, text)

    lines = ["📋 *Recent seismic events near you:*\n"]
    for i, ev in enumerate(events[:5], 1):
        mag        = ev.get("mag", "?")
        depth      = ev.get("depth", "?")
        place      = ev.get("place", "Unknown")
        dist       = ev.get("distance_km", "?")
        event_time = ev.get("event_time")
        time_str   = _fmt_event_time(event_time)
        lines.append(
            f"{i}. M{mag:.1f} | {place}\n"
            f"   📏 {dist} km away | ⬇️ {depth:.0f} km deep | 🕐 {time_str}"
        )

    lines.append("\nReply *HELP* for commands or ask me anything about earthquakes.")
    return send_message(to, "\n".join(lines))


# ---------------------------------------------------------------------------
# Weekly digest
# ---------------------------------------------------------------------------

def send_weekly_digest(to: str, events: list[dict], radius_km: int) -> bool:
    """
    Sunday 9am local — summary of the past 7 days.
    `events` is the full list from usgs.fetch_events_for_digest; this function
    builds the summary stats itself.
    """
    count = len(events)

    if count == 0:
        text = (
            f"📅 *Weekly Seismic Digest*\n"
            f"────────────────────\n"
            f"No significant events within {radius_km} km this week. "
            f"All quiet — have a safe week ahead."
        )
        return send_message(to, text)

    # Find the strongest event
    strongest = max(events, key=lambda e: e.get("mag", 0.0))
    s_mag     = strongest.get("mag", 0.0)
    s_place   = strongest.get("place", "Unknown location")
    s_time    = _fmt_event_time(strongest.get("event_time"))

    text = (
        f"📅 *Weekly Seismic Digest*\n"
        f"────────────────────\n"
        f"📍 Radius: {radius_km} km | Past 7 days\n\n"
        f"📊 *{count}* event{'s' if count != 1 else ''} detected\n"
        f"💥 Strongest: *M{s_mag:.1f}* — {s_place}\n"
        f"🕐 {s_time}\n"
        f"────────────────────\n"
        f"{'⚠️ Stay informed — seismic activity was elevated this week.' if s_mag >= 5.0 else '✅ No major activity this week — all clear.'}\n\n"
        f"Reply *HISTORY* for individual events or ask me anything."
    )
    return send_message(to, text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_event_time(event_time) -> str:
    """Format a UTC datetime for display in alert messages."""
    if event_time is None:
        return "Time unknown"
    try:
        if isinstance(event_time, (int, float)):
            event_time = datetime.fromtimestamp(event_time / 1000, tz=timezone.utc)
        return event_time.strftime("%-d %b %Y, %H:%M UTC")
    except Exception:
        return "Time unknown"


def _build_history_block(historical: Optional[dict]) -> str:
    """
    Build the historical summary paragraph for the onboarding complete message.
    Returns an empty string if no data is available.
    """
    if not historical or historical.get("total_events", 0) == 0:
        return (
            "\n📭 No significant seismic events recorded near you in the "
            "past 5 years.\n\n"
        )

    total     = historical["total_events"]
    max_mag   = historical.get("max_mag")
    max_date  = historical.get("max_mag_date", "unknown date")
    last_mag  = historical.get("last_event_mag")
    last_date = historical.get("last_event_date", "recently")
    last_dist = historical.get("last_event_distance_km")

    dist_str = f" ({last_dist} km away)" if last_dist is not None else ""

    lines = [
        f"\n📊 *5-year seismic history near you:*",
        f"• {total} event{'s' if total != 1 else ''} recorded (M4.0+)",
    ]
    if max_mag is not None:
        lines.append(f"• Strongest: *M{max_mag:.1f}* on {max_date}")
    if last_mag is not None:
        lines.append(f"• Most recent: M{last_mag:.1f} on {last_date}{dist_str}")
    lines.append("")   # blank line before command reference

    return "\n".join(lines) + "\n"