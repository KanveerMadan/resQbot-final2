"""
dashboard.py — Password-protected admin dashboard for ResQbot.

Routes:
  GET  /admin          — login page (redirects to /admin/dashboard if already authed)
  POST /admin/login    — validate password, set session cookie
  GET  /admin/dashboard — main stats page (requires auth)
  GET  /admin/logout   — clear session cookie
  GET  /admin/api/stats — JSON stats endpoint (requires auth, used by dashboard JS)

Auth: simple single-password session cookie. No user accounts.
The cookie is signed with ADMIN_SECRET (falls back to ADMIN_PASSWORD if not set).
Not a replacement for network-level access controls on production.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Cookie, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlmodel import Session, col, func, select

from models import AlertLog, EventLog, User, engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
# Use a separate secret for signing cookies; fall back to password-derived value
ADMIN_SECRET   = os.getenv("ADMIN_SECRET", hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest())
COOKIE_NAME    = "resqbot_admin"
COOKIE_TTL_H   = 8   # hours before re-login required


def _make_token(ts: int) -> str:
    """Create an HMAC-signed token embedding a Unix timestamp."""
    payload = str(ts)
    sig = hmac.new(ADMIN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_token(token: str) -> bool:
    """Verify token signature and expiry. Returns True if valid."""
    try:
        payload, sig = token.rsplit(".", 1)
        expected = hmac.new(ADMIN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return False
        ts = int(payload)
        age_hours = (time.time() - ts) / 3600
        return age_hours < COOKIE_TTL_H
    except Exception:
        return False


def _is_authed(session_token: Optional[str] = None) -> bool:
    return bool(session_token and _verify_token(session_token))


@router.get("/", response_class=HTMLResponse)
async def admin_root(resqbot_admin: Optional[str] = Cookie(default=None)):
    if _is_authed(resqbot_admin):
        return RedirectResponse("/admin/dashboard", status_code=302)
    return RedirectResponse("/admin/login", status_code=302)


@router.get("/login", response_class=HTMLResponse)
async def login_page(
    error: str = "",
    resqbot_admin: Optional[str] = Cookie(default=None),
):
    if _is_authed(resqbot_admin):
        return RedirectResponse("/admin/dashboard", status_code=302)
    return HTMLResponse(_render_login(error=bool(error)))


@router.post("/login")
async def do_login(
    response: Response,
    password: str = Form(...),
):
    if password == ADMIN_PASSWORD:
        token = _make_token(int(time.time()))
        resp  = RedirectResponse("/admin/dashboard", status_code=302)
        resp.set_cookie(
            COOKIE_NAME,
            token,
            httponly=True,
            samesite="lax",
            max_age=COOKIE_TTL_H * 3600,
        )
        logger.info("Admin login successful")
        return resp
    logger.warning("Admin login failed — wrong password")
    return RedirectResponse("/admin/login?error=1", status_code=302)


@router.get("/logout")
async def logout():
    resp = RedirectResponse("/admin/login", status_code=302)
    resp.delete_cookie(COOKIE_NAME)
    return resp


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(resqbot_admin: Optional[str] = Cookie(default=None)):
    if not _is_authed(resqbot_admin):
        return RedirectResponse("/admin/login", status_code=302)
    return HTMLResponse(_render_dashboard())


@router.get("/api/stats")
async def api_stats(resqbot_admin: Optional[str] = Cookie(default=None)):
    if not _is_authed(resqbot_admin):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse(_collect_stats())


def _collect_stats() -> dict:
    """Query the DB and return a dict of dashboard metrics."""
    now = datetime.now(timezone.utc)

    with Session(engine) as session:
        total_users     = session.exec(select(func.count(col(User.id)))).one()
        active_users    = session.exec(select(func.count(col(User.id))).where(User.active == True)).one()
        near_fault      = session.exec(select(func.count(col(User.id))).where(User.near_fault == True)).one()
        total_alerts    = session.exec(select(func.count(col(AlertLog.id)))).one()
        alerts_24h      = session.exec(
            select(func.count(col(AlertLog.id)))
            .where(AlertLog.sent_at >= now - timedelta(hours=24))
        ).one()
        total_events    = session.exec(select(func.count(col(EventLog.id)))).one()
        events_24h      = session.exec(
            select(func.count(col(EventLog.id)))
            .where(EventLog.processed_at >= now - timedelta(hours=24))
        ).one()

        # Tier breakdown (all time)
        tier_rows = session.exec(
            select(AlertLog.tier, func.count(col(AlertLog.id)))
            .group_by(AlertLog.tier)
        ).all()
        tier_counts = {tier: count for tier, count in tier_rows}

        # Check-in stats
        red_sent       = session.exec(
            select(func.count(col(AlertLog.id)))
            .where(AlertLog.tier == "RED")
        ).one()
        checkin_safe   = session.exec(
            select(func.count(col(AlertLog.id)))
            .where(AlertLog.checkin_response == "SAFE")
        ).one()
        checkin_help   = session.exec(
            select(func.count(col(AlertLog.id)))
            .where(AlertLog.checkin_response == "HELP")
        ).one()
        escalated      = session.exec(
            select(func.count(col(AlertLog.id)))
            .where(AlertLog.escalated == True)
        ).one()

        # Recent 10 alerts
        recent_alerts = session.exec(
            select(AlertLog)
            .order_by(AlertLog.sent_at.desc())
            .limit(10)
        ).all()

        # Recent 10 events processed
        recent_events = session.exec(
            select(EventLog)
            .order_by(EventLog.processed_at.desc())
            .limit(10)
        ).all()

        # Strongest event ever processed
        strongest = session.exec(
            select(EventLog).order_by(EventLog.mag.desc()).limit(1)
        ).first()

    return {
        "generated_at": now.strftime("%d %b %Y %H:%M UTC"),
        "users": {
            "total": total_users,
            "active": active_users,
            "stopped": total_users - active_users,
            "near_fault": near_fault,
        },
        "alerts": {
            "total": total_alerts,
            "last_24h": alerts_24h,
            "by_tier": {
                "GREEN":  tier_counts.get("GREEN", 0),
                "YELLOW": tier_counts.get("YELLOW", 0),
                "ORANGE": tier_counts.get("ORANGE", 0),
                "RED":    tier_counts.get("RED", 0),
            },
        },
        "checkins": {
            "red_total":   red_sent,
            "safe":        checkin_safe,
            "help":        checkin_help,
            "escalated":   escalated,
            "no_response": red_sent - checkin_safe - checkin_help,
        },
        "events": {
            "total_processed": total_events,
            "last_24h":        events_24h,
            "strongest_mag":   strongest.mag if strongest else None,
            "strongest_place": _get_event_place(strongest),
        },
        "recent_alerts": [
            {
                "tier":       a.tier,
                "mag":        a.mag,
                "depth":      a.depth,
                "dist":       a.distance_km,
                "sent_at":    a.sent_at.strftime("%d %b %H:%M") if a.sent_at else "",
                "checkin":    a.checkin_response or ("pending" if a.checkin_sent else "—"),
            }
            for a in recent_alerts
        ],
        "recent_events": [
            {
                "mag":          e.mag,
                "depth":        e.depth,
                "tier":         e.tier,
                "confidence":   e.confidence,
                "lr":           round(e.lr_score, 3),
                "rf":           round(e.rf_score, 3),
                "processed_at": e.processed_at.strftime("%d %b %H:%M") if e.processed_at else "",
            }
            for e in recent_events
        ],
    }


def _get_event_place(event: Optional[EventLog]) -> str:
    if not event:
        return "—"
    return f"M{event.mag:.1f} at {event.latitude:.2f}°, {event.longitude:.2f}°"


def _render_login(error: bool = False) -> str:
    error_html = (
        '<p class="error">Incorrect password. Try again.</p>' if error else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ResQbot — Admin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #1e1e2e;
    --accent: #e8ff47;
    --accent-dim: #b8cc30;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --red: #ff4757;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .login-wrap {{
    width: 100%;
    max-width: 380px;
    padding: 0 1.5rem;
  }}
  .wordmark {{
    font-family: var(--mono);
    font-size: 1.1rem;
    letter-spacing: 0.15em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }}
  .subtitle {{
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    margin-bottom: 2.5rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2rem;
  }}
  label {{
    display: block;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
    font-family: var(--mono);
  }}
  input[type=password] {{
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.95rem;
    padding: 0.65rem 0.85rem;
    outline: none;
    transition: border-color 0.15s;
  }}
  input[type=password]:focus {{ border-color: var(--accent); }}
  button {{
    margin-top: 1.2rem;
    width: 100%;
    background: var(--accent);
    color: #0a0a0f;
    font-family: var(--mono);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: none;
    border-radius: 3px;
    padding: 0.75rem;
    cursor: pointer;
    transition: background 0.15s;
  }}
  button:hover {{ background: var(--accent-dim); }}
  .error {{
    margin-top: 1rem;
    font-size: 0.8rem;
    color: var(--red);
    font-family: var(--mono);
  }}
</style>
</head>
<body>
<div class="login-wrap">
  <div class="wordmark">ResQbot</div>
  <div class="subtitle">Admin Console — Restricted Access</div>
  <div class="card">
    <form method="POST" action="/admin/login">
      <label for="pw">Password</label>
      <input type="password" id="pw" name="password" autofocus autocomplete="current-password">
      <button type="submit">Enter</button>
      {error_html}
    </form>
  </div>
</div>
</body>
</html>"""


def _render_dashboard() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ResQbot — Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #16161f;
    --border: #1e1e2e;
    --accent: #e8ff47;
    --text: #e8e8f0;
    --muted: #6b6b80;
    --green: #47ffb2;
    --yellow: #ffd447;
    --orange: #ff8c47;
    --red: #ff4757;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 15px; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
  }

  /* ── Top bar ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 10;
  }
  .wordmark {
    font-family: var(--mono);
    font-size: 0.95rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
  }
  .header-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }
  .generated-at {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.05em;
  }
  .btn-logout {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    text-decoration: none;
    border: 1px solid var(--border);
    padding: 0.35rem 0.75rem;
    border-radius: 3px;
    transition: color 0.15s, border-color 0.15s;
  }
  .btn-logout:hover { color: var(--text); border-color: var(--muted); }

  /* ── Layout ── */
  main { padding: 2rem; max-width: 1200px; margin: 0 auto; }

  .section-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    margin-top: 2.5rem;
  }
  .section-label:first-child { margin-top: 0; }

  /* ── Stat cards ── */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  .stat-card {
    background: var(--surface);
    padding: 1.25rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .stat-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .stat-value {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }
  .stat-value.accent { color: var(--accent); }
  .stat-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.1rem;
  }

  /* ── Tier bar ── */
  .tier-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  .tier-card {
    background: var(--surface);
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
  }
  .tier-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .tier-dot.GREEN  { background: var(--green); }
  .tier-dot.YELLOW { background: var(--yellow); }
  .tier-dot.ORANGE { background: var(--orange); }
  .tier-dot.RED    { background: var(--red); box-shadow: 0 0 8px var(--red); }
  .tier-name {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
  }
  .tier-count {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
  }
  .tier-count.GREEN  { color: var(--green);  }
  .tier-count.YELLOW { color: var(--yellow); }
  .tier-count.ORANGE { color: var(--orange); }
  .tier-count.RED    { color: var(--red);    }

  /* ── Tables ── */
  .table-wrap {
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }
  thead th {
    background: var(--surface);
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.65rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-weight: 400;
  }
  tbody tr { border-bottom: 1px solid var(--border); }
  tbody tr:last-child { border-bottom: none; }
  tbody tr:hover { background: var(--surface2); }
  td {
    padding: 0.6rem 1rem;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.72rem;
    white-space: nowrap;
  }
  td.muted { color: var(--muted); }
  .tier-badge {
    display: inline-block;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.45rem;
    border-radius: 2px;
    font-weight: 700;
  }
  .tier-badge.GREEN  { background: rgba(71,255,178,0.12); color: var(--green);  }
  .tier-badge.YELLOW { background: rgba(255,212,71,0.12);  color: var(--yellow); }
  .tier-badge.ORANGE { background: rgba(255,140,71,0.12);  color: var(--orange); }
  .tier-badge.RED    { background: rgba(255,71,87,0.15);   color: var(--red);    }
  .tier-badge.BELOW_THRESHOLD { background: rgba(107,107,128,0.15); color: var(--muted); }
  .conf-pip {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 2px;
    background: var(--border);
  }
  .conf-pip.on { background: var(--accent); }

  /* ── Loading / error state ── */
  .loading { color: var(--muted); font-family: var(--mono); font-size: 0.8rem; padding: 2rem 0; }
  .pulse { animation: pulse 1.4s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }

  /* ── Responsive ── */
  @media (max-width: 640px) {
    .tier-row { grid-template-columns: repeat(2,1fr); }
    .stat-grid { grid-template-columns: repeat(2,1fr); }
    main { padding: 1rem; }
    header { padding: 0.85rem 1rem; }
  }
</style>
</head>
<body>
<header>
  <div class="wordmark">ResQbot</div>
  <div class="header-right">
    <span class="generated-at" id="gen-at">Loading…</span>
    <a href="/admin/logout" class="btn-logout">Logout</a>
  </div>
</header>

<main>
  <div id="app"><p class="loading pulse">Fetching stats…</p></div>
</main>

<script>
const $ = id => document.getElementById(id);

function tierBadge(tier) {
  return `<span class="tier-badge ${tier}">${tier}</span>`;
}

function confPips(n) {
  return [0,1,2].map(i => `<span class="conf-pip ${i < n ? 'on' : ''}"></span>`).join('');
}

function render(d) {
  $('gen-at').textContent = d.generated_at;

  const u = d.users, a = d.alerts, e = d.events, c = d.checkins;

  const alertRows = d.recent_alerts.map(r => `
    <tr>
      <td>${tierBadge(r.tier)}</td>
      <td>M${r.mag.toFixed(1)}</td>
      <td class="muted">${r.depth.toFixed(0)} km</td>
      <td class="muted">${r.dist.toFixed(0)} km</td>
      <td class="muted">${r.sent_at}</td>
      <td class="muted">${r.checkin}</td>
    </tr>`).join('');

  const eventRows = d.recent_events.map(r => `
    <tr>
      <td>M${r.mag.toFixed(1)}</td>
      <td class="muted">${r.depth.toFixed(0)} km</td>
      <td>${tierBadge(r.tier)}</td>
      <td>${confPips(r.confidence)}</td>
      <td class="muted">${r.lr.toFixed(3)}</td>
      <td class="muted">${r.rf.toFixed(3)}</td>
      <td class="muted">${r.processed_at}</td>
    </tr>`).join('');

  $('app').innerHTML = `
    <div class="section-label">Subscribers</div>
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">Total</div>
        <div class="stat-value accent">${u.total}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Active</div>
        <div class="stat-value">${u.active}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Stopped</div>
        <div class="stat-value">${u.stopped}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Near Fault</div>
        <div class="stat-value">${u.near_fault}</div>
        <div class="stat-sub">Lower threshold</div>
      </div>
    </div>

    <div class="section-label">Alerts sent</div>
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">All time</div>
        <div class="stat-value">${a.total}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Last 24 h</div>
        <div class="stat-value accent">${a.last_24h}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">RED check-ins</div>
        <div class="stat-value">${c.red_total}</div>
        <div class="stat-sub">${c.safe} safe · ${c.help} help · ${c.escalated} escalated</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Events processed</div>
        <div class="stat-value">${e.total_processed}</div>
        <div class="stat-sub">${e.last_24h} in last 24 h</div>
      </div>
    </div>

    <div class="section-label">Alerts by tier</div>
    <div class="tier-row">
      ${['GREEN','YELLOW','ORANGE','RED'].map(t => `
        <div class="tier-card">
          <span class="tier-dot ${t}"></span>
          <span class="tier-name">${t}</span>
          <span class="tier-count ${t}">${a.by_tier[t]}</span>
        </div>`).join('')}
    </div>

    <div class="section-label">Recent alerts</div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Tier</th><th>Mag</th><th>Depth</th>
          <th>Distance</th><th>Sent</th><th>Check-in</th>
        </tr></thead>
        <tbody>${alertRows || '<tr><td colspan="6" class="muted">No alerts yet</td></tr>'}</tbody>
      </table>
    </div>

    <div class="section-label">Recent events processed</div>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Mag</th><th>Depth</th><th>Tier</th>
          <th>Confidence</th><th>LR</th><th>RF</th><th>Processed</th>
        </tr></thead>
        <tbody>${eventRows || '<tr><td colspan="7" class="muted">No events yet</td></tr>'}</tbody>
      </table>
    </div>
  `;
}

async function loadStats() {
  try {
    const res  = await fetch('/admin/api/stats');
    if (res.status === 401) { window.location = '/admin/login'; return; }
    const data = await res.json();
    render(data);
  } catch(err) {
    $('app').innerHTML = '<p class="loading">Failed to load stats. Try refreshing.</p>';
  }
}

loadStats();
// Auto-refresh every 60 seconds
setInterval(loadStats, 60_000);
</script>
</body>
</html>"""
