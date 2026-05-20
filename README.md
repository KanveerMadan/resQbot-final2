# ResQbot 🌍⚡

> **Real-time earthquake monitoring and alert system — delivered entirely over WhatsApp.**

ResQbot watches for seismic activity near you 24/7 and sends instant alerts when an earthquake is detected. No app to download, no website to visit — just WhatsApp.

---

## How It Works

1. **Share your location** via WhatsApp's native location feature
2. **ResQbot monitors** the USGS real-time seismic API every 10 minutes
3. **ML models classify** every detected event and assign an urgency tier
4. **You get alerted** instantly if anything significant is detected near you

---

## Features

### Core Alerts
- **4-tier urgency system** — GREEN / YELLOW / ORANGE / RED based on magnitude and depth
- **Configurable watch radius** — 100 km, 300 km, or 500 km
- **Quiet hours** — no alerts between 10pm–7am local time (RED always bypasses)
- **Aftershock cluster detection** — alerts when 3+ events occur within 50 km in 2 hours after an M5.0+
- **Are you safe? check-in** — 30 minutes after every RED alert, auto-escalates if no reply in 60 minutes

### Probabilistic Forecasting
- **Poisson probability model** — calculates probability of M4+ and M5+ events in the next 24h / 72h / 7 days
- **Foreshock swarm detection** — fires a cluster warning when recent seismicity is 3x above the 90-day baseline
- **5-year historical report** — sent at registration with event count, strongest magnitude, and last significant event

### Smart Features
- **Tectonic zone tagging** — detects if you're within 200 km of a plate boundary and lowers alert threshold to M3.5+
- **Natural language Q&A** — ask anything about earthquakes; powered by Groq (Llama 3.1)
- **Weekly Sunday digest** — past 7 days summary + updated probabilistic forecast
- **Admin dashboard** — password-protected live stats at `/admin`

### Commands
| Command | Action |
|---------|--------|
| `STOP` | Pause all alerts |
| `START` | Resume alerts |
| `UPDATE LOCATION` | Change your monitored location |
| `HISTORY` | Last 5 seismic events near you |
| `SAMPLE ALERT` | Preview all 4 alert tiers |
| `HELP` | Full command list + emergency numbers |

---

## Alert Format

```
🔴 RED EMERGENCY
────────────────────
📌 Chamoli, Uttarakhand, India
📏 298 km from your location
💥 Magnitude 6.8 | Depth 10 km
🕐 19 May 2026, 16:33 UTC
🤖 Confidence: 2/2 models
────────────────────
⚠️ MAJOR EARTHQUAKE DETECTED. Drop, cover, and hold on.
After shaking stops, evacuate if structurally unsafe.
Expect aftershocks.
```

---

## ML Model

ResQbot uses an ensemble of two scikit-learn models loaded from `combined_model.pkl`:

- **Logistic Regression** — `LogisticRegression(C=10, class_weight='balanced')`
- **Random Forest** — `RandomForestClassifier(max_depth=5, class_weight='balanced')`
- **2-model majority vote** — earthquake flagged if at least 1 of 2 models votes yes
- **Confidence score** — 0, 1, or 2 models agreed (shown in every alert)

### Input Features (5 features)
| Feature | Source |
|---------|--------|
| `latitude` | USGS GeoJSON coordinates |
| `longitude` | USGS GeoJSON coordinates |
| `depth` | USGS GeoJSON coordinates |
| `mag` | USGS properties |
| `gap` | USGS properties (azimuthal gap, defaults to 180 if null) |

### Urgency Classification
| Tier | Condition |
|------|-----------|
| 🟢 GREEN | M4.0 – M4.4 |
| 🟡 YELLOW | M4.5 – M5.4 |
| 🟠 ORANGE | M5.5+ or depth < 30 km with M5.0+ |
| 🔴 RED | M6.5+ and depth < 70 km |

---

## Probabilistic Forecasting

ResQbot uses the **Poisson process model** — the same methodology used by USGS, GNS Science (New Zealand), and JMA (Japan) for operational short-term earthquake forecasting.

```
P(at least 1 event in T days) = 1 - e^(-rate × T)

where rate = M4+ events per day in your region (90-day rolling average)
```

Adjusted for:
- **Fault proximity** — 1.4x baseline boost for users within 200 km of a plate boundary
- **Cluster multiplier** — up to 3x boost when recent seismicity exceeds baseline
- **Foreshock swarm signal** — cluster warning fires when 48h rate ≥ 3x the 90-day baseline

---

## Architecture

```
WhatsApp User
     │
     ▼
Meta WhatsApp Cloud API
     │
     ▼
FastAPI (Render free tier)
     │
     ├── POST /webhook ──► webhook.py (message router)
     │                          ├── location  ──► save to DB, tectonic check
     │                          ├── command   ──► STOP / START / HISTORY / etc.
     │                          └── free text ──► claude_qa.py (Groq LLM)
     │
     ├── GET  /health  ──► Render keepalive
     └── GET  /admin   ──► Admin dashboard
     │
     └── APScheduler (background)
               ├── Every 10 min ──► USGS poll → ML prediction → alert
               ├── Every 10 min ──► Aftershock cluster detection
               ├── Every  5 min ──► RED check-in escalation
               ├── Every Sunday ──► Weekly digest + forecast
               └── Every 14 min ──► Render keepalive ping
```

---

## Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| Backend | FastAPI + Uvicorn | Free |
| Database | Neon (serverless PostgreSQL) | Free |
| Background jobs | APScheduler | Free |
| ML inference | scikit-learn + joblib | Free |
| Seismic data | USGS Earthquake Hazards API | Free |
| WhatsApp | Meta WhatsApp Business Cloud API | Free (1000 conv/month) |
| AI Q&A | Groq API — Llama 3.1 8B | Free |
| Hosting | Render | Free |

**Total infrastructure cost: $0/month**

---

## Project Structure

```
ResQbot/
├── main.py              # FastAPI app entry point
├── models.py            # SQLModel database schema
├── prediction.py        # ML inference + urgency classifier
├── usgs.py              # USGS API client + event normaliser
├── whatsapp.py          # WhatsApp messaging layer
├── webhook.py           # Inbound message router + onboarding
├── scheduler.py         # APScheduler background jobs
├── claude_qa.py         # Groq LLM natural language Q&A
├── forecast.py          # Poisson probabilistic forecasting
├── dashboard.py         # Admin dashboard
├── combined_model.pkl   # Trained ML model bundle
├── render.yaml          # Render deployment config
├── requirements.txt     # Python dependencies
└── .python-version      # Python 3.11.9
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `WHATSAPP_TOKEN` | Meta WhatsApp permanent access token |
| `PHONE_NUMBER_ID` | Meta WhatsApp phone number ID |
| `VERIFY_TOKEN` | Webhook verification token |
| `GROQ_API_KEY` | Groq API key for LLM Q&A |
| `ADMIN_PASSWORD` | Password for /admin dashboard |
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `MODEL_PATH` | Path to combined_model.pkl |

---

## Local Development

```bash
# Clone the repo
git clone https://github.com/KanveerMadan/ResQbot.git
cd ResQbot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Fill in .env with your credentials

# Run locally
uvicorn main:app --reload --port 8000
```

For local WhatsApp webhook testing, use [ngrok](https://ngrok.com):
```bash
ngrok http 8000
# Set the ngrok URL as your Meta webhook callback URL
```

---

## Deployment

ResQbot is designed for **Render free tier** with zero configuration:

1. Fork this repo
2. Connect to [Render](https://render.com) — it auto-detects `render.yaml`
3. Set the environment variables in Render dashboard
4. Deploy — `render.yaml` handles everything else

---
## Docker Deployment

Run ResQbot locally with Docker:

```bash
docker compose build
docker compose up
```

Open:

http://localhost:8000

FastAPI Docs:

http://localhost:8000/docs

## Data Sources

- **Real-time seismic data** — [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/fdsnws/event/1/)
- **Tectonic plate boundaries** — [PB2002 Plate Boundary Dataset](https://github.com/fraxen/tectonicplates)

---

## Disclaimer

ResQbot alerts are based on data from USGS sensors and ML classification. Probabilistic forecasts are statistical estimates based on historical seismicity patterns — not deterministic predictions. Always follow official guidance from your national disaster management agency.

🇮🇳 India: [NDMA](https://ndma.gov.in) | Emergency: **112**

---

## Author

**Kanveer Madan**

Built with the goal of making earthquake awareness accessible to everyone — no app required.

---

*ResQbot is open source. If you find it useful, give it a ⭐*
