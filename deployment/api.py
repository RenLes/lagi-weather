"""
Lagi -- Fiji Weather Guardian: FastAPI Server
==============================================
Secured REST API with email verification, auth tokens, and CORS restrictions.

Run:
    uvicorn deployment.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests as http_requests
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add project root and scripts dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from config import (
    VALID_LOCATIONS, VALID_SEASONS, VALID_ENSO, API_VERSION, ALLOWED_ORIGINS,
    SIGNUP_SECRET, USERS_FILE, RESEND_API_KEY, RESEND_FROM_EMAIL,
    VERIFICATION_CODE_EXPIRY,
)
from .inference import LagiInference
from validation import log_actual, compute_correlation, generate_validation_report

app = FastAPI(
    title="Lagi - Fiji Weather Guardian",
    description="AI-corrected weather forecasts for Fiji with 3-source ensemble and dynamic validation. Bula!",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Lagi-Token"],
    allow_credentials=True,
)

_assets_dir = Path(__file__).parent / "assets"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


# ---------------------------------------------------------------------------
# Rate limiting middleware
# ---------------------------------------------------------------------------
class RateLimitMiddleware:
    """Simple in-memory rate limiter by client IP."""

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self.clients: dict[str, list[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        timestamps = self.clients.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < self.window]
        if len(timestamps) >= self.max_requests:
            self.clients[client_ip] = timestamps
            return False
        timestamps.append(now)
        self.clients[client_ip] = timestamps
        return True


_rate_limiter = RateLimitMiddleware(max_requests=30, window_seconds=60.0)

# Stricter per-IP rate limiters for auth endpoints (signup / verify)
_signup_limiter = RateLimitMiddleware(max_requests=5, window_seconds=300.0)   # 5 per 5 min
_verify_limiter = RateLimitMiddleware(max_requests=10, window_seconds=300.0)  # 10 per 5 min

# Track consecutive failed verify attempts per email to lock out brute-force
_verify_failures: dict[str, list[float]] = {}
_MAX_VERIFY_FAILURES = 5
_VERIFY_FAILURE_WINDOW = 300.0  # 5 minutes

def _check_verify_failures(email: str) -> bool:
    """Return True if email is within allowed failure threshold."""
    now = time.time()
    failures = _verify_failures.get(email, [])
    failures = [t for t in failures if now - t < _VERIFY_FAILURE_WINDOW]
    _verify_failures[email] = failures
    return len(failures) < _MAX_VERIFY_FAILURES

def _record_verify_failure(email: str) -> None:
    now = time.time()
    _verify_failures.setdefault(email, []).append(now)

def _clear_verify_failures(email: str) -> None:
    _verify_failures.pop(email, None)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Attach security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # Only set HSTS on HTTPS responses
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please slow down and try again in a moment."},
        )
    response = await call_next(request)
    return response


def _purge_expired_codes() -> None:
    """Remove expired pending verification codes to prevent memory growth."""
    now = time.time()
    expired = [email for email, (_, expiry) in _pending_codes.items() if now > expiry]
    for email in expired:
        del _pending_codes[email]


engine = LagiInference()


# ---------------------------------------------------------------------------
# Auth: Email verification via Resend
# ---------------------------------------------------------------------------
def _make_token(email: str) -> str:
    return hashlib.sha256(f"{email}:{SIGNUP_SECRET}".encode()).hexdigest()[:32]


def _validate_email(email: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email))


# In-memory stores
_users: set[str] = set()
_pending_codes: dict[str, tuple[str, float]] = {}  # email -> (code, expiry_timestamp)

if USERS_FILE.exists():
    with open(USERS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                _users.add(json.loads(line).get("email", ""))


def _send_verification_email(email: str, code: str) -> bool:
    """Send verification code via Resend API."""
    if not RESEND_API_KEY:
        # No Resend key — auto-verify for development
        return True

    try:
        resp = http_requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": RESEND_FROM_EMAIL,
                "to": [email],
                "subject": f"Lagi Weather Guardian — Your verification code: {code}",
                "html": (
                    f"<div style='font-family:sans-serif;max-width:400px;margin:0 auto;padding:20px'>"
                    f"<h2 style='color:#0077b6'>Bula! Welcome to Lagi</h2>"
                    f"<p>Your verification code is:</p>"
                    f"<div style='background:#f0f7ff;border-radius:12px;padding:20px;text-align:center;"
                    f"font-size:32px;font-weight:700;color:#0077b6;letter-spacing:8px;margin:16px 0'>{code}</div>"
                    f"<p>This code expires in 10 minutes.</p>"
                    f"<p style='color:#888;font-size:0.9em'>Lagi — Fiji Weather Guardian | renles.com</p>"
                    f"</div>"
                ),
            },
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


async def verify_token(request: Request) -> str:
    """Dependency: verify auth token from header. Returns email."""
    token = request.headers.get("X-Lagi-Token", "")
    if not token:
        raise HTTPException(401, "Authentication required. Please sign up first.")

    # Check token against all known users
    for email in _users:
        if _make_token(email) == token:
            return email

    raise HTTPException(401, "Invalid token. Please sign up again.")


# ---------------------------------------------------------------------------
# Public endpoints (no auth required)
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    email: str = Field(..., description="User email address")


class VerifyRequest(BaseModel):
    email: str = Field(..., description="Email used during signup")
    code: str = Field(..., description="6-digit verification code")


@app.post("/signup")
async def signup(req: SignupRequest, request: Request):
    """Send a verification code to the provided email."""
    client_ip = request.client.host if request.client else "unknown"
    if not _signup_limiter.is_allowed(client_ip):
        raise HTTPException(429, "Too many signup attempts. Please wait a few minutes.")

    _purge_expired_codes()

    email = req.email.strip().lower()
    if not _validate_email(email):
        raise HTTPException(400, "Please enter a valid email address.")

    # If already verified, just return token
    if email in _users:
        return {"message": "Email already verified!", "token": _make_token(email), "verified": True}

    # Generate 6-digit code
    code = f"{random.randint(100000, 999999)}"
    expiry = time.time() + VERIFICATION_CODE_EXPIRY
    _pending_codes[email] = (code, expiry)

    # Send via Resend
    sent = _send_verification_email(email, code)

    if not sent and RESEND_API_KEY:
        raise HTTPException(503, "Could not send verification email. Please try again.")

    # In dev mode (no Resend key), auto-verify
    if not RESEND_API_KEY:
        _users.add(email)
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_FILE, "a") as f:
            f.write(json.dumps({"email": email, "signed_up": datetime.utcnow().isoformat()}) + "\n")
        return {"message": "Dev mode: auto-verified.", "token": _make_token(email), "verified": True}

    return {"message": f"Verification code sent to {email}. Check your inbox!", "verified": False}


@app.post("/verify")
async def verify_code(req: VerifyRequest, request: Request):
    """Verify the 6-digit code and return an auth token."""
    client_ip = request.client.host if request.client else "unknown"
    if not _verify_limiter.is_allowed(client_ip):
        raise HTTPException(429, "Too many verification attempts. Please wait a few minutes.")

    email = req.email.strip().lower()
    code = req.code.strip()

    # Reject non-numeric or wrong-length codes immediately
    if not code.isdigit() or len(code) != 6:
        raise HTTPException(400, "Invalid code format. Please enter the 6-digit code.")

    if not _check_verify_failures(email):
        raise HTTPException(429, "Too many failed attempts for this email. Please request a new code.")

    pending = _pending_codes.get(email)
    if not pending:
        raise HTTPException(400, "No verification code found for this email. Please sign up first.")

    stored_code, expiry = pending
    if time.time() > expiry:
        del _pending_codes[email]
        raise HTTPException(400, "Verification code expired. Please request a new one.")

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(code, stored_code):
        _record_verify_failure(email)
        raise HTTPException(400, "Invalid code. Please check and try again.")

    # Verified — clean up
    del _pending_codes[email]
    _clear_verify_failures(email)
    if email not in _users:
        _users.add(email)
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_FILE, "a") as f:
            f.write(json.dumps({"email": email, "signed_up": datetime.utcnow().isoformat(), "verified": True}) + "\n")

    token = _make_token(email)
    return {"message": "Bula! Email verified. Welcome to Lagi!", "token": token, "verified": True}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    location: str = Field(..., description="Fiji location: Suva, Nadi, Labasa, or Lautoka")
    enso_phase: str = Field("Neutral", description="ENSO phase: Nina, Neutral, Nino")


class ActualObservation(BaseModel):
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    location: str = Field(..., description="Location name")
    temperature: float = Field(..., description="Actual temperature in Celsius")
    rain_probability: float = Field(..., ge=0, le=100, description="Actual rain probability 0-100%")
    wind: float = Field(None, ge=0, description="Actual wind speed km/h")
    source: str = Field("manual", description="Data source name")


# ---------------------------------------------------------------------------
# Public info endpoints (no auth)
# ---------------------------------------------------------------------------
_startup_time = time.time()


@app.get("/")
async def root():
    return {
        "message": "Bula! Welcome to Lagi, the Fiji Weather Guardian.",
        "version": API_VERSION,
        "locations": VALID_LOCATIONS,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "Lagi",
        "version": API_VERSION,
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/locations")
async def list_locations():
    return {"locations": VALID_LOCATIONS}


# ---------------------------------------------------------------------------
# Protected endpoints (require auth token)
# ---------------------------------------------------------------------------
@app.post("/forecast")
async def get_forecast(req: ForecastRequest):
    """Auto-fetch weather predictions, apply Lagi corrections. Public endpoint."""
    if req.location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")
    if req.enso_phase not in VALID_ENSO:
        raise HTTPException(400, f"Invalid ENSO phase. Choose from: {VALID_ENSO}")

    ensemble_data = engine.fetch_ensemble_forecast(req.location)
    ensemble = ensemble_data["ensemble"]

    if ensemble["temperature"] is None:
        raise HTTPException(503, "Could not fetch forecasts from any source. Please try again shortly.")

    season = engine.get_season(datetime.utcnow().month)

    result = engine.correct_forecast(
        location=req.location,
        T_f=ensemble["temperature"],
        P_f=ensemble["rain_probability"] or 50.0,
        W_f=ensemble["wind"] or 15.0,
        humidity_f=ensemble["humidity"] or 75.0,
        season=season,
        enso_phase=req.enso_phase,
        ensemble_sources=ensemble.get("sources", []),
        ensemble_T=ensemble["temperature"],
        ensemble_P=ensemble["rain_probability"],
    )

    cyclone_risk = engine.assess_cyclone_risk(
        location=req.location,
        pressure=ensemble.get("pressure"),
        wind_gust=ensemble.get("wind_gust"),
        precip_24h=ensemble.get("precip_mm"),
        season=season,
        enso_phase=req.enso_phase,
    )

    raw_daily = ensemble.get("daily_forecast", [])
    daily_forecast = engine.correct_multi_day(raw_daily, season, req.enso_phase)

    commentary = engine.generate_forecast_commentary(
        req.location, result, cyclone_risk, daily_forecast
    )

    sources_used = ensemble.get("sources", [])
    source_names = {
        "fiji_met": "Fiji Meteorological Service",
        "bom_australia": "BOM Australia",
        "open_meteo": "Open-Meteo (ECMWF + GFS)",
    }
    source_labels = [source_names.get(s, s) for s in sources_used]

    return {
        "bula": f"Bula! Here's your adjusted forecast for {req.location}.",
        "location": req.location,
        "commentary": commentary,
        "source_message": f"Weather predictions pulled from: {', '.join(source_labels) or 'weather sites'}",
        "sources": source_labels,
        "n_sources": ensemble["n_sources"],
        "raw_forecast": {
            "temperature": ensemble["temperature"],
            "rain_probability": ensemble["rain_probability"],
            "wind": ensemble["wind"],
            "humidity": ensemble["humidity"],
        },
        "adjusted_forecast": result["adjusted_forecast"],
        "corrections": result["corrections"],
        "monte_carlo": result["monte_carlo"],
        "daily_forecast": daily_forecast,
        "cyclone_risk": cyclone_risk,
        "season": season,
        "enso_phase": req.enso_phase,
        "disclaimer": "This is guidance, not a guarantee. Always check official Fiji Met Service warnings.",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/cyclone")
async def get_cyclone_risk(location: str = "Suva", enso_phase: str = "Neutral",
                           user: str = Depends(verify_token)):
    """Cyclone risk assessment. Requires auth."""
    if location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")

    ensemble_data = engine.fetch_ensemble_forecast(location)
    ensemble = ensemble_data["ensemble"]
    season = engine.get_season(datetime.utcnow().month)

    cyclone_risk = engine.assess_cyclone_risk(
        location=location,
        pressure=ensemble.get("pressure"),
        wind_gust=ensemble.get("wind_gust"),
        precip_24h=ensemble.get("precip_mm"),
        season=season,
        enso_phase=enso_phase,
    )

    return {
        "location": location,
        "cyclone_risk": cyclone_risk,
        "data_sources": ensemble.get("sources", []),
        "timestamp": datetime.utcnow().isoformat(),
    }


class ChatRequest(BaseModel):
    question: str = Field(..., description="Natural language weather question", max_length=500)
    location: str = Field("Suva", description="Fiji location")
    enso_phase: str = Field("Neutral", description="ENSO phase")


@app.post("/chat")
async def chat(req: ChatRequest, user: str = Depends(verify_token)):
    """Natural language weather chat. Requires auth."""
    if req.location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")

    season = engine.get_season(datetime.utcnow().month)

    ensemble = None
    try:
        ensemble_data = engine.fetch_ensemble_forecast(req.location)
        ensemble = ensemble_data["ensemble"]
        T_f = ensemble["temperature"] or 28.0
        P_f = ensemble["rain_probability"] or 55.0
        W_f = ensemble["wind"] or 15.0
        H_f = ensemble["humidity"] or 75.0
    except Exception:
        T_f, P_f, W_f, H_f = 28.0, 55.0, 15.0, 75.0

    result = engine.correct_forecast(
        location=req.location,
        T_f=T_f, P_f=P_f, W_f=W_f, humidity_f=H_f,
        season=season, enso_phase=req.enso_phase,
    )

    cyclone_risk = engine.assess_cyclone_risk(
        location=req.location,
        pressure=ensemble.get("pressure") if ensemble else None,
        wind_gust=ensemble.get("wind_gust") if ensemble else None,
        precip_24h=ensemble.get("precip_mm") if ensemble else None,
        season=season,
        enso_phase=req.enso_phase,
    )

    reply = engine.natural_language_response(req.question, req.location, result, cyclone_risk)
    certainty = engine.compute_certainty()

    return {
        "question": req.question,
        "reply": reply,
        "location": req.location,
        "forecast": result["adjusted_forecast"],
        "cyclone_risk": cyclone_risk,
        "certainty": certainty,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/certainty")
async def get_certainty():
    """Prediction certainty level. Public endpoint."""
    return engine.compute_certainty()


# ---------------------------------------------------------------------------
# Admin endpoints (blocked from public — require token + admin check)
# ---------------------------------------------------------------------------
ADMIN_TOKEN = hashlib.sha256(f"admin:{SIGNUP_SECRET}".encode()).hexdigest()[:32]


async def verify_admin(request: Request):
    """Only allow admin access to validation endpoints."""
    token = request.headers.get("X-Lagi-Token", "")
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "Admin access required.")
    return token


@app.get("/validation/report")
async def validation_report(admin: str = Depends(verify_admin)):
    report = generate_validation_report()
    return PlainTextResponse(report)


@app.post("/validation/log-actual")
async def log_actual_observation(obs: ActualObservation, admin: str = Depends(verify_admin)):
    if obs.location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")
    log_actual(date=obs.date, location=obs.location, T_a=obs.temperature,
               P_a=obs.rain_probability, W_a=obs.wind, source=obs.source)
    return {"message": f"Vinaka! Actual observation logged for {obs.location} on {obs.date}."}


@app.post("/validation/compute")
async def recompute_coefficients(admin: str = Depends(verify_admin)):
    result = compute_correlation()
    engine.reload_dynamic_coefficients()
    return {"message": "Dynamic coefficients recomputed and reloaded.", "coefficients": result}


@app.get("/performance")
async def get_performance(admin: str = Depends(verify_admin)):
    from config import PERFORMANCE_LOG
    if not PERFORMANCE_LOG.exists():
        return {"message": "No performance data yet.", "metrics": []}
    records = []
    with open(PERFORMANCE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    recent = records[-12:] if len(records) > 12 else records
    return {"metrics": recent, "total_weeks": len(records)}


# ---------------------------------------------------------------------------
# Serve the web UI (public)
# ---------------------------------------------------------------------------
@app.get("/ui")
async def serve_ui():
    ui_path = Path(__file__).parent / "web_ui.html"
    if not ui_path.exists():
        raise HTTPException(404, "Web UI not found")
    return FileResponse(
        ui_path,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
