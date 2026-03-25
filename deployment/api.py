"""
Lagi -- Fiji Weather Guardian: FastAPI Server
==============================================
REST API with:
  - /forecast — manual forecast input with correction
  - /forecast/auto — auto-fetch from 5 sources, ensemble, correct
  - /validation/report — view prediction vs actual correlation
  - /validation/log-actual — log observed weather for validation
  - /validation/compute — recompute dynamic coefficients

Run:
    uvicorn deployment.api:app --host 0.0.0.0 --port 8000
"""

import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

# Add scripts dir so validation/data_fetcher imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from .inference import LagiInference
from validation import log_actual, compute_correlation, generate_validation_report

app = FastAPI(
    title="Lagi - Fiji Weather Guardian",
    description="AI-corrected weather forecasts for Fiji with 5-source ensemble and dynamic validation. Bula!",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = LagiInference()

VALID_LOCATIONS = ["Suva", "Nadi", "Labasa", "Lautoka"]
VALID_SEASONS = ["DJF", "MAM", "JJA", "SON"]
VALID_ENSO = ["Nina", "Neutral", "Nino"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    location: str = Field(..., description="Fiji location: Suva, Nadi, Labasa, or Lautoka")
    temperature: float = Field(..., description="Forecast temperature in Celsius")
    rain_probability: float = Field(..., ge=0, le=100, description="Rain probability 0-100%")
    wind: float = Field(..., ge=0, description="Wind speed in km/h")
    humidity: float = Field(75.0, ge=0, le=100, description="Humidity forecast 0-100%")
    season: str = Field(None, description="Season: DJF, MAM, JJA, SON (auto-detected if omitted)")
    enso_phase: str = Field("Neutral", description="ENSO phase: Nina, Neutral, Nino")


class AutoForecastRequest(BaseModel):
    location: str = Field(..., description="Fiji location: Suva, Nadi, Labasa, or Lautoka")
    enso_phase: str = Field("Neutral", description="ENSO phase: Nina, Neutral, Nino")


class ActualObservation(BaseModel):
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    location: str = Field(..., description="Location name")
    temperature: float = Field(..., description="Actual temperature in Celsius")
    rain_probability: float = Field(..., ge=0, le=100, description="Actual rain probability 0-100%")
    wind: float = Field(None, ge=0, description="Actual wind speed km/h")
    source: str = Field("manual", description="Data source name")


class ForecastResponse(BaseModel):
    bula: str
    location: str
    raw_forecast: dict
    adjusted_forecast: dict
    corrections: dict
    monte_carlo: dict
    season: str
    enso_phase: str
    disclaimer: str
    timestamp: str


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Bula! Welcome to Lagi, the Fiji Weather Guardian.",
        "version": "2.0.0",
        "locations": VALID_LOCATIONS,
        "endpoints": {
            "forecast": "POST /forecast — correct a manual forecast input",
            "auto_forecast": "POST /forecast/auto — auto-fetch 5 sources + ensemble + correct",
            "validation_report": "GET /validation/report — prediction vs actual correlation",
            "log_actual": "POST /validation/log-actual — log observed weather",
            "recompute": "POST /validation/compute — recompute dynamic coefficients",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "Lagi", "timestamp": datetime.utcnow().isoformat()}


@app.post("/forecast", response_model=ForecastResponse)
async def get_corrected_forecast(req: ForecastRequest):
    """Correct a manually provided forecast using static + dynamic adjustments."""
    if req.location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")
    if req.enso_phase not in VALID_ENSO:
        raise HTTPException(400, f"Invalid ENSO phase. Choose from: {VALID_ENSO}")

    season = req.season or engine.get_season(datetime.utcnow().month)
    if season not in VALID_SEASONS:
        raise HTTPException(400, f"Invalid season. Choose from: {VALID_SEASONS}")

    result = engine.correct_forecast(
        location=req.location,
        T_f=req.temperature,
        P_f=req.rain_probability,
        W_f=req.wind,
        humidity_f=req.humidity,
        season=season,
        enso_phase=req.enso_phase,
    )

    return ForecastResponse(
        bula=f"Bula! Here's your adjusted forecast for {req.location}.",
        location=result["location"],
        raw_forecast=result["raw_forecast"],
        adjusted_forecast=result["adjusted_forecast"],
        corrections=result["corrections"],
        monte_carlo=result["monte_carlo"],
        season=result["season"],
        enso_phase=result["enso_phase"],
        disclaimer="This is guidance, not a guarantee. Always check official Fiji Met Service warnings.",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/forecast/auto")
async def auto_forecast(req: AutoForecastRequest):
    """
    Auto-fetch forecasts from all 5 sources, compute ensemble average,
    apply static + dynamic corrections, return adjusted forecast.
    """
    if req.location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")

    # Step 1: Fetch from 5 sources
    ensemble_data = engine.fetch_ensemble_forecast(req.location)
    ensemble = ensemble_data["ensemble"]

    if ensemble["temperature"] is None:
        raise HTTPException(503, "Could not fetch forecasts from any source. Check API keys.")

    season = engine.get_season(datetime.utcnow().month)

    # Step 2: Apply corrections to ensemble average
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

    return {
        "bula": f"Bula! Here's your ensemble-adjusted forecast for {req.location}.",
        "location": req.location,
        "ensemble": ensemble,
        "individual_forecasts": ensemble_data["individual_forecasts"],
        "adjusted_forecast": result["adjusted_forecast"],
        "corrections": result["corrections"],
        "monte_carlo": result["monte_carlo"],
        "season": season,
        "enso_phase": req.enso_phase,
        "disclaimer": "This is guidance, not a guarantee. Always check official Fiji Met Service warnings.",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Validation endpoints
# ---------------------------------------------------------------------------
@app.get("/validation/report")
async def validation_report():
    """Get the current prediction vs actual validation report."""
    report = generate_validation_report()
    return PlainTextResponse(report)


@app.post("/validation/log-actual")
async def log_actual_observation(obs: ActualObservation):
    """Log an actual weather observation for validation against past predictions."""
    if obs.location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")

    log_actual(
        date=obs.date,
        location=obs.location,
        T_a=obs.temperature,
        P_a=obs.rain_probability,
        W_a=obs.wind,
        source=obs.source,
    )

    return {
        "message": f"Vinaka! Actual observation logged for {obs.location} on {obs.date}.",
        "next_step": "POST /validation/compute to recompute correlation coefficients.",
    }


@app.post("/validation/compute")
async def recompute_coefficients():
    """Recompute Pearson r correlation and dynamic adjustment coefficients."""
    result = compute_correlation()
    engine.reload_dynamic_coefficients()

    return {
        "message": "Dynamic coefficients recomputed and reloaded.",
        "coefficients": result,
    }


@app.get("/locations")
async def list_locations():
    return {"locations": VALID_LOCATIONS}


# ---------------------------------------------------------------------------
# Natural language chat endpoint
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., description="Natural language weather question")
    location: str = Field("Suva", description="Fiji location")
    enso_phase: str = Field("Neutral", description="ENSO phase")


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Natural language weather chat. Ask Lagi anything about the weather
    in plain English and get a friendly, informative Fijian-tone reply.
    """
    if req.location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")

    season = engine.get_season(datetime.utcnow().month)

    # Use default forecast values for NL responses (will use ensemble when API keys are set)
    result = engine.correct_forecast(
        location=req.location,
        T_f=28.0,  # tropical default
        P_f=55.0,
        W_f=15.0,
        humidity_f=75.0,
        season=season,
        enso_phase=req.enso_phase,
    )

    reply = engine.natural_language_response(req.question, req.location, result)
    certainty = engine.compute_certainty()

    return {
        "question": req.question,
        "reply": reply,
        "location": req.location,
        "forecast": result["adjusted_forecast"],
        "certainty": certainty,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Certainty endpoint
# ---------------------------------------------------------------------------
@app.get("/certainty")
async def get_certainty():
    """Get the current prediction certainty level."""
    certainty = engine.compute_certainty()
    return certainty


# Serve the web UI
@app.get("/ui")
async def serve_ui():
    ui_path = Path(__file__).parent / "web_ui.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    raise HTTPException(404, "Web UI not found")
