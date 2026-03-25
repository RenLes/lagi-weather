"""
Lagi -- Fiji Weather Guardian: FastAPI Server
==============================================
REST API for weather forecast correction.
Serves adjusted forecasts for Suva, Nadi, Labasa, Lautoka.

Run:
    uvicorn deployment.api:app --host 0.0.0.0 --port 8000
"""

from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import LagiInference

app = FastAPI(
    title="Lagi - Fiji Weather Guardian",
    description="AI-corrected weather forecasts for Fiji. Bula!",
    version="1.0.0",
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


class ForecastRequest(BaseModel):
    location: str = Field(..., description="Fiji location: Suva, Nadi, Labasa, or Lautoka")
    temperature: float = Field(..., description="Forecast temperature in Celsius")
    rain_probability: float = Field(..., ge=0, le=100, description="Rain probability 0-100%")
    wind: float = Field(..., ge=0, description="Wind speed in km/h")
    humidity: float = Field(75.0, ge=0, le=100, description="Humidity forecast 0-100%")
    season: str = Field(None, description="Season: DJF, MAM, JJA, SON (auto-detected if omitted)")
    enso_phase: str = Field("Neutral", description="ENSO phase: Nina, Neutral, Nino")


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


@app.get("/")
async def root():
    return {
        "message": "Bula! Welcome to Lagi, the Fiji Weather Guardian.",
        "locations": VALID_LOCATIONS,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "Lagi", "timestamp": datetime.utcnow().isoformat()}


@app.post("/forecast", response_model=ForecastResponse)
async def get_corrected_forecast(req: ForecastRequest):
    if req.location not in VALID_LOCATIONS:
        raise HTTPException(400, f"Invalid location. Choose from: {VALID_LOCATIONS}")
    if req.enso_phase not in VALID_ENSO:
        raise HTTPException(400, f"Invalid ENSO phase. Choose from: {VALID_ENSO}")

    # Auto-detect season from current month
    season = req.season
    if not season:
        season = engine.get_season(datetime.utcnow().month)
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


@app.get("/locations")
async def list_locations():
    return {"locations": VALID_LOCATIONS}
