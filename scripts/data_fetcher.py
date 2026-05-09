#!/usr/bin/env python3
from __future__ import annotations
"""
Lagi -- Fiji Weather Guardian: Multi-Source Data Fetcher
=========================================================
Fetches historical weather data from 4 credible sources and
current forecasts from 5 forecast providers.

Historical Sources (15+ years):
  1. Fiji Meteorological Service (met.gov.fj)
  2. NOAA GHCN (Global Historical Climatology Network)
  3. CRU / ERA5 via World Bank Climate API
  4. NIWA Pacific Climate Portal

Current Forecast Sources:
  1. Fiji Met Service (met.gov.fj)
  2. Windy.com API
  3. AccuWeather API
  4. Weather Underground API
  5. BOM Australia / Metvuw
"""

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, CACHE_DIR, LOCATIONS,
    RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS,
    HTTP_TIMEOUT, HTTP_MAX_RETRIES,
)

logger = logging.getLogger("lagi.data_fetcher")


# ---------------------------------------------------------------------------
# Rate limiting helper
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple rate limiter: max N requests per window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps: list[float] = []

    def wait(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < self.window]
        if len(self.timestamps) >= self.max_requests:
            sleep_time = self.window - (now - self.timestamps[0]) + 0.1
            logger.info("Rate limit: sleeping %.1f seconds", sleep_time)
            time.sleep(sleep_time)
        self.timestamps.append(time.time())


rate_limiter = RateLimiter(max_requests=RATE_LIMIT_MAX_REQUESTS, window_seconds=RATE_LIMIT_WINDOW_SECONDS)


def _safe_get(url: str, params: dict = None, headers: dict = None, timeout: int = HTTP_TIMEOUT) -> requests.Response | None:
    """HTTP GET with rate limiting, retries, and error handling."""
    rate_limiter.wait()
    for attempt in range(HTTP_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                logger.warning("Rate limited (429). Waiting %ds before retry %d/3", wait, attempt + 1)
                time.sleep(wait)
            else:
                logger.error("HTTP error %s fetching %s: %s", resp.status_code, url, e)
                return None
        except requests.exceptions.RequestException as e:
            logger.error("Request error fetching %s (attempt %d/3): %s", url, attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


# ===========================================================================
# HISTORICAL DATA SOURCES (15+ years)
# ===========================================================================

def fetch_noaa_ghcn(location: str, start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
    """
    Fetch daily weather data from NOAA GHCN-Daily via their public API.
    Source: https://www.ncei.noaa.gov/cdo-web/api/v2/
    Requires NOAA CDO token (free): https://www.ncdc.noaa.gov/cdo-web/token

    Returns DataFrame with columns: date, location, T_a, P_a, W_a
    """
    token = os.environ.get("NOAA_CDO_TOKEN", "")
    if not token:
        logger.warning("NOAA_CDO_TOKEN not set. Set it via: export NOAA_CDO_TOKEN=your_token")
        logger.info("Get a free token at: https://www.ncdc.noaa.gov/cdo-web/token")
        return pd.DataFrame()

    loc = LOCATIONS.get(location, {})
    station_id = loc.get("ghcn_id", "")
    if not station_id:
        logger.warning("No GHCN station ID for %s", location)
        return pd.DataFrame()

    headers = {"token": token}
    records = []

    for year in range(start_year, end_year + 1):
        url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
        params = {
            "datasetid": "GHCND",
            "stationid": f"GHCND:{station_id}",
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "datatypeid": "TMAX,TMIN,PRCP,AWND",
            "units": "metric",
            "limit": 1000,
        }

        resp = _safe_get(url, params=params, headers=headers)
        if not resp:
            continue

        data = resp.json()
        results = data.get("results", [])

        # Group by date
        daily = {}
        for r in results:
            date = r["date"][:10]
            if date not in daily:
                daily[date] = {}
            daily[date][r["datatype"]] = r["value"]

        for date, vals in daily.items():
            tmax = vals.get("TMAX")
            tmin = vals.get("TMIN")
            t_a = ((tmax + tmin) / 2) if tmax is not None and tmin is not None else None
            p_a = vals.get("PRCP", 0)  # mm
            w_a = vals.get("AWND")  # m/s -> km/h

            if t_a is not None:
                records.append({
                    "date": date,
                    "location": location,
                    "T_a": round(t_a, 1),
                    "P_a": round(min(p_a / 10 * 100, 100), 1) if p_a else 0,  # Convert to rain indicator
                    "W_a": round(w_a * 3.6, 1) if w_a else None,
                    "source": "NOAA_GHCN",
                })

        logger.info("NOAA GHCN: %s %d — %d records", location, year, len(results))

    df = pd.DataFrame(records)
    logger.info("NOAA GHCN total: %d records for %s", len(df), location)
    return df


def fetch_worldbank_cru(location: str, start_year: int = 2010, end_year: int = 2020) -> pd.DataFrame:
    """
    Fetch climate data from World Bank Climate API (CRU/ERA5 derived).
    Source: https://climateknowledgeportal.worldbank.org/
    No API key needed — public endpoint.

    Returns monthly averages (temperature, precipitation).
    """
    loc = LOCATIONS.get(location, {})
    records = []

    # World Bank Climate API — monthly data
    for var, var_name in [("tas", "temperature"), ("pr", "precipitation")]:
        url = f"https://climateknowledgeportal.worldbank.org/api/data/get-download-data/historical/cru/{var}/monthly/FJI"

        resp = _safe_get(url)
        if not resp:
            logger.warning("World Bank CRU: could not fetch %s data", var_name)
            continue

        try:
            lines = resp.text.strip().split("\n")
            reader = csv.DictReader(lines)
            for row in reader:
                year_str = row.get("Year", "")
                if not year_str.isdigit():
                    continue
                year = int(year_str)
                if year < start_year or year > end_year:
                    continue

                for month_name in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                    val = row.get(month_name)
                    if val and val.strip():
                        try:
                            value = float(val)
                            month_num = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(month_name) + 1
                            records.append({
                                "year": year,
                                "month": month_num,
                                "location": location,
                                "variable": var_name,
                                "value": round(value, 2),
                                "source": "WorldBank_CRU",
                            })
                        except ValueError:
                            pass
        except Exception as e:
            logger.error("World Bank CRU parse error: %s", e)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info("World Bank CRU: %d monthly records for %s", len(df), location)
    return df


def fetch_niwa_pacific(location: str) -> pd.DataFrame:
    """
    Fetch data from NIWA Pacific Climate Portal.
    Source: https://www.pacificclimate.org/
    Public datasets for Pacific Island nations.

    Returns available climate records for the location.
    """
    loc = LOCATIONS.get(location, {})
    records = []

    # NIWA CliFlo-style endpoint for Pacific stations
    url = "https://cliflo.niwa.co.nz/pls/niwp/wgenf.genform1_proc"
    params = {
        "cession": "public",
        "lession": location,
        "lat": loc.get("lat", ""),
        "lon": loc.get("lon", ""),
    }

    resp = _safe_get(url, params=params)
    if resp:
        logger.info("NIWA Pacific: response received for %s (%d bytes)", location, len(resp.content))
        # Parse HTML/CSV response — structure varies by endpoint
        # Store raw response for manual processing
        cache_path = CACHE_DIR / f"niwa_{location.lower()}.html"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text)
        logger.info("NIWA Pacific: cached response to %s", cache_path)
    else:
        logger.warning("NIWA Pacific: could not fetch data for %s", location)

    return pd.DataFrame(records)


def fetch_fiji_met_historical(location: str) -> pd.DataFrame:
    """
    Fetch historical data from Fiji Meteorological Service.
    Source: https://www.met.gov.fj/

    Note: Fiji Met Service data often requires formal data request.
    This function checks their public pages and archives.
    """
    records = []

    # Check Fiji Met public climate summary page
    url = "https://www.met.gov.fj/ClimateofFiji.pdf"
    resp = _safe_get(url)
    if resp:
        cache_path = CACHE_DIR / "fiji_met_climate_summary.pdf"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        logger.info("Fiji Met: downloaded climate summary PDF (%d KB)", len(resp.content) // 1024)

    # Check for recent observations page
    obs_url = "https://www.met.gov.fj/aifs_prods/synop_latest.txt"
    resp = _safe_get(obs_url)
    if resp:
        logger.info("Fiji Met: latest synop observations available (%d bytes)", len(resp.content))
        cache_path = CACHE_DIR / "fiji_met_latest_synop.txt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text)

    logger.info("Fiji Met: for comprehensive historical data, submit a formal request to climate@met.gov.fj")
    return pd.DataFrame(records)


def fetch_all_historical(start_year: int = 2010, end_year: int = 2025) -> pd.DataFrame:
    """
    Fetch historical weather data from all 4 credible sources for all locations.
    Merges into a unified DataFrame.
    """
    all_data = []

    for location in LOCATIONS:
        logger.info("Fetching historical data for %s...", location)

        # Source 1: NOAA GHCN
        ghcn = fetch_noaa_ghcn(location, start_year, end_year)
        if not ghcn.empty:
            all_data.append(ghcn)

        # Source 2: World Bank CRU
        cru = fetch_worldbank_cru(location, start_year, min(end_year, 2020))
        if not cru.empty:
            all_data.append(cru)

        # Source 3: NIWA Pacific
        niwa = fetch_niwa_pacific(location)
        if not niwa.empty:
            all_data.append(niwa)

        # Source 4: Fiji Met Service
        fiji_met = fetch_fiji_met_historical(location)
        if not fiji_met.empty:
            all_data.append(fiji_met)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Save combined historical data
        output_path = DATA_DIR / "historical_actuals.csv"
        combined.to_csv(output_path, index=False)
        logger.info("Saved %d historical records to %s", len(combined), output_path)
        return combined

    logger.warning("No historical data fetched. Check API keys and network connectivity.")
    return pd.DataFrame()


# ===========================================================================
# CURRENT FORECAST SOURCES (5 providers)
# ===========================================================================

def fetch_fiji_met_forecast(location: str) -> dict | None:
    """
    Fetch current forecast from Fiji Meteorological Service.
    Source: https://www.met.gov.fj/
    """
    url = "https://www.met.gov.fj/aifs_prods/forecast_latest.txt"
    resp = _safe_get(url)
    if not resp:
        return None

    # Parse text forecast — Fiji Met uses plain text bulletins
    text = resp.text
    logger.info("Fiji Met forecast: received %d chars", len(text))

    # Extract temperature and rain indicators from bulletin text
    # This is a simplified parser — production would use NLP or regex patterns
    loc = LOCATIONS.get(location, {})
    forecast = {
        "source": "fiji_met",
        "location": location,
        "raw_text": text[:500],
        "temperature": None,
        "rain_probability": None,
        "wind": None,
        "humidity": None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Try to extract numbers from forecast text
    import re
    temp_match = re.search(r'(\d{2,3})\s*(?:degrees?|°|C)', text, re.IGNORECASE)
    if temp_match:
        forecast["temperature"] = float(temp_match.group(1))

    wind_match = re.search(r'(\d{1,3})\s*(?:km/?h|knots|kts)', text, re.IGNORECASE)
    if wind_match:
        val = float(wind_match.group(1))
        if "knot" in text[wind_match.start():wind_match.end() + 10].lower():
            val *= 1.852  # knots to km/h
        forecast["wind"] = round(val, 1)

    # Rain probability from keywords
    text_lower = text.lower()
    if any(w in text_lower for w in ["heavy rain", "flooding", "downpour"]):
        forecast["rain_probability"] = 90
    elif any(w in text_lower for w in ["rain", "showers", "wet"]):
        forecast["rain_probability"] = 70
    elif any(w in text_lower for w in ["scattered", "isolated", "possible"]):
        forecast["rain_probability"] = 40
    elif any(w in text_lower for w in ["fine", "sunny", "clear", "dry"]):
        forecast["rain_probability"] = 10

    return forecast


def fetch_windy_forecast(location: str) -> dict | None:
    """
    Fetch forecast from Windy.com API.
    Requires API key: https://api.windy.com/
    """
    api_key = os.environ.get("WINDY_API_KEY", "")
    if not api_key:
        logger.warning("WINDY_API_KEY not set")
        return None

    loc = LOCATIONS.get(location, {})
    url = "https://api.windy.com/api/point-forecast/v2"
    payload = {
        "lat": loc["lat"],
        "lon": loc["lon"],
        "model": "gfs",
        "parameters": ["temp", "precip", "wind", "rh"],
        "levels": ["surface"],
        "key": api_key,
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        temps = data.get("temp-surface", [])
        precips = data.get("past3hprecip-surface", [])
        winds_u = data.get("wind_u-surface", [])
        winds_v = data.get("wind_v-surface", [])
        humidity = data.get("rh-surface", [])

        # Take first forecast point (next available)
        temp = temps[0] - 273.15 if temps else None  # Kelvin to Celsius
        precip = precips[0] if precips else None
        wind_speed = None
        if winds_u and winds_v:
            wind_speed = round(np.sqrt(winds_u[0] ** 2 + winds_v[0] ** 2) * 3.6, 1)  # m/s to km/h

        return {
            "source": "windy",
            "location": location,
            "temperature": round(temp, 1) if temp else None,
            "rain_probability": min(precip * 20, 100) if precip else None,
            "wind": wind_speed,
            "humidity": humidity[0] if humidity else None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Windy API error: %s", e)
        return None


def fetch_accuweather_forecast(location: str) -> dict | None:
    """
    Fetch forecast from AccuWeather API.
    Requires API key: https://developer.accuweather.com/
    """
    api_key = os.environ.get("ACCUWEATHER_API_KEY", "")
    if not api_key:
        logger.warning("ACCUWEATHER_API_KEY not set")
        return None

    # AccuWeather location keys for Fiji cities
    location_keys = {
        "Suva": "241555",
        "Nadi": "241538",
        "Labasa": "241530",
        "Lautoka": "241532",
    }

    loc_key = location_keys.get(location)
    if not loc_key:
        return None

    url = f"http://dataservice.accuweather.com/forecasts/v1/daily/1day/{loc_key}"
    params = {"apikey": api_key, "details": "true", "metric": "true"}

    resp = _safe_get(url, params=params)
    if not resp:
        return None

    try:
        data = resp.json()
        forecast = data.get("DailyForecasts", [{}])[0]
        temp_max = forecast.get("Temperature", {}).get("Maximum", {}).get("Value")
        temp_min = forecast.get("Temperature", {}).get("Minimum", {}).get("Value")
        temp = (temp_max + temp_min) / 2 if temp_max and temp_min else None

        day = forecast.get("Day", {})
        rain_prob = day.get("RainProbability", day.get("PrecipitationProbability"))
        wind_speed = day.get("Wind", {}).get("Speed", {}).get("Value")

        return {
            "source": "accuweather",
            "location": location,
            "temperature": round(temp, 1) if temp else None,
            "rain_probability": rain_prob,
            "wind": wind_speed,
            "humidity": None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("AccuWeather parse error: %s", e)
        return None


def fetch_wunderground_forecast(location: str) -> dict | None:
    """
    Fetch forecast from Weather Underground API.
    Requires API key: https://www.wunderground.com/weather/api/
    """
    api_key = os.environ.get("WUNDERGROUND_API_KEY", "")
    if not api_key:
        logger.warning("WUNDERGROUND_API_KEY not set")
        return None

    loc = LOCATIONS.get(location, {})
    url = f"https://api.weather.com/v3/wx/forecast/daily/5day"
    params = {
        "geocode": f"{loc['lat']},{loc['lon']}",
        "format": "json",
        "units": "m",
        "language": "en-US",
        "apiKey": api_key,
    }

    resp = _safe_get(url, params=params)
    if not resp:
        return None

    try:
        data = resp.json()
        temps = data.get("temperatureMax", [])
        temps_min = data.get("temperatureMin", [])
        precip = data.get("qpf", [])
        wind = data.get("windSpeed", [])
        humidity = data.get("relativeHumidity", [])

        temp = (temps[0] + temps_min[0]) / 2 if temps and temps_min else None

        return {
            "source": "wunderground",
            "location": location,
            "temperature": round(temp, 1) if temp else None,
            "rain_probability": min(precip[0] * 10, 100) if precip else None,
            "wind": wind[0] if wind else None,
            "humidity": humidity[0] if humidity else None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Wunderground parse error: %s", e)
        return None


def fetch_bom_metvuw_forecast(location: str) -> dict | None:
    """
    Fetch forecast from BOM Australia / Metvuw (covers SW Pacific).
    Source: http://www.bom.gov.au/pacific/ (public, no key needed)
    """
    # BOM Pacific forecast page
    url = "http://www.bom.gov.au/fwo/IDZ00066.json"

    resp = _safe_get(url)
    if not resp:
        # Fallback: try Metvuw
        resp = _safe_get("http://metvuw.com/forecast/forecast.php?type=rain&region=pacific&noofdays=2")
        if resp:
            cache_path = CACHE_DIR / "metvuw_pacific.html"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(resp.text)
            logger.info("Metvuw: cached Pacific forecast page")
        return None

    try:
        data = resp.json()
        observations = data.get("observations", {}).get("data", [])

        loc = LOCATIONS.get(location, {})
        # Find closest station
        best = None
        best_dist = float("inf")
        for obs in observations:
            lat = obs.get("lat", 0)
            lon = obs.get("lon", 0)
            dist = (lat - loc["lat"]) ** 2 + (lon - loc["lon"]) ** 2
            if dist < best_dist:
                best_dist = dist
                best = obs

        if best:
            return {
                "source": "bom_australia",
                "location": location,
                "temperature": best.get("air_temp"),
                "rain_probability": None,
                "wind": best.get("wind_spd_kmh"),
                "humidity": best.get("rel_hum"),
                "timestamp": datetime.utcnow().isoformat(),
            }
    except Exception as e:
        logger.error("BOM/Metvuw parse error: %s", e)
    return None


def fetch_openmeteo_forecast(location: str) -> dict | None:
    """
    Fetch forecast from Open-Meteo API.
    Source: https://open-meteo.com/
    FREE, no API key needed. Excellent coverage for Pacific Islands.
    """
    loc = LOCATIONS.get(location, {})
    if not loc:
        return None

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_probability_max", "precipitation_sum",
            "wind_speed_10m_max", "wind_gusts_10m_max",
            "relative_humidity_2m_max",
        ]),
        "hourly": "surface_pressure",
        "timezone": "Pacific/Fiji",
        "forecast_days": 7,
    }

    resp = _safe_get(url, params=params)
    if not resp:
        return None

    try:
        data = resp.json()
        daily = data.get("daily", {})

        dates = daily.get("time", [])
        t_maxes = daily.get("temperature_2m_max", [])
        t_mins = daily.get("temperature_2m_min", [])
        rain_probs = daily.get("precipitation_probability_max", [])
        winds = daily.get("wind_speed_10m_max", [])
        humidities = daily.get("relative_humidity_2m_max", [])
        wind_gusts = daily.get("wind_gusts_10m_max", [])
        precip_sums = daily.get("precipitation_sum", [])

        # Build 7-day forecast array
        daily_forecast = []
        for i in range(min(7, len(dates))):
            t_max_i = t_maxes[i] if i < len(t_maxes) else None
            t_min_i = t_mins[i] if i < len(t_mins) else None
            temp_i = round((t_max_i + t_min_i) / 2, 1) if t_max_i is not None and t_min_i is not None else None

            daily_forecast.append({
                "date": dates[i] if i < len(dates) else None,
                "temperature": temp_i,
                "temp_max": round(t_max_i, 1) if t_max_i is not None else None,
                "temp_min": round(t_min_i, 1) if t_min_i is not None else None,
                "rain_probability": rain_probs[i] if i < len(rain_probs) else None,
                "wind": round(winds[i], 1) if i < len(winds) and winds[i] is not None else None,
                "wind_gust": round(wind_gusts[i], 1) if i < len(wind_gusts) and wind_gusts[i] is not None else None,
                "precip_mm": round(precip_sums[i], 1) if i < len(precip_sums) and precip_sums[i] is not None else None,
                "humidity": humidities[i] if i < len(humidities) else None,
            })

        # Day 0 values for backward compat
        day0 = daily_forecast[0] if daily_forecast else {}
        temp = day0.get("temperature")
        rain_prob = day0.get("rain_probability")
        wind = day0.get("wind")
        humidity = day0.get("humidity")
        wind_gust = day0.get("wind_gust")
        precip_mm = day0.get("precip_mm")

        # Get mean surface pressure from hourly data (first 24h)
        hourly = data.get("hourly", {})
        pressure_values = hourly.get("surface_pressure", [])
        pressure_today = pressure_values[:24] if pressure_values else []
        pressure = round(float(np.mean(pressure_today)), 1) if pressure_today else None

        return {
            "source": "open_meteo",
            "location": location,
            "temperature": temp,
            "rain_probability": rain_prob,
            "wind": wind,
            "humidity": humidity,
            "wind_gust": wind_gust,
            "pressure": pressure,
            "precip_mm": precip_mm,
            "daily_forecast": daily_forecast,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Open-Meteo parse error: %s", e)
        return None


# Active weather sources (all free, no API keys required)
WEATHER_SOURCES = [
    "Fiji Meteorological Service (met.gov.fj)",
    "BOM Australia / Metvuw",
    "Open-Meteo (ECMWF + GFS)",
]


def fetch_all_current_forecasts(location: str) -> list[dict]:
    """
    Fetch current forecasts from 3 free sources for a given location.
    Sources: Fiji Met Service, BOM/Metvuw, Open-Meteo
    Returns list of forecast dicts (only successful fetches).
    """
    fetchers = [
        ("Fiji Met Service", fetch_fiji_met_forecast),
        ("BOM/Metvuw", fetch_bom_metvuw_forecast),
        ("Open-Meteo", fetch_openmeteo_forecast),
    ]

    forecasts = []
    for name, fetcher in fetchers:
        logger.info("Fetching forecast from %s for %s...", name, location)
        try:
            result = fetcher(location)
            if result and result.get("temperature") is not None:
                forecasts.append(result)
                logger.info("  %s: T=%.1f, P=%s, W=%s",
                            name,
                            result["temperature"],
                            result.get("rain_probability", "N/A"),
                            result.get("wind", "N/A"))
            else:
                logger.warning("  %s: no data returned", name)
        except Exception as e:
            logger.error("  %s: error — %s", name, e)

    logger.info("Got %d/%d forecasts for %s", len(forecasts), len(fetchers), location)
    return forecasts


def load_source_weights(location: str) -> dict:
    """
    Load per-source accuracy weights for a location.
    Weights are derived from historical MAE against actuals.
    Returns dict mapping source name -> weight (higher = more accurate).
    Falls back to equal weights if no weights file exists.
    """
    weights_path = DATA_DIR / "source_weights.json"
    if not weights_path.exists():
        return {}
    try:
        with open(weights_path) as f:
            all_weights = json.load(f)
        return all_weights.get(location, {})
    except Exception as e:
        logger.warning("Could not load source weights: %s", e)
        return {}


def compute_ensemble_average(forecasts: list[dict], location: str | None = None) -> dict:
    """
    Compute weighted ensemble average from multiple forecast sources.
    Uses per-source accuracy weights when available (from source_weights.json),
    falls back to simple averaging when weights are unavailable.
    """
    if not forecasts:
        return {"temperature": None, "rain_probability": None, "wind": None, "humidity": None, "n_sources": 0}

    # Load weights for this location
    weights_map = load_source_weights(location) if location else {}

    def _weighted_mean(values: list[float], sources: list[str]) -> float:
        if not values:
            return None
        if not weights_map:
            return round(float(np.mean(values)), 1)

        weights = []
        for src in sources:
            w = weights_map.get(src, 1.0)
            weights.append(w)

        total_w = sum(weights)
        if total_w == 0:
            return round(float(np.mean(values)), 1)

        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return round(weighted_sum / total_w, 1)

    # Collect values and their source names in parallel
    temp_vals, temp_srcs = [], []
    rain_vals, rain_srcs = [], []
    wind_vals, wind_srcs = [], []
    hum_vals, hum_srcs = [], []

    gust_vals, gust_srcs = [], []
    pressure_vals, pressure_srcs = [], []
    precip_vals, precip_srcs = [], []

    for f in forecasts:
        src = f.get("source", "unknown")
        if f.get("temperature") is not None:
            temp_vals.append(f["temperature"])
            temp_srcs.append(src)
        if f.get("rain_probability") is not None:
            rain_vals.append(f["rain_probability"])
            rain_srcs.append(src)
        if f.get("wind") is not None:
            wind_vals.append(f["wind"])
            wind_srcs.append(src)
        if f.get("humidity") is not None:
            hum_vals.append(f["humidity"])
            hum_srcs.append(src)
        if f.get("wind_gust") is not None:
            gust_vals.append(f["wind_gust"])
            gust_srcs.append(src)
        if f.get("pressure") is not None:
            pressure_vals.append(f["pressure"])
            pressure_srcs.append(src)
        if f.get("precip_mm") is not None:
            precip_vals.append(f["precip_mm"])
            precip_srcs.append(src)

    weighting_method = "weighted" if weights_map else "equal"

    # Merge daily_forecast arrays — use the longest one available
    # (typically only Open-Meteo provides 7-day data)
    daily_forecast = []
    for f in forecasts:
        df = f.get("daily_forecast", [])
        if len(df) > len(daily_forecast):
            daily_forecast = df

    return {
        "temperature": _weighted_mean(temp_vals, temp_srcs),
        "rain_probability": _weighted_mean(rain_vals, rain_srcs),
        "wind": _weighted_mean(wind_vals, wind_srcs),
        "humidity": _weighted_mean(hum_vals, hum_srcs),
        "wind_gust": _weighted_mean(gust_vals, gust_srcs),
        "pressure": _weighted_mean(pressure_vals, pressure_srcs),
        "precip_mm": _weighted_mean(precip_vals, precip_srcs),
        "daily_forecast": daily_forecast,
        "n_sources": len(forecasts),
        "sources": [f["source"] for f in forecasts],
        "weighting": weighting_method,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Cyclone warnings from external sources
# ---------------------------------------------------------------------------
def fetch_cyclone_warnings() -> list[dict]:
    """
    Check for active tropical cyclone warnings from BOM and Fiji Met Service.
    Returns list of active systems with name, category, and source.
    """
    warnings = []

    # Check Fiji Met Service warning bulletin
    try:
        resp = _safe_get("https://www.met.gov.fj/aifs_prods/TCW_latest.txt")
        if resp and resp.text:
            text = resp.text.lower()
            # Check for active cyclone keywords
            if any(kw in text for kw in ["tropical cyclone", "tropical disturbance",
                                          "tropical depression", "category"]):
                import re
                # Try to extract cyclone name
                name_match = re.search(r'tropical cyclone\s+(\w+)', text, re.IGNORECASE)
                cat_match = re.search(r'category\s+(\d)', text, re.IGNORECASE)
                warnings.append({
                    "source": "fiji_met",
                    "name": name_match.group(1).title() if name_match else "Unknown",
                    "category": int(cat_match.group(1)) if cat_match else None,
                    "type": "tropical_cyclone",
                    "raw_text": resp.text[:300],
                })
                logger.info("FMS cyclone warning detected: %s", warnings[-1]["name"])
    except Exception as e:
        logger.warning("Could not check FMS cyclone warnings: %s", e)

    # Check BOM tropical cyclone feed
    try:
        resp = _safe_get("http://www.bom.gov.au/fwo/IDZ00066.json")
        if resp:
            data = resp.json()
            observations = data.get("observations", {}).get("data", [])
            # Look for cyclone-strength observations near Fiji (lat -12 to -22, lon 176-180)
            for obs in observations:
                lat = obs.get("lat", 0)
                lon = obs.get("lon", 0)
                wind_spd = obs.get("wind_spd_kmh", 0)
                if -22 <= lat <= -12 and 170 <= lon <= 185 and wind_spd and wind_spd > 90:
                    warnings.append({
                        "source": "bom_australia",
                        "name": obs.get("name", "Unknown"),
                        "category": None,
                        "type": "severe_weather",
                        "wind_kmh": wind_spd,
                    })
                    logger.info("BOM severe weather detected near Fiji: wind %d km/h", wind_spd)
    except Exception as e:
        logger.warning("Could not check BOM cyclone data: %s", e)

    return warnings


# ---------------------------------------------------------------------------
# CLI entry point for manual data fetching
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Lagi Data Fetcher")
    parser.add_argument("--mode", choices=["historical", "forecast", "both"], default="both")
    parser.add_argument("--location", choices=list(LOCATIONS.keys()), default=None)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("historical", "both"):
        logger.info("=== Fetching historical data ===")
        historical = fetch_all_historical()
        if not historical.empty:
            logger.info("Total historical records: %d", len(historical))

    if args.mode in ("forecast", "both"):
        logger.info("=== Fetching current forecasts ===")
        locations = [args.location] if args.location else list(LOCATIONS.keys())
        for loc in locations:
            forecasts = fetch_all_current_forecasts(loc)
            ensemble = compute_ensemble_average(forecasts)
            logger.info("Ensemble for %s: T=%s, P=%s, W=%s (%d sources)",
                        loc, ensemble["temperature"], ensemble["rain_probability"],
                        ensemble["wind"], ensemble["n_sources"])
