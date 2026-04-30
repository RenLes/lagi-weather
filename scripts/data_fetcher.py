#!/usr/bin/env python3
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
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("lagi.data_fetcher")

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

LOCATIONS = {
    "Suva": {"lat": -18.1416, "lon": 178.4419, "ghcn_id": "FJM00091680", "wmo_id": "91680"},
    "Nadi": {"lat": -17.7765, "lon": 177.9640, "ghcn_id": "FJM00091680", "wmo_id": "91685"},
    "Labasa": {"lat": -16.4167, "lon": 179.3667, "ghcn_id": "FJ000091670", "wmo_id": "91670"},
    "Lautoka": {"lat": -17.6065, "lon": 177.4540, "ghcn_id": "FJ000091684", "wmo_id": "91684"},
}


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


rate_limiter = RateLimiter(max_requests=10, window_seconds=60)


def _safe_get(url: str, params: dict = None, headers: dict = None, timeout: int = 30) -> requests.Response | None:
    """HTTP GET with rate limiting, retries, and error handling."""
    rate_limiter.wait()
    for attempt in range(3):
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

# Fiji Met groups locations into multi-city forecast sections in the bulletin.
# Map each Lagi location to the section header that introduces its forecast.
_FIJI_MET_REGION = {
    "Suva": "Navua / Suva / Nausori",
    "Nadi": "Nadi / Lautoka / Ba",
    "Lautoka": "Nadi / Lautoka / Ba",
    "Labasa": "Labasa",
}


def _strip_html(html: str) -> str:
    """Strip HTML tags, scripts, and styles to plain whitespace-collapsed text."""
    from html.parser import HTMLParser
    import re as _re

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: list[str] = []
            self.skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self.skip = max(0, self.skip - 1)

        def handle_data(self, data):
            if not self.skip:
                self.parts.append(data)

    s = _Stripper()
    s.feed(html)
    return _re.sub(r"\s+", " ", " ".join(s.parts)).strip()


def _classify_rain(text: str) -> int | None:
    """
    Map a Fiji Met bulletin snippet to a rain probability percentage.

    Order matters: negation and "mainly fine" patterns are checked BEFORE the
    bare "rain"/"showers" substring match, otherwise bulletins like
    "no rain expected" or "mainly fine with isolated showers" would trip the
    rain branch and report 70% — the source of the historical perma-rain bias.
    """
    text_lower = text.lower()

    heavy_patterns = ["heavy rain", "flooding", "downpour", "torrential"]
    negation_patterns = [
        "no rain", "rain unlikely", "chance of rain low", "low chance of rain",
        "no significant rain", "minimal rain", "rain not expected",
    ]
    fine_patterns = ["mainly fine", "mostly fine", "mainly dry", "mostly dry",
                     "fine and sunny", "fine and clear", "fine weather"]
    scattered_patterns = ["scattered showers", "isolated showers", "brief showers",
                          "possible showers", "occasional showers", "light showers",
                          "afternoon or evening showers"]

    if any(p in text_lower for p in heavy_patterns):
        return 90
    if any(p in text_lower for p in negation_patterns):
        return 15
    if any(p in text_lower for p in fine_patterns):
        return 25
    if any(p in text_lower for p in scattered_patterns):
        return 35
    if any(w in text_lower for w in ["fine", "sunny", "clear", "dry"]):
        return 25
    if "cloudy" in text_lower and "shower" in text_lower:
        return 65
    if any(w in text_lower for w in ["showers", "wet"]):
        return 60
    if "rain" in text_lower:
        return 70
    return None


def fetch_fiji_met_forecast(location: str) -> dict | None:
    """
    Fetch current forecast from Fiji Meteorological Service.

    Bulletin lives at https://www.met.gov.fj/fiji-weather/fiji-weather/public/
    (the previous /aifs_prods/forecast_latest.txt URL was retired in early 2026
    when met.gov.fj migrated to their new CMS). The page is server-rendered
    HTML containing per-region paragraphs like:

        For Navua / Suva / Nausori : Cloudy with some showers. Tomorrow:
        Min: 22, Max: 31. Outlook for Saturday : ...

    We strip HTML, slice out the section for the requested location, then
    pull min/max temperatures and a coarse rain class from the text.
    """
    import re

    url = "https://www.met.gov.fj/fiji-weather/fiji-weather/public/"
    resp = _safe_get(url)
    if not resp:
        return None

    text = _strip_html(resp.text)
    logger.info("Fiji Met forecast: received %d chars (stripped)", len(text))

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

    region = _FIJI_MET_REGION.get(location)
    section = None
    if region:
        # Find "For <region> :" and take everything up to the next "For " header
        # or "Outlook for" subheader, whichever ends the per-region section.
        marker = f"For {region}"
        idx = text.lower().find(marker.lower())
        if idx >= 0:
            tail = text[idx:idx + 800]
            # Stop at the next region header. Bulletin uses both spaced and
            # unspaced colon formats — "For Navua / Suva / Nausori :" and
            # "For Natadradave/Korovou:" — and slashes both with and without
            # surrounding spaces, so allow word chars / slashes / dashes /
            # spaces and an optional gap before the colon.
            next_for = re.search(r"\bFor [A-Z][\w/\- ]+?\s*:", tail[len(marker):])
            section = tail[:len(marker) + next_for.start()] if next_for else tail

    # Use the per-region section if we found one; otherwise fall back to whole
    # bulletin so the source still produces *something* if the page layout
    # changes.
    snippet = section or text

    # Extract Min / Max temperatures from the per-region forecast and use the
    # midpoint as a temperature reading. Fiji Met formats them as "Min: 22, Max: 31".
    tmin_match = re.search(r"Min[:\s]+(\d{1,2})", snippet)
    tmax_match = re.search(r"Max[:\s]+(\d{1,2})", snippet)
    if tmin_match and tmax_match:
        tmin = float(tmin_match.group(1))
        tmax = float(tmax_match.group(1))
        if 10 <= tmin <= 35 and 15 <= tmax <= 45 and tmax >= tmin:
            forecast["temperature"] = round((tmin + tmax) / 2, 1)

    # Wind in the bulletin is rare per-region, but check the whole bulletin
    # for a km/h or knots reading.
    wind_match = re.search(r"(\d{1,3})\s*(?:km/?h|knots|kts)", text, re.IGNORECASE)
    if wind_match:
        val = float(wind_match.group(1))
        ctx_end = min(len(text), wind_match.end() + 10)
        if "knot" in text[wind_match.start():ctx_end].lower():
            val *= 1.852  # knots → km/h
        forecast["wind"] = round(val, 1)

    forecast["rain_probability"] = _classify_rain(snippet)

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
            # Windy returns past-3h precipitation in mm, not a probability.
            # Multiplying mm by 20 to fake a percent (the previous behaviour)
            # turned 1 mm of rain into a 20% "probability" and 5 mm into 100%,
            # which dominated the ensemble and biased every forecast toward rain.
            # Real rain probabilities come from Open-Meteo and AccuWeather; expose
            # the raw amount as precip_mm for callers that want it.
            "rain_probability": None,
            "precip_mm": round(float(precip), 2) if precip else None,
            "wind": wind_speed,
            "humidity": humidity[0] if humidity else None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("Windy API error: %s", e)
        return None


# AccuWeather free tier: 50 calls/day total. With 4 cities, even modest traffic
# blows the quota fast — so we cache responses on disk for several hours and
# serve from cache between refreshes. /tmp is the only writable path on Vercel
# serverless functions; we fall back to it if the project CACHE_DIR isn't
# writable (which is the normal case in production).
_ACCUWEATHER_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours -> ~16 calls/day worst case


def _accuweather_cache_path(location: str) -> Path:
    base = Path("/tmp") if not os.access(str(CACHE_DIR), os.W_OK) else CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"accuweather_{location.lower()}.json"


def _read_accuweather_cache(location: str) -> dict | None:
    path = _accuweather_cache_path(location)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            cached = json.load(f)
        age = time.time() - cached.get("_cached_at", 0)
        if age > _ACCUWEATHER_CACHE_TTL_SECONDS:
            return None
        logger.info("AccuWeather cache hit for %s (age %.0fs)", location, age)
        return cached.get("payload")
    except Exception as e:
        logger.warning("AccuWeather cache read failed for %s: %s", location, e)
        return None


def _write_accuweather_cache(location: str, payload: dict) -> None:
    path = _accuweather_cache_path(location)
    try:
        with open(path, "w") as f:
            json.dump({"_cached_at": time.time(), "payload": payload}, f)
    except Exception as e:
        logger.warning("AccuWeather cache write failed for %s: %s", location, e)


def fetch_accuweather_forecast(location: str) -> dict | None:
    """
    Fetch forecast from AccuWeather API.
    Requires API key: https://developer.accuweather.com/

    Free tier is 50 calls/day. Responses are cached on disk for 6 hours
    (see _ACCUWEATHER_CACHE_TTL_SECONDS) so per-location refreshes happen at
    most 4×/day across 4 cities → ~16 calls/day, well under the cap.
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

    cached = _read_accuweather_cache(location)
    if cached is not None:
        # Refresh the timestamp so consumers see "now" while still respecting
        # the upstream forecast period.
        cached["timestamp"] = datetime.utcnow().isoformat()
        return cached

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

        payload = {
            "source": "accuweather",
            "location": location,
            "temperature": round(temp, 1) if temp else None,
            "rain_probability": rain_prob,
            "wind": wind_speed,
            "humidity": None,
            "timestamp": datetime.utcnow().isoformat(),
        }
        _write_accuweather_cache(location, payload)
        return payload
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
            # Wunderground's qpf is mm of forecast precipitation, not a probability.
            # The previous `min(qpf_mm * 10, 100)` mapping turned routine tropical
            # forecasts (1–10 mm) into 10–100% "rain probabilities" that swamped the
            # ensemble. Real probabilities come from Open-Meteo / AccuWeather.
            "rain_probability": None,
            "precip_mm": round(float(precip[0]), 2) if precip else None,
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


def fetch_all_current_forecasts(location: str) -> list[dict]:
    """
    Fetch current forecasts from all 5 sources for a given location.
    Returns list of forecast dicts (only successful fetches).
    """
    fetchers = [
        ("Fiji Met Service", fetch_fiji_met_forecast),
        ("Windy.com", fetch_windy_forecast),
        ("AccuWeather", fetch_accuweather_forecast),
        ("Weather Underground", fetch_wunderground_forecast),
        ("BOM/Metvuw", fetch_bom_metvuw_forecast),
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


def compute_ensemble_average(forecasts: list[dict]) -> dict:
    """
    Compute simple ensemble average from multiple forecast sources.
    """
    if not forecasts:
        return {"temperature": None, "rain_probability": None, "wind": None, "humidity": None, "n_sources": 0}

    temps = [f["temperature"] for f in forecasts if f.get("temperature") is not None]
    rains = [f["rain_probability"] for f in forecasts if f.get("rain_probability") is not None]
    winds = [f["wind"] for f in forecasts if f.get("wind") is not None]
    humidities = [f["humidity"] for f in forecasts if f.get("humidity") is not None]

    return {
        "temperature": round(np.mean(temps), 1) if temps else None,
        "rain_probability": round(np.mean(rains), 1) if rains else None,
        "wind": round(np.mean(winds), 1) if winds else None,
        "humidity": round(np.mean(humidities), 1) if humidities else None,
        "n_sources": len(forecasts),
        "sources": [f["source"] for f in forecasts],
        "timestamp": datetime.utcnow().isoformat(),
    }


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
