#!/usr/bin/env python3
from __future__ import annotations
"""
Lagi -- Fiji Weather Guardian: Automated Actuals Collection
=============================================================
Fetches yesterday's actual weather observations from Open-Meteo
(free, no API key) and logs them for validation against predictions.

Run daily via cron:
    0 8 * * * cd /path/to/lagi-weather && python scripts/collect_actuals.py

Or manually:
    python scripts/collect_actuals.py
    python scripts/collect_actuals.py --date 2026-04-01
    python scripts/collect_actuals.py --recompute
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOCATIONS, HTTP_TIMEOUT
from validation import log_actual, compute_correlation

logger = logging.getLogger("lagi.collect_actuals")


def fetch_actual_from_openmeteo(location: str, date: str) -> dict | None:
    """
    Fetch actual observed weather for a location and date from Open-Meteo historical API.
    Free, no API key required.

    Args:
        location: City name (must be in LOCATIONS)
        date: Date string in YYYY-MM-DD format

    Returns:
        Dict with T_a, P_a, W_a or None if fetch failed.
    """
    loc = LOCATIONS.get(location)
    if not loc:
        logger.warning("Unknown location: %s", location)
        return None

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": date,
        "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max",
        "timezone": "Pacific/Fiji",
    }

    try:
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Open-Meteo archive fetch failed for %s on %s: %s", location, date, e)
        return None

    daily = data.get("daily", {})
    t_max = daily.get("temperature_2m_max", [None])[0]
    t_min = daily.get("temperature_2m_min", [None])[0]
    precip_mm = daily.get("precipitation_sum", [None])[0]
    wind = daily.get("wind_speed_10m_max", [None])[0]

    if t_max is None or t_min is None:
        logger.warning("No temperature data for %s on %s", location, date)
        return None

    t_avg = round((t_max + t_min) / 2, 1)

    # Convert precipitation mm to a rain probability indicator (0-100)
    # 0mm = 0%, 1mm = ~30%, 5mm = ~70%, 10mm+ = ~90%
    if precip_mm is not None:
        if precip_mm <= 0:
            rain_pct = 0.0
        elif precip_mm < 1:
            rain_pct = round(precip_mm * 30, 1)
        elif precip_mm < 5:
            rain_pct = round(30 + (precip_mm - 1) * 10, 1)
        elif precip_mm < 10:
            rain_pct = round(70 + (precip_mm - 5) * 4, 1)
        else:
            rain_pct = 95.0
    else:
        rain_pct = None

    return {
        "T_a": t_avg,
        "P_a": rain_pct,
        "W_a": round(wind, 1) if wind is not None else None,
    }


def collect_all(date: str, recompute: bool = False) -> dict:
    """
    Collect actual observations for all locations for a given date.

    Args:
        date: Date string YYYY-MM-DD
        recompute: If True, recompute dynamic coefficients after logging

    Returns:
        Summary dict with results per location.
    """
    results = {}

    for location in LOCATIONS:
        logger.info("Fetching actuals for %s on %s...", location, date)
        actual = fetch_actual_from_openmeteo(location, date)

        if actual:
            log_actual(
                date=date,
                location=location,
                T_a=actual["T_a"],
                P_a=actual["P_a"],
                W_a=actual["W_a"],
                source="open_meteo_archive",
            )
            results[location] = {"status": "logged", **actual}
            logger.info("  %s: T=%.1f, P=%s, W=%s",
                        location, actual["T_a"], actual["P_a"], actual["W_a"])
        else:
            results[location] = {"status": "failed"}
            logger.warning("  %s: no data available", location)

    if recompute and any(r["status"] == "logged" for r in results.values()):
        logger.info("Recomputing dynamic coefficients...")
        coefficients = compute_correlation()
        results["_coefficients"] = coefficients
        logger.info("Dynamic coefficients updated: temp_r=%s, precip_r=%s",
                     coefficients.get("temp_pearson_r"), coefficients.get("precip_pearson_r"))

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect actual weather observations for Lagi validation")
    parser.add_argument("--date", help="Date to collect (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute dynamic coefficients after logging actuals")
    args = parser.parse_args()

    target_date = args.date or (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("=" * 50)
    logger.info("Lagi Actuals Collection — %s", target_date)
    logger.info("=" * 50)

    results = collect_all(target_date, recompute=args.recompute)

    logged = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "logged")
    logger.info("Done: %d/%d locations logged for %s", logged, len(LOCATIONS), target_date)
