#!/usr/bin/env python3
from __future__ import annotations
"""
Lagi -- Fiji Weather Guardian: Prediction Validation Module
=============================================================
Stores every Lagi prediction alongside actual observed data.
Computes Pearson r correlation between predictions and actuals.
Produces a dynamic adjustment coefficient for the ensemble.

Prediction Log Format (predictions_log.jsonl):
  {"timestamp": "...", "location": "...", "predicted_T": 28.5, "predicted_P": 65.0,
   "predicted_W": 12.0, "ensemble_T": 28.0, "ensemble_P": 60.0, "sources": [...]}

Actuals Log Format (actuals_log.jsonl):
  {"date": "2026-03-25", "location": "Suva", "T_a": 29.1, "P_a": 70.0, "W_a": 14.0,
   "source": "NOAA_GHCN"}

Coefficient Output (dynamic_coefficients.json):
  {"temp_pearson_r": 0.87, "temp_bias": 0.5, "precip_pearson_r": 0.72,
   "precip_bias": 3.2, "last_updated": "...", "n_samples": 150}
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, OUTPUT_DIR, PREDICTIONS_LOG, ACTUALS_LOG,
    DYNAMIC_COEFFICIENTS_FILE, MIN_CORRELATION_SAMPLES, DYNAMIC_R_THRESHOLD,
)

logger = logging.getLogger("lagi.validation")


# ---------------------------------------------------------------------------
# Prediction Storage
# ---------------------------------------------------------------------------
def log_prediction(
    location: str,
    predicted_T: float,
    predicted_P: float,
    predicted_W: float,
    ensemble_T: float | None = None,
    ensemble_P: float | None = None,
    ensemble_W: float | None = None,
    sources: list[str] | None = None,
    corrections: dict | None = None,
    monte_carlo: dict | None = None,
):
    """
    Log a Lagi-generated prediction for later validation against actuals.
    Appends to predictions_log.jsonl.
    """
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "location": location,
        "predicted_T": predicted_T,
        "predicted_P": predicted_P,
        "predicted_W": predicted_W,
        "ensemble_T": ensemble_T,
        "ensemble_P": ensemble_P,
        "ensemble_W": ensemble_W,
        "sources": sources or [],
        "corrections_applied": corrections,
        "monte_carlo": monte_carlo,
    }

    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Logged prediction for %s: T=%.1f, P=%.1f, W=%.1f", location, predicted_T, predicted_P, predicted_W)


def log_actual(
    date: str,
    location: str,
    T_a: float,
    P_a: float,
    W_a: float | None = None,
    source: str = "manual",
):
    """
    Log actual observed weather data for validation against predictions.
    Appends to actuals_log.jsonl.
    """
    ACTUALS_LOG.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "date": date,
        "location": location,
        "T_a": T_a,
        "P_a": P_a,
        "W_a": W_a,
        "source": source,
    }

    with open(ACTUALS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Logged actual for %s on %s: T=%.1f, P=%.1f", location, date, T_a, P_a)


# ---------------------------------------------------------------------------
# Load Logs
# ---------------------------------------------------------------------------
def load_predictions() -> pd.DataFrame:
    """Load all logged predictions into a DataFrame."""
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame()

    records = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return pd.DataFrame(records)


def load_actuals() -> pd.DataFrame:
    """Load all logged actuals into a DataFrame."""
    if not ACTUALS_LOG.exists():
        return pd.DataFrame()

    records = []
    with open(ACTUALS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Correlation & Coefficient Computation
# ---------------------------------------------------------------------------
def compute_correlation() -> dict:
    """
    Compute Pearson r correlation between Lagi predictions and actuals.
    Also computes bias offset for dynamic adjustment.

    Returns dict with correlation coefficients and bias values.
    """
    predictions = load_predictions()
    actuals = load_actuals()

    if predictions.empty or actuals.empty:
        logger.warning("Not enough data for correlation. Predictions: %d, Actuals: %d",
                        len(predictions), len(actuals))
        return _default_coefficients()

    # Match predictions to actuals by date and location
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.strftime("%Y-%m-%d")
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")

    merged = predictions.merge(
        actuals[["date", "location", "T_a", "P_a", "W_a"]],
        on=["date", "location"],
        how="inner",
    )

    if len(merged) < MIN_CORRELATION_SAMPLES:
        logger.warning("Only %d matched prediction-actual pairs. Need at least %d for correlation.",
                        len(merged), MIN_CORRELATION_SAMPLES)
        return _default_coefficients()

    result = {"last_updated": datetime.utcnow().isoformat(), "n_samples": len(merged)}

    # Temperature correlation
    temp_mask = merged["predicted_T"].notna() & merged["T_a"].notna()
    if temp_mask.sum() >= MIN_CORRELATION_SAMPLES:
        pred_t = merged.loc[temp_mask, "predicted_T"].astype(float)
        actual_t = merged.loc[temp_mask, "T_a"].astype(float)

        r, p_value = stats.pearsonr(pred_t, actual_t)
        bias = float(np.mean(actual_t - pred_t))

        # Linear regression for slope-based adjustment
        slope, intercept, r_value, _, std_err = stats.linregress(pred_t, actual_t)

        result["temp_pearson_r"] = round(float(r), 4)
        result["temp_p_value"] = round(float(p_value), 6)
        result["temp_bias"] = round(bias, 3)
        result["temp_slope"] = round(float(slope), 4)
        result["temp_intercept"] = round(float(intercept), 4)
        result["temp_std_err"] = round(float(std_err), 4)
        result["temp_r_squared"] = round(float(r_value ** 2), 4)
        result["temp_n"] = int(temp_mask.sum())

        logger.info("Temperature: Pearson r=%.4f (p=%.6f), bias=%.3f, slope=%.4f, n=%d",
                     r, p_value, bias, slope, temp_mask.sum())
    else:
        result.update({"temp_pearson_r": None, "temp_bias": 0.0, "temp_slope": 1.0, "temp_intercept": 0.0})

    # Precipitation correlation
    precip_mask = merged["predicted_P"].notna() & merged["P_a"].notna()
    if precip_mask.sum() >= MIN_CORRELATION_SAMPLES:
        pred_p = merged.loc[precip_mask, "predicted_P"].astype(float)
        actual_p = merged.loc[precip_mask, "P_a"].astype(float)

        r, p_value = stats.pearsonr(pred_p, actual_p)
        bias = float(np.mean(actual_p - pred_p))
        slope, intercept, r_value, _, std_err = stats.linregress(pred_p, actual_p)

        result["precip_pearson_r"] = round(float(r), 4)
        result["precip_p_value"] = round(float(p_value), 6)
        result["precip_bias"] = round(bias, 3)
        result["precip_slope"] = round(float(slope), 4)
        result["precip_intercept"] = round(float(intercept), 4)
        result["precip_std_err"] = round(float(std_err), 4)
        result["precip_r_squared"] = round(float(r_value ** 2), 4)
        result["precip_n"] = int(precip_mask.sum())

        logger.info("Precipitation: Pearson r=%.4f (p=%.6f), bias=%.3f, slope=%.4f, n=%d",
                     r, p_value, bias, slope, precip_mask.sum())
    else:
        result.update({"precip_pearson_r": None, "precip_bias": 0.0, "precip_slope": 1.0, "precip_intercept": 0.0})

    # Save coefficients
    DYNAMIC_COEFFICIENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DYNAMIC_COEFFICIENTS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Dynamic coefficients saved to %s", DYNAMIC_COEFFICIENTS_FILE)
    return result


def _default_coefficients() -> dict:
    """Return default coefficients when not enough data exists."""
    return {
        "temp_pearson_r": None,
        "temp_bias": 0.0,
        "temp_slope": 1.0,
        "temp_intercept": 0.0,
        "precip_pearson_r": None,
        "precip_bias": 0.0,
        "precip_slope": 1.0,
        "precip_intercept": 0.0,
        "n_samples": 0,
        "last_updated": datetime.utcnow().isoformat(),
    }


def load_dynamic_coefficients() -> dict:
    """Load the most recent dynamic coefficients from file."""
    if DYNAMIC_COEFFICIENTS_FILE.exists():
        with open(DYNAMIC_COEFFICIENTS_FILE) as f:
            return json.load(f)
    return _default_coefficients()


# ---------------------------------------------------------------------------
# Dynamic Adjustment
# ---------------------------------------------------------------------------
def apply_dynamic_adjustment(
    ensemble_T: float,
    ensemble_P: float,
    coefficients: dict | None = None,
) -> dict:
    """
    Apply dynamic correlation-based adjustment to ensemble forecast.

    Method:
      final_T = ensemble_T * slope + intercept  (linear regression correction)
      OR simpler: final_T = ensemble_T + bias_offset

      Uses slope/intercept when Pearson r is strong (>0.5),
      falls back to simple bias offset otherwise.
    """
    if coefficients is None:
        coefficients = load_dynamic_coefficients()

    # Temperature adjustment
    temp_r = coefficients.get("temp_pearson_r")
    if temp_r is not None and abs(temp_r) > DYNAMIC_R_THRESHOLD:
        # Strong correlation: use regression-based correction
        adjusted_T = ensemble_T * coefficients["temp_slope"] + coefficients["temp_intercept"]
        method_T = "regression"
    else:
        # Weak or no correlation: use simple bias offset
        adjusted_T = ensemble_T + coefficients.get("temp_bias", 0.0)
        method_T = "bias_offset"

    # Precipitation adjustment
    precip_r = coefficients.get("precip_pearson_r")
    if precip_r is not None and abs(precip_r) > DYNAMIC_R_THRESHOLD:
        adjusted_P = ensemble_P * coefficients["precip_slope"] + coefficients["precip_intercept"]
        method_P = "regression"
    else:
        adjusted_P = ensemble_P + coefficients.get("precip_bias", 0.0)
        method_P = "bias_offset"

    # Clamp values
    adjusted_P = max(0.0, min(100.0, adjusted_P))

    return {
        "adjusted_T": round(adjusted_T, 1),
        "adjusted_P": round(adjusted_P, 1),
        "temp_method": method_T,
        "precip_method": method_P,
        "temp_pearson_r": temp_r,
        "precip_pearson_r": precip_r,
        "n_samples": coefficients.get("n_samples", 0),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def generate_validation_report() -> str:
    """Generate a human-readable validation report."""
    coefficients = load_dynamic_coefficients()
    predictions = load_predictions()
    actuals = load_actuals()

    report = [
        "=" * 60,
        "  LAGI VALIDATION REPORT",
        "=" * 60,
        f"  Generated: {datetime.utcnow().isoformat()}",
        f"  Total predictions logged: {len(predictions)}",
        f"  Total actuals logged: {len(actuals)}",
        f"  Matched pairs (n_samples): {coefficients.get('n_samples', 0)}",
        "",
        "  TEMPERATURE CORRELATION",
        f"    Pearson r:     {coefficients.get('temp_pearson_r', 'N/A')}",
        f"    R-squared:     {coefficients.get('temp_r_squared', 'N/A')}",
        f"    Bias (offset): {coefficients.get('temp_bias', 'N/A')}",
        f"    Slope:         {coefficients.get('temp_slope', 'N/A')}",
        f"    Intercept:     {coefficients.get('temp_intercept', 'N/A')}",
        "",
        "  PRECIPITATION CORRELATION",
        f"    Pearson r:     {coefficients.get('precip_pearson_r', 'N/A')}",
        f"    R-squared:     {coefficients.get('precip_r_squared', 'N/A')}",
        f"    Bias (offset): {coefficients.get('precip_bias', 'N/A')}",
        f"    Slope:         {coefficients.get('precip_slope', 'N/A')}",
        f"    Intercept:     {coefficients.get('precip_intercept', 'N/A')}",
        "",
        f"  Last updated: {coefficients.get('last_updated', 'Never')}",
        "=" * 60,
    ]

    return "\n".join(report)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Lagi Validation Module")
    parser.add_argument("--action", choices=["compute", "report", "log-actual"], default="report")
    parser.add_argument("--date", help="Date for actual observation (YYYY-MM-DD)")
    parser.add_argument("--location", help="Location name")
    parser.add_argument("--temp", type=float, help="Actual temperature")
    parser.add_argument("--rain", type=float, help="Actual rain probability")
    parser.add_argument("--wind", type=float, help="Actual wind speed")
    args = parser.parse_args()

    if args.action == "compute":
        result = compute_correlation()
        print(json.dumps(result, indent=2))
    elif args.action == "report":
        print(generate_validation_report())
    elif args.action == "log-actual":
        if not all([args.date, args.location, args.temp is not None, args.rain is not None]):
            print("Error: --date, --location, --temp, and --rain are required for log-actual")
        else:
            log_actual(args.date, args.location, args.temp, args.rain, args.wind)
            print(f"Logged actual for {args.location} on {args.date}")
