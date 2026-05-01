#!/usr/bin/env python3
"""
Lagi -- Fiji Weather Guardian: Performance Tracker
=====================================================
Tracks MAE/RMSE per city over time to detect model drift.
Logs weekly performance metrics and alerts when accuracy degrades.

Run weekly via cron:
    0 9 * * 1 cd /path/to/lagi-weather && python scripts/performance_tracker.py

Or manually:
    python scripts/performance_tracker.py
    python scripts/performance_tracker.py --alert-threshold 3.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, PREDICTIONS_LOG, ACTUALS_LOG, PERFORMANCE_LOG, LOCATIONS

logger = logging.getLogger("lagi.performance")


def compute_weekly_metrics(weeks_back: int = 1) -> dict:
    """
    Compute MAE and RMSE per city for the most recent week.

    Args:
        weeks_back: How many weeks back to compute (1 = last week)

    Returns:
        Dict with per-city and overall metrics.
    """
    if not PREDICTIONS_LOG.exists() or not ACTUALS_LOG.exists():
        logger.warning("No prediction or actual logs found.")
        return {}

    # Load data
    predictions = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    actuals = []
    with open(ACTUALS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                actuals.append(json.loads(line))

    if not predictions or not actuals:
        return {}

    pred_df = pd.DataFrame(predictions)
    actual_df = pd.DataFrame(actuals)

    pred_df["date"] = pd.to_datetime(pred_df["date"])
    actual_df["date"] = pd.to_datetime(actual_df["date"])

    # Filter to the target week
    now = datetime.utcnow()
    week_end = now - timedelta(days=(weeks_back - 1) * 7)
    week_start = week_end - timedelta(days=7)

    pred_df = pred_df[(pred_df["date"] >= week_start) & (pred_df["date"] < week_end)]
    actual_df = actual_df[(actual_df["date"] >= week_start) & (actual_df["date"] < week_end)]

    if pred_df.empty or actual_df.empty:
        logger.info("No data for week %s to %s", week_start.date(), week_end.date())
        return {}

    # Match predictions to actuals
    pred_df["date_str"] = pred_df["date"].dt.strftime("%Y-%m-%d")
    actual_df["date_str"] = actual_df["date"].dt.strftime("%Y-%m-%d")

    merged = pred_df.merge(
        actual_df[["date_str", "location", "T_a", "P_a"]],
        left_on=["date_str", "location"],
        right_on=["date_str", "location"],
        how="inner",
    )

    if merged.empty:
        logger.info("No matched pairs for this week.")
        return {}

    result = {
        "week_start": week_start.strftime("%Y-%m-%d"),
        "week_end": week_end.strftime("%Y-%m-%d"),
        "computed_at": datetime.utcnow().isoformat(),
        "cities": {},
        "overall": {},
    }

    # Per-city metrics
    all_temp_errors = []
    all_precip_errors = []

    for location in LOCATIONS:
        city_data = merged[merged["location"] == location]
        if city_data.empty:
            continue

        temp_errors = (city_data["predicted_T"].astype(float) - city_data["T_a"].astype(float)).dropna()
        precip_errors = (city_data["predicted_P"].astype(float) - city_data["P_a"].astype(float)).dropna()

        city_metrics = {"n_samples": len(city_data)}

        if len(temp_errors) > 0:
            city_metrics["temp_mae"] = round(float(np.mean(np.abs(temp_errors))), 3)
            city_metrics["temp_rmse"] = round(float(np.sqrt(np.mean(temp_errors ** 2))), 3)
            city_metrics["temp_bias"] = round(float(np.mean(temp_errors)), 3)
            all_temp_errors.extend(temp_errors.tolist())

        if len(precip_errors) > 0:
            city_metrics["precip_mae"] = round(float(np.mean(np.abs(precip_errors))), 3)
            city_metrics["precip_rmse"] = round(float(np.sqrt(np.mean(precip_errors ** 2))), 3)
            city_metrics["precip_bias"] = round(float(np.mean(precip_errors)), 3)
            all_precip_errors.extend(precip_errors.tolist())

        result["cities"][location] = city_metrics

    # Overall metrics
    if all_temp_errors:
        arr = np.array(all_temp_errors)
        result["overall"]["temp_mae"] = round(float(np.mean(np.abs(arr))), 3)
        result["overall"]["temp_rmse"] = round(float(np.sqrt(np.mean(arr ** 2))), 3)

    if all_precip_errors:
        arr = np.array(all_precip_errors)
        result["overall"]["precip_mae"] = round(float(np.mean(np.abs(arr))), 3)
        result["overall"]["precip_rmse"] = round(float(np.sqrt(np.mean(arr ** 2))), 3)

    result["overall"]["n_samples"] = len(merged)

    return result


def log_metrics(metrics: dict):
    """Append weekly metrics to the performance log (JSONL)."""
    if not metrics:
        return

    PERFORMANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PERFORMANCE_LOG, "a") as f:
        f.write(json.dumps(metrics) + "\n")
    logger.info("Performance metrics logged to %s", PERFORMANCE_LOG)


def check_drift(alert_threshold: float = 3.0) -> list[str]:
    """
    Check if model accuracy has degraded beyond the alert threshold.

    Compares the most recent week's MAE to the historical average.
    If current MAE exceeds historical average by more than alert_threshold,
    returns a list of alert messages.

    Args:
        alert_threshold: Degrees C above historical average MAE to trigger alert

    Returns:
        List of alert message strings (empty if no drift detected).
    """
    if not PERFORMANCE_LOG.exists():
        return []

    records = []
    with open(PERFORMANCE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) < 3:
        return []  # Need at least 3 weeks of data

    # Compare latest to historical average
    latest = records[-1]
    historical = records[:-1]

    alerts = []

    latest_temp_mae = latest.get("overall", {}).get("temp_mae")
    if latest_temp_mae is not None:
        hist_maes = [r["overall"]["temp_mae"] for r in historical
                     if r.get("overall", {}).get("temp_mae") is not None]
        if hist_maes:
            hist_avg = float(np.mean(hist_maes))
            if latest_temp_mae > hist_avg + alert_threshold:
                alerts.append(
                    f"DRIFT ALERT: Temperature MAE ({latest_temp_mae:.2f}°C) exceeds "
                    f"historical average ({hist_avg:.2f}°C) by {latest_temp_mae - hist_avg:.2f}°C. "
                    f"Consider retraining."
                )

    latest_precip_mae = latest.get("overall", {}).get("precip_mae")
    if latest_precip_mae is not None:
        hist_maes = [r["overall"]["precip_mae"] for r in historical
                     if r.get("overall", {}).get("precip_mae") is not None]
        if hist_maes:
            hist_avg = float(np.mean(hist_maes))
            if latest_precip_mae > hist_avg + 10.0:  # 10% threshold for precip
                alerts.append(
                    f"DRIFT ALERT: Precipitation MAE ({latest_precip_mae:.1f}%) exceeds "
                    f"historical average ({hist_avg:.1f}%) by {latest_precip_mae - hist_avg:.1f}%. "
                    f"Consider retraining."
                )

    # Per-city checks
    for city, city_metrics in latest.get("cities", {}).items():
        city_temp_mae = city_metrics.get("temp_mae")
        if city_temp_mae is not None and city_temp_mae > alert_threshold * 2:
            alerts.append(
                f"CITY ALERT: {city} temperature MAE is {city_temp_mae:.2f}°C "
                f"(threshold: {alert_threshold * 2:.1f}°C)"
            )

    return alerts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Lagi Performance Tracker")
    parser.add_argument("--weeks-back", type=int, default=1, help="How many weeks back to compute")
    parser.add_argument("--alert-threshold", type=float, default=3.0,
                        help="Degrees C above historical average to trigger alert")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Lagi Performance Tracker")
    logger.info("=" * 50)

    metrics = compute_weekly_metrics(args.weeks_back)
    if metrics:
        log_metrics(metrics)

        print(f"\nWeek: {metrics['week_start']} to {metrics['week_end']}")
        print(f"Samples: {metrics['overall'].get('n_samples', 0)}")

        if metrics["overall"].get("temp_mae") is not None:
            print(f"Overall Temp MAE:    {metrics['overall']['temp_mae']:.3f}°C")
            print(f"Overall Temp RMSE:   {metrics['overall']['temp_rmse']:.3f}°C")

        if metrics["overall"].get("precip_mae") is not None:
            print(f"Overall Precip MAE:  {metrics['overall']['precip_mae']:.1f}%")
            print(f"Overall Precip RMSE: {metrics['overall']['precip_rmse']:.1f}%")

        for city, cm in metrics["cities"].items():
            print(f"\n  {city}: {cm.get('n_samples', 0)} samples, "
                  f"T_MAE={cm.get('temp_mae', 'N/A')}, P_MAE={cm.get('precip_mae', 'N/A')}")

        # Check for drift
        alerts = check_drift(args.alert_threshold)
        if alerts:
            print("\n" + "!" * 50)
            for alert in alerts:
                print(f"  {alert}")
            print("!" * 50)
        else:
            print("\nNo drift detected.")
    else:
        print("Not enough data for performance metrics.")
        print("Collect actuals first: python scripts/collect_actuals.py --recompute")
