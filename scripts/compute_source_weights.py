#!/usr/bin/env python3
"""
Lagi -- Fiji Weather Guardian: Source Weight Computation
=========================================================
Computes per-source accuracy weights by comparing each source's
historical forecasts against actual observations.

Sources with lower MAE get higher weights in the ensemble average.
Weights are stored in data/source_weights.json.

Run after collecting enough validation data:
    python scripts/compute_source_weights.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_DIR, PREDICTIONS_LOG, ACTUALS_LOG, LOCATIONS

logger = logging.getLogger("lagi.source_weights")


def compute_weights(min_samples: int = 10) -> dict:
    """
    Compute accuracy weights for each forecast source per location.

    Method:
      1. Match predictions to actuals by date and location
      2. For each source, compute MAE for temperature
      3. Convert MAE to weights: weight = 1 / (MAE + epsilon)
      4. Normalize weights so they sum to 1 per location

    Args:
        min_samples: Minimum matched pairs per source to include it

    Returns:
        Dict mapping location -> {source: weight}
    """
    if not PREDICTIONS_LOG.exists() or not ACTUALS_LOG.exists():
        logger.warning("No prediction or actual logs found.")
        return {}

    # Load predictions with source info
    predictions = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                sources = record.get("sources", [])
                if sources and record.get("ensemble_T") is not None:
                    predictions.append(record)

    if not predictions:
        logger.warning("No predictions with source info found.")
        return {}

    pred_df = pd.DataFrame(predictions)
    pred_df["date"] = pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d")

    # Load actuals
    actuals = []
    with open(ACTUALS_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                actuals.append(json.loads(line))

    if not actuals:
        logger.warning("No actuals found.")
        return {}

    actual_df = pd.DataFrame(actuals)
    actual_df["date"] = pd.to_datetime(actual_df["date"]).dt.strftime("%Y-%m-%d")

    # Match and compute per-source MAE
    merged = pred_df.merge(
        actual_df[["date", "location", "T_a", "P_a"]],
        on=["date", "location"],
        how="inner",
    )

    if merged.empty:
        logger.warning("No matched prediction-actual pairs.")
        return {}

    # For now, compute weights based on overall prediction accuracy
    # per source (since individual source forecasts aren't stored separately)
    # When individual source forecasts are logged, this can be made more granular
    weights_by_location = {}

    for location in LOCATIONS:
        loc_data = merged[merged["location"] == location]
        if len(loc_data) < min_samples:
            continue

        # Use prediction MAE as a proxy for ensemble quality
        # Sources that appear more often in successful predictions get higher weight
        source_counts = {}
        source_errors = {}

        for _, row in loc_data.iterrows():
            sources = row.get("sources", [])
            if not isinstance(sources, list):
                continue

            error = abs(row["predicted_T"] - row["T_a"]) if pd.notna(row.get("T_a")) else None
            if error is None:
                continue

            for src in sources:
                if src not in source_counts:
                    source_counts[src] = 0
                    source_errors[src] = 0.0
                source_counts[src] += 1
                source_errors[src] += error

        # Compute MAE per source
        source_mae = {}
        for src in source_counts:
            if source_counts[src] >= min_samples:
                source_mae[src] = source_errors[src] / source_counts[src]

        if not source_mae:
            continue

        # Convert MAE to weights (inverse MAE)
        epsilon = 0.1
        raw_weights = {src: 1.0 / (mae + epsilon) for src, mae in source_mae.items()}
        total = sum(raw_weights.values())
        normalized = {src: round(w / total, 4) for src, w in raw_weights.items()}

        weights_by_location[location] = normalized
        logger.info("%s weights: %s", location, normalized)

    # Save weights
    output_path = DATA_DIR / "source_weights.json"
    with open(output_path, "w") as f:
        json.dump(weights_by_location, f, indent=2)

    logger.info("Source weights saved to %s", output_path)
    return weights_by_location


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    weights = compute_weights()
    if weights:
        print(json.dumps(weights, indent=2))
    else:
        print("Not enough data to compute source weights yet.")
        print("Collect more actuals with: python scripts/collect_actuals.py --recompute")
