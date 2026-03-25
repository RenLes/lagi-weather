"""
Lagi -- Fiji Weather Guardian: Inference Engine
================================================
Full inference pipeline:
  1. Fetch current forecasts from 5 sources (ensemble)
  2. Compute ensemble average
  3. Apply static bias correction (from training)
  4. Apply dynamic correlation-based adjustment (from validation)
  5. Monte Carlo uncertainty intervals
  6. Log prediction for future validation
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from data_fetcher import fetch_all_current_forecasts, compute_ensemble_average
from validation import log_prediction, load_dynamic_coefficients, apply_dynamic_adjustment

logger = logging.getLogger("lagi.inference")

SEASONS = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
ENSO_PHASES = {"Nina": -1, "Neutral": 0, "Nino": 1}
MONTE_CARLO_SAMPLES = 1000

MODEL_DIR = Path(__file__).parent.parent / "output"


class LagiInference:
    """Lagi weather correction inference engine with ensemble + dynamic adjustment."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.coefficients = self._load_coefficients()
        self.error_distributions = self._load_error_distributions()
        self.dynamic_coefficients = load_dynamic_coefficients()

    def _load_coefficients(self) -> dict:
        coeff_path = self.model_dir / "coefficients.json"
        if coeff_path.exists():
            with open(coeff_path) as f:
                return json.load(f)
        return {
            "temp": {"beta_0": 0.5, "beta_1": -0.02, "beta_2": 0.1, "beta_3": 0.05, "r_squared": 0.0},
            "precip": {"gamma_0": 5.0, "gamma_1": -0.05, "gamma_2": 0.02, "gamma_3": -0.1, "r_squared": 0.0},
        }

    def _load_error_distributions(self):
        pkl_path = self.model_dir / "error_distributions.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        return None

    def reload_dynamic_coefficients(self):
        """Reload dynamic coefficients from file (call after validation.compute_correlation)."""
        self.dynamic_coefficients = load_dynamic_coefficients()

    def get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "DJF"
        elif month in [3, 4, 5]:
            return "MAM"
        elif month in [6, 7, 8]:
            return "JJA"
        return "SON"

    def fetch_ensemble_forecast(self, location: str) -> dict:
        """
        Fetch forecasts from all 5 sources and compute ensemble average.
        Sources: Fiji Met, Windy, AccuWeather, Wunderground, BOM/Metvuw
        """
        forecasts = fetch_all_current_forecasts(location)
        ensemble = compute_ensemble_average(forecasts)
        return {
            "individual_forecasts": forecasts,
            "ensemble": ensemble,
        }

    def correct_forecast(
        self,
        location: str,
        T_f: float,
        P_f: float,
        W_f: float,
        humidity_f: float,
        season: str,
        enso_phase: str,
        ensemble_sources: list[str] | None = None,
        ensemble_T: float | None = None,
        ensemble_P: float | None = None,
    ) -> dict:
        """
        Full correction pipeline:
          1. Apply static bias correction (from QLoRA training)
          2. Apply dynamic correlation adjustment (from prediction validation)
          3. Compute Monte Carlo intervals
          4. Log prediction for future validation
        """
        beta = self.coefficients["temp"]
        gamma = self.coefficients["precip"]
        season_code = SEASONS.get(season, 0)
        enso_code = ENSO_PHASES.get(enso_phase, 0)

        # Step 1: Static bias correction (from training)
        delta_t = beta["beta_0"] + beta["beta_1"] * T_f + beta["beta_2"] * season_code + beta["beta_3"] * enso_code
        delta_p = gamma["gamma_0"] + gamma["gamma_1"] * P_f + gamma["gamma_2"] * humidity_f + gamma["gamma_3"] * W_f

        T_static = T_f + delta_t
        P_static = float(sigmoid(P_f + delta_p)) * 100

        # Step 2: Dynamic correlation adjustment (from validation history)
        dynamic = apply_dynamic_adjustment(T_static, P_static, self.dynamic_coefficients)
        T_adj = dynamic["adjusted_T"]
        P_adj = dynamic["adjusted_P"]

        result = {
            "location": location,
            "raw_forecast": {"temperature": T_f, "rain_probability": P_f, "wind": W_f},
            "static_correction": {"temperature": round(T_static, 1), "rain_probability": round(P_static, 1)},
            "adjusted_forecast": {"temperature": T_adj, "rain_probability": P_adj, "wind": W_f},
            "corrections": {
                "static_delta_T": round(delta_t, 2),
                "static_delta_P": round(delta_p, 2),
                "dynamic_method_T": dynamic["temp_method"],
                "dynamic_method_P": dynamic["precip_method"],
                "temp_pearson_r": dynamic["temp_pearson_r"],
                "precip_pearson_r": dynamic["precip_pearson_r"],
                "validation_samples": dynamic["n_samples"],
            },
            "season": season,
            "enso_phase": enso_phase,
        }

        # Step 3: Monte Carlo intervals
        if self.error_distributions:
            mc = self._monte_carlo(T_f, P_f, delta_t, delta_p)
            result["monte_carlo"] = mc
        else:
            result["monte_carlo"] = {
                "T_mean": T_adj,
                "T_sigma": 1.2,
                "T_95_CI": [round(T_adj - 2.4, 1), round(T_adj + 2.4, 1)],
                "P_mean": P_adj,
                "P_95_CI": [max(0, round(P_adj - 15, 1)), min(100, round(P_adj + 15, 1))],
            }

        # Step 4: Log prediction for future validation
        try:
            log_prediction(
                location=location,
                predicted_T=T_adj,
                predicted_P=P_adj,
                predicted_W=W_f,
                ensemble_T=ensemble_T or T_f,
                ensemble_P=ensemble_P or P_f,
                sources=ensemble_sources,
                corrections=result["corrections"],
                monte_carlo=result["monte_carlo"],
            )
        except Exception as e:
            logger.error("Failed to log prediction: %s", e)

        return result

    def _monte_carlo(self, T_f: float, P_f: float, delta_t: float, delta_p: float) -> dict:
        temp_errors = self.error_distributions["temp_gmm"].sample(MONTE_CARLO_SAMPLES)[0].flatten()
        precip_errors = self.error_distributions["precip_gmm"].sample(MONTE_CARLO_SAMPLES)[0].flatten()

        t_samples = T_f + delta_t + temp_errors
        p_samples = sigmoid(P_f + delta_p + precip_errors) * 100

        return {
            "T_mean": round(float(np.mean(t_samples)), 1),
            "T_sigma": round(float(np.std(t_samples)), 2),
            "T_95_CI": [round(float(np.percentile(t_samples, 2.5)), 1), round(float(np.percentile(t_samples, 97.5)), 1)],
            "P_mean": round(float(np.mean(p_samples)), 1),
            "P_95_CI": [round(float(np.percentile(p_samples, 2.5)), 1), round(float(np.percentile(p_samples, 97.5)), 1)],
        }
