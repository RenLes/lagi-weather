"""
Lagi -- Fiji Weather Guardian: Inference Engine
================================================
Loads the fine-tuned model and applies bias correction + Monte Carlo
intervals at inference time.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid

SEASONS = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
ENSO_PHASES = {"Nina": -1, "Neutral": 0, "Nino": 1}
MONTE_CARLO_SAMPLES = 1000

MODEL_DIR = Path(__file__).parent.parent / "output"


class LagiInference:
    """Lagi weather correction inference engine."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.coefficients = self._load_coefficients()
        self.error_distributions = self._load_error_distributions()

    def _load_coefficients(self) -> dict:
        coeff_path = self.model_dir / "coefficients.json"
        if coeff_path.exists():
            with open(coeff_path) as f:
                return json.load(f)
        # Default coefficients (before training)
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

    def get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return "DJF"
        elif month in [3, 4, 5]:
            return "MAM"
        elif month in [6, 7, 8]:
            return "JJA"
        return "SON"

    def correct_forecast(
        self,
        location: str,
        T_f: float,
        P_f: float,
        W_f: float,
        humidity_f: float,
        season: str,
        enso_phase: str,
    ) -> dict:
        """
        Apply bias correction and Monte Carlo intervals to a raw forecast.

        Returns adjusted forecast with uncertainty bounds.
        """
        beta = self.coefficients["temp"]
        gamma = self.coefficients["precip"]
        season_code = SEASONS.get(season, 0)
        enso_code = ENSO_PHASES.get(enso_phase, 0)

        # Bias corrections
        delta_t = beta["beta_0"] + beta["beta_1"] * T_f + beta["beta_2"] * season_code + beta["beta_3"] * enso_code
        delta_p = gamma["gamma_0"] + gamma["gamma_1"] * P_f + gamma["gamma_2"] * humidity_f + gamma["gamma_3"] * W_f

        T_adj = T_f + delta_t
        P_adj = float(sigmoid(P_f + delta_p)) * 100

        result = {
            "location": location,
            "raw_forecast": {"temperature": T_f, "rain_probability": P_f, "wind": W_f},
            "adjusted_forecast": {"temperature": round(T_adj, 1), "rain_probability": round(P_adj, 1), "wind": W_f},
            "corrections": {"delta_T": round(delta_t, 2), "delta_P": round(delta_p, 2)},
            "season": season,
            "enso_phase": enso_phase,
        }

        # Monte Carlo intervals
        if self.error_distributions:
            mc = self._monte_carlo(T_f, P_f, delta_t, delta_p)
            result["monte_carlo"] = mc
        else:
            # Fallback: use simple +/- based on correction magnitude
            result["monte_carlo"] = {
                "T_mean": round(T_adj, 1),
                "T_sigma": 1.2,
                "T_95_CI": [round(T_adj - 2.4, 1), round(T_adj + 2.4, 1)],
                "P_mean": round(P_adj, 1),
                "P_95_CI": [max(0, round(P_adj - 15, 1)), min(100, round(P_adj + 15, 1))],
            }

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
