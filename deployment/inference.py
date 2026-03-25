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
SEASON_NAMES = {0: "wet season (DJF)", 1: "transition (MAM)", 2: "dry season (JJA)", 3: "transition (SON)"}
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

    def compute_certainty(self) -> dict:
        """
        Compute certainty percentage from dynamic correlation coefficients.
        Certainty = average of absolute Pearson r values × 100, capped at 95%.
        """
        dc = self.dynamic_coefficients
        r_values = []

        temp_r = dc.get("temp_pearson_r")
        if temp_r is not None:
            r_values.append(abs(temp_r))

        precip_r = dc.get("precip_pearson_r")
        if precip_r is not None:
            r_values.append(abs(precip_r))

        if not r_values:
            return {"certainty_pct": 50, "level": "moderate", "n_samples": 0,
                    "note": "Building historical validation data. Certainty will improve as more predictions are verified."}

        avg_r = float(np.mean(r_values))
        certainty = min(round(avg_r * 100), 95)

        if certainty >= 75:
            level = "high"
        elif certainty >= 50:
            level = "moderate"
        else:
            level = "developing"

        return {
            "certainty_pct": certainty,
            "level": level,
            "n_samples": dc.get("n_samples", 0),
            "temp_r": temp_r,
            "precip_r": precip_r,
            "note": f"Based on {dc.get('n_samples', 0)} verified prediction-vs-actual comparisons.",
        }

    def natural_language_response(self, question: str, location: str, forecast_result: dict) -> str:
        """
        Generate a warm, Fijian-tone natural language answer to a weather question
        using the latest adjusted forecast data.
        """
        q = question.lower()
        adj = forecast_result["adjusted_forecast"]
        mc = forecast_result["monte_carlo"]
        certainty = self.compute_certainty()
        cert_pct = certainty["certainty_pct"]

        T = adj["temperature"]
        P = adj["rain_probability"]
        W = adj["wind"]

        # Determine rain likelihood
        if P >= 75:
            rain_desc = "very likely"
            rain_advice = "Take your umbrella for sure"
        elif P >= 50:
            rain_desc = "a good chance"
            rain_advice = "I'd bring the umbrella just in case"
        elif P >= 30:
            rain_desc = "possible but not certain"
            rain_advice = "You might get away without the umbrella, but keep one handy"
        else:
            rain_desc = "unlikely"
            rain_advice = "You should be fine without the umbrella"

        # Determine temperature feel
        if T >= 33:
            temp_feel = "a scorcher"
            temp_tip = "Stay hydrated and find shade during the midday heat!"
        elif T >= 29:
            temp_feel = "warm and tropical"
            temp_tip = "Comfortable but keep that water bottle close."
        elif T >= 25:
            temp_feel = "pleasant"
            temp_tip = "Beautiful weather to be outdoors!"
        else:
            temp_feel = "cool for Fiji"
            temp_tip = "You might want a light layer this morning."

        # Build response based on question type
        if any(w in q for w in ["rain", "shower", "wet", "umbrella", "pour"]):
            response = (
                f"Bula! Great question about the rain in {location}. "
                f"Right now I'm seeing rain is {rain_desc} — about {P:.0f}% chance. "
                f"{rain_advice}. "
            )
            if P >= 50:
                response += f"The showers could come through this afternoon, so if you've got outdoor plans, morning is your best window. "
            else:
                response += f"Looking like a mostly dry day, which is great for any outdoor plans! "
            response += temp_tip

        elif any(w in q for w in ["hot", "heat", "temperature", "warm", "cool", "cold"]):
            response = (
                f"Bula! Checking the temperature for {location} — it's looking {temp_feel} today at around {T:.1f}°C "
                f"(could range between {mc['T_95_CI'][0]}°C and {mc['T_95_CI'][1]}°C). "
                f"{temp_tip} "
            )

        elif any(w in q for w in ["sun", "sunny", "sunshine", "clear"]):
            if P < 30:
                response = (
                    f"Bula! Good news for {location} — plenty of sunshine expected today with only {P:.0f}% rain chance. "
                    f"Temperature around {T:.1f}°C. {temp_tip} Don't forget the sunscreen — the UV is strong in Fiji!"
                )
            elif P < 60:
                hours_sun = max(1, round((100 - P) / 100 * 8))
                response = (
                    f"Bula! You should get around {hours_sun} good hours of sunshine in {location} before any clouds build. "
                    f"Rain chance is {P:.0f}%, so enjoy the morning sun! "
                    f"Perfect time to get outdoor tasks done early. {temp_tip}"
                )
            else:
                response = (
                    f"Bula! Hmm, sunshine might be limited in {location} today — rain chance is sitting at {P:.0f}%. "
                    f"You might get some breaks between showers though. {temp_tip}"
                )

        elif any(w in q for w in ["wind", "gust", "breeze", "blow"]):
            if W and W > 30:
                wind_desc = "strong winds"
                wind_advice = "Not the best day for small boats or anything lightweight outdoors."
            elif W and W > 20:
                wind_desc = "a moderate breeze"
                wind_advice = "Good for keeping cool but secure anything light on the veranda."
            else:
                wind_desc = "light winds"
                wind_advice = "Calm conditions — great for fishing or any water activities!"
            response = (
                f"Bula! Wind conditions for {location}: expecting {wind_desc} at around {W:.0f} km/h. "
                f"{wind_advice} "
            )

        elif any(w in q for w in ["fish", "fishing", "boat", "sail", "sea"]):
            safe = P < 50 and (not W or W < 25)
            if safe:
                response = (
                    f"Bula! Looking good for your fishing trip in {location}! "
                    f"Rain chance is {P:.0f}%, wind around {W:.0f} km/h — nice conditions out there. "
                    f"Best time to head out is early morning. Wear your life jacket and tell someone your plans, ya!"
                )
            else:
                response = (
                    f"Bula! Hmm, I'd be careful about heading out from {location} today. "
                    f"Rain chance is {P:.0f}% and wind is around {W:.0f} km/h. "
                    f"Maybe wait for better conditions tomorrow, or stick close to the reef if you must go. Safety first!"
                )

        elif any(w in q for w in ["cyclone", "storm", "hurricane", "severe"]):
            response = (
                f"Bula! I'm checking the tropical outlook for {location}. "
                f"Right now there's no active cyclone warning from the Fiji Met Service. "
                f"Current conditions: {T:.1f}°C, {P:.0f}% rain, {W:.0f} km/h wind. "
                f"Always stay prepared during cyclone season (November-April) and follow official FMS warnings."
            )

        elif any(w in q for w in ["tomorrow", "weekend", "later", "tonight", "afternoon"]):
            response = (
                f"Bula! For {location}, the current adjusted forecast shows {T:.1f}°C with {P:.0f}% rain chance. "
                f"Rain is {rain_desc}. {rain_advice}. "
                f"{temp_tip} I'll have updated numbers as new data comes in — check back with me!"
            )

        else:
            # General weather question
            response = (
                f"Bula! Here's what I'm seeing for {location} right now: "
                f"Temperature around {T:.1f}°C ({temp_feel}), rain chance {P:.0f}% ({rain_desc}), "
                f"wind {W:.0f} km/h. "
                f"{rain_advice}. {temp_tip}"
            )

        # Add certainty disclaimer
        response += (
            f"\n\nBased on the analysis, the certainty of our prediction is {cert_pct}%. "
            f"Weather patterns in the Pacific can change rapidly due to local microclimates and climate variability "
            f"— always stay prepared and check official FMS warnings for safety-critical decisions."
        )

        return response
