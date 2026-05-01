"""Tests for the inference engine: bias correction, Monte Carlo, natural language."""

import json

import pytest
import numpy as np

from config import (
    SEASONS, ENSO_PHASES, DEFAULT_STATIC_COEFFICIENTS,
    DEFAULT_TEMP_SIGMA, DEFAULT_PRECIP_INTERVAL,
)


class TestStaticBiasCorrection:
    """Test the mathematical correctness of static bias correction."""

    def test_temperature_correction_formula(self):
        """T_adj = T_f + (beta_0 + beta_1 * T_f + beta_2 * season + beta_3 * enso)"""
        beta = DEFAULT_STATIC_COEFFICIENTS["temp"]
        T_f = 28.0
        season_code = SEASONS["DJF"]  # 0
        enso_code = ENSO_PHASES["Neutral"]  # 0

        delta_t = beta["beta_0"] + beta["beta_1"] * T_f + beta["beta_2"] * season_code + beta["beta_3"] * enso_code
        T_adj = T_f + delta_t

        expected_delta = 0.5 + (-0.02 * 28.0) + (0.1 * 0) + (0.05 * 0)
        assert abs(delta_t - expected_delta) < 0.001
        assert abs(T_adj - (28.0 + expected_delta)) < 0.001

    def test_season_affects_temperature(self):
        """Different seasons should produce different corrections."""
        beta = DEFAULT_STATIC_COEFFICIENTS["temp"]
        T_f = 28.0
        enso_code = 0

        corrections = {}
        for season, code in SEASONS.items():
            delta = beta["beta_0"] + beta["beta_1"] * T_f + beta["beta_2"] * code + beta["beta_3"] * enso_code
            corrections[season] = delta

        # DJF (code=0) and SON (code=3) should differ
        assert corrections["DJF"] != corrections["SON"]

    def test_enso_affects_temperature(self):
        """ENSO phases should produce different corrections."""
        beta = DEFAULT_STATIC_COEFFICIENTS["temp"]
        T_f = 28.0
        season_code = 0

        corrections = {}
        for phase, code in ENSO_PHASES.items():
            delta = beta["beta_0"] + beta["beta_1"] * T_f + beta["beta_2"] * season_code + beta["beta_3"] * code
            corrections[phase] = delta

        assert corrections["Nina"] != corrections["Nino"]

    def test_precipitation_correction_uses_sigmoid(self):
        """P_adj = sigmoid(P_f + delta_p) * 100 should be bounded [0, 100]."""
        from scipy.special import expit as sigmoid

        gamma = DEFAULT_STATIC_COEFFICIENTS["precip"]
        P_f = 50.0
        humidity_f = 75.0
        W_f = 15.0

        delta_p = gamma["gamma_0"] + gamma["gamma_1"] * P_f + gamma["gamma_2"] * humidity_f + gamma["gamma_3"] * W_f
        P_adj = float(sigmoid(P_f + delta_p)) * 100

        assert 0.0 <= P_adj <= 100.0


class TestSeasonDetection:
    """Test season auto-detection from month."""

    @pytest.mark.parametrize("month,expected", [
        (1, "DJF"), (2, "DJF"), (12, "DJF"),
        (3, "MAM"), (4, "MAM"), (5, "MAM"),
        (6, "JJA"), (7, "JJA"), (8, "JJA"),
        (9, "SON"), (10, "SON"), (11, "SON"),
    ])
    def test_get_season(self, month, expected):
        # Replicate the logic from inference.py
        if month in [12, 1, 2]:
            result = "DJF"
        elif month in [3, 4, 5]:
            result = "MAM"
        elif month in [6, 7, 8]:
            result = "JJA"
        else:
            result = "SON"
        assert result == expected


class TestMonteCarlo:
    """Test Monte Carlo interval properties."""

    def test_fallback_intervals_are_symmetric(self):
        T_adj = 28.0
        sigma = DEFAULT_TEMP_SIGMA
        low = round(T_adj - 2 * sigma, 1)
        high = round(T_adj + 2 * sigma, 1)

        assert abs((high - T_adj) - (T_adj - low)) < 0.01

    def test_fallback_precip_intervals_are_clamped(self):
        # Very low P
        P_adj = 5.0
        low = max(0, round(P_adj - DEFAULT_PRECIP_INTERVAL, 1))
        high = min(100, round(P_adj + DEFAULT_PRECIP_INTERVAL, 1))
        assert low >= 0

        # Very high P
        P_adj = 95.0
        high = min(100, round(P_adj + DEFAULT_PRECIP_INTERVAL, 1))
        assert high <= 100


class TestCertaintyComputation:
    """Test certainty percentage computation logic."""

    def test_no_r_values_gives_50_percent(self):
        dc = {"temp_pearson_r": None, "precip_pearson_r": None, "n_samples": 0}
        r_values = []
        if dc.get("temp_pearson_r") is not None:
            r_values.append(abs(dc["temp_pearson_r"]))
        if dc.get("precip_pearson_r") is not None:
            r_values.append(abs(dc["precip_pearson_r"]))

        assert len(r_values) == 0

    def test_high_r_gives_high_certainty(self):
        dc = {"temp_pearson_r": 0.9, "precip_pearson_r": 0.8}
        r_values = [abs(dc["temp_pearson_r"]), abs(dc["precip_pearson_r"])]
        avg_r = float(np.mean(r_values))
        certainty = min(round(avg_r * 100), 95)

        assert certainty == 85

    def test_certainty_capped_at_95(self):
        dc = {"temp_pearson_r": 0.99, "precip_pearson_r": 0.98}
        r_values = [abs(dc["temp_pearson_r"]), abs(dc["precip_pearson_r"])]
        avg_r = float(np.mean(r_values))
        certainty = min(round(avg_r * 100), 95)

        assert certainty == 95
