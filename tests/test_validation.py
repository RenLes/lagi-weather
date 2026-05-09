"""Tests for the validation module: logging, correlation, and dynamic adjustment."""

import json

import pytest

from validation import (
    log_prediction, log_actual, load_predictions, load_actuals,
    compute_correlation, apply_dynamic_adjustment, load_dynamic_coefficients,
    generate_validation_report, _default_coefficients,
)
from config import MIN_CORRELATION_SAMPLES, DYNAMIC_R_THRESHOLD


class TestLogPrediction:
    def test_logs_prediction_to_file(self, tmp_data_dir):
        log_prediction(
            location="Suva",
            predicted_T=28.5,
            predicted_P=65.0,
            predicted_W=12.0,
        )
        df = load_predictions()
        assert len(df) == 1
        assert df.iloc[0]["location"] == "Suva"
        assert df.iloc[0]["predicted_T"] == 28.5

    def test_appends_multiple_predictions(self, tmp_data_dir):
        log_prediction(location="Suva", predicted_T=28.0, predicted_P=60.0, predicted_W=10.0)
        log_prediction(location="Nadi", predicted_T=30.0, predicted_P=40.0, predicted_W=8.0)
        df = load_predictions()
        assert len(df) == 2


class TestLogActual:
    def test_logs_actual_to_file(self, tmp_data_dir):
        log_actual(date="2026-04-01", location="Suva", T_a=29.0, P_a=70.0, W_a=14.0)
        df = load_actuals()
        assert len(df) == 1
        assert df.iloc[0]["T_a"] == 29.0

    def test_logs_with_optional_wind(self, tmp_data_dir):
        log_actual(date="2026-04-01", location="Suva", T_a=29.0, P_a=70.0)
        df = load_actuals()
        assert df.iloc[0]["W_a"] is None


class TestComputeCorrelation:
    def test_returns_defaults_when_no_data(self, tmp_data_dir):
        result = compute_correlation()
        assert result["n_samples"] == 0
        assert result["temp_pearson_r"] is None

    def test_returns_defaults_when_too_few_pairs(self, tmp_data_dir):
        for i in range(3):
            log_prediction(location="Suva", predicted_T=28.0 + i, predicted_P=50.0, predicted_W=10.0)
            log_actual(date=f"2026-04-0{i+1}", location="Suva", T_a=28.5 + i, P_a=55.0)
        result = compute_correlation()
        assert result["n_samples"] == 0  # defaults returned

    def test_computes_correlation_with_enough_data(self, tmp_data_dir):
        import config
        # Generate enough matched pairs
        for i in range(10):
            date = f"2026-04-{i+1:02d}"
            # Predictions with some noise
            log_prediction(location="Suva", predicted_T=25.0 + i * 0.5, predicted_P=40.0 + i * 3, predicted_W=10.0)
            # Actuals with correlated values
            log_actual(date=date, location="Suva", T_a=25.5 + i * 0.5, P_a=42.0 + i * 3)

        result = compute_correlation()
        assert result["n_samples"] >= MIN_CORRELATION_SAMPLES
        assert result["temp_pearson_r"] is not None
        assert -1.0 <= result["temp_pearson_r"] <= 1.0

        # Coefficients should be saved to file
        assert config.DYNAMIC_COEFFICIENTS_FILE.exists()


class TestApplyDynamicAdjustment:
    def test_uses_regression_when_strong_correlation(self, sample_dynamic_coefficients):
        result = apply_dynamic_adjustment(28.0, 60.0, sample_dynamic_coefficients)
        # With r=0.85 > threshold, should use regression
        assert result["temp_method"] == "regression"
        expected_T = 28.0 * 0.95 + 1.2
        assert result["adjusted_T"] == round(expected_T, 1)

    def test_uses_bias_offset_when_weak_correlation(self):
        weak_coeffs = {
            "temp_pearson_r": 0.3,
            "temp_bias": 0.5,
            "temp_slope": 1.0,
            "temp_intercept": 0.0,
            "precip_pearson_r": 0.2,
            "precip_bias": 2.0,
            "precip_slope": 1.0,
            "precip_intercept": 0.0,
            "n_samples": 10,
        }
        result = apply_dynamic_adjustment(28.0, 60.0, weak_coeffs)
        assert result["temp_method"] == "bias_offset"
        assert result["adjusted_T"] == round(28.0 + 0.5, 1)

    def test_clamps_precipitation_to_valid_range(self):
        coeffs = _default_coefficients()
        coeffs["precip_bias"] = -80.0
        result = apply_dynamic_adjustment(28.0, 10.0, coeffs)
        assert result["adjusted_P"] >= 0.0

        coeffs["precip_bias"] = 80.0
        result = apply_dynamic_adjustment(28.0, 60.0, coeffs)
        assert result["adjusted_P"] <= 100.0

    def test_returns_none_r_when_no_coefficients(self):
        result = apply_dynamic_adjustment(28.0, 60.0, _default_coefficients())
        assert result["temp_pearson_r"] is None
        assert result["precip_pearson_r"] is None


class TestGenerateValidationReport:
    def test_report_contains_header(self, tmp_data_dir):
        report = generate_validation_report()
        assert "LAGI VALIDATION REPORT" in report

    def test_report_shows_sample_count(self, tmp_data_dir):
        report = generate_validation_report()
        assert "n_samples" in report.lower() or "Matched pairs" in report
