"""Shared fixtures for Lagi test suite."""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root and scripts dir are importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Override config paths to use temp directories for test isolation."""
    import config

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "CACHE_DIR", data_dir / "cache")
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(config, "PREDICTIONS_LOG", data_dir / "predictions_log.jsonl")
    monkeypatch.setattr(config, "ACTUALS_LOG", data_dir / "actuals_log.jsonl")
    monkeypatch.setattr(config, "DYNAMIC_COEFFICIENTS_FILE", output_dir / "dynamic_coefficients.json")
    monkeypatch.setattr(config, "PERFORMANCE_LOG", data_dir / "performance_log.jsonl")

    # Also patch the validation module's references (it imports at module load)
    import validation
    monkeypatch.setattr(validation, "PREDICTIONS_LOG", data_dir / "predictions_log.jsonl")
    monkeypatch.setattr(validation, "ACTUALS_LOG", data_dir / "actuals_log.jsonl")
    monkeypatch.setattr(validation, "DYNAMIC_COEFFICIENTS_FILE", output_dir / "dynamic_coefficients.json")

    return tmp_path


@pytest.fixture
def sample_dynamic_coefficients():
    """Sample dynamic coefficients for testing."""
    return {
        "temp_pearson_r": 0.85,
        "temp_bias": 0.5,
        "temp_slope": 0.95,
        "temp_intercept": 1.2,
        "temp_r_squared": 0.72,
        "temp_n": 50,
        "precip_pearson_r": 0.70,
        "precip_bias": 3.0,
        "precip_slope": 0.90,
        "precip_intercept": 5.0,
        "precip_r_squared": 0.49,
        "precip_n": 50,
        "n_samples": 50,
        "last_updated": "2026-04-01T00:00:00",
    }
