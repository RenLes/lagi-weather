"""Tests for the data fetcher: ensemble averaging, source degradation."""

import pytest
import numpy as np

from data_fetcher import compute_ensemble_average
from config import LOCATIONS


class TestComputeEnsembleAverage:
    def test_empty_forecasts_returns_none(self):
        result = compute_ensemble_average([])
        assert result["temperature"] is None
        assert result["rain_probability"] is None
        assert result["n_sources"] == 0

    def test_single_source(self):
        forecasts = [
            {"source": "open_meteo", "temperature": 28.5, "rain_probability": 60.0,
             "wind": 15.0, "humidity": 75.0},
        ]
        result = compute_ensemble_average(forecasts)
        assert result["temperature"] == 28.5
        assert result["rain_probability"] == 60.0
        assert result["n_sources"] == 1

    def test_averages_multiple_sources(self):
        forecasts = [
            {"source": "a", "temperature": 28.0, "rain_probability": 60.0, "wind": 10.0, "humidity": 70.0},
            {"source": "b", "temperature": 30.0, "rain_probability": 40.0, "wind": 20.0, "humidity": 80.0},
        ]
        result = compute_ensemble_average(forecasts)
        assert result["temperature"] == 29.0
        assert result["rain_probability"] == 50.0
        assert result["wind"] == 15.0
        assert result["humidity"] == 75.0
        assert result["n_sources"] == 2

    def test_handles_missing_values_gracefully(self):
        forecasts = [
            {"source": "a", "temperature": 28.0, "rain_probability": 60.0, "wind": None, "humidity": 70.0},
            {"source": "b", "temperature": 30.0, "rain_probability": None, "wind": 20.0, "humidity": None},
        ]
        result = compute_ensemble_average(forecasts)
        assert result["temperature"] == 29.0
        assert result["rain_probability"] == 60.0  # only one value
        assert result["wind"] == 20.0  # only one value
        assert result["humidity"] == 70.0  # only one value

    def test_sources_list_populated(self):
        forecasts = [
            {"source": "fiji_met", "temperature": 28.0, "rain_probability": 50.0, "wind": 10.0, "humidity": 70.0},
            {"source": "open_meteo", "temperature": 29.0, "rain_probability": 55.0, "wind": 12.0, "humidity": 72.0},
        ]
        result = compute_ensemble_average(forecasts)
        assert "fiji_met" in result["sources"]
        assert "open_meteo" in result["sources"]


class TestLocations:
    def test_all_locations_have_coordinates(self):
        for name, loc in LOCATIONS.items():
            assert "lat" in loc, f"{name} missing lat"
            assert "lon" in loc, f"{name} missing lon"
            assert isinstance(loc["lat"], float)
            assert isinstance(loc["lon"], float)

    def test_fiji_latitude_range(self):
        """Fiji is roughly between latitudes -12 and -22."""
        for name, loc in LOCATIONS.items():
            assert -22 <= loc["lat"] <= -12, f"{name} latitude {loc['lat']} outside Fiji range"

    def test_fiji_longitude_range(self):
        """Fiji is roughly between longitudes 176 and 180."""
        for name, loc in LOCATIONS.items():
            assert 176 <= loc["lon"] <= 180, f"{name} longitude {loc['lon']} outside Fiji range"
