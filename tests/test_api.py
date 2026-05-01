"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from deployment.api import app
from config import VALID_LOCATIONS, VALID_ENSO, API_VERSION

client = TestClient(app)


class TestRootEndpoint:
    def test_root_returns_bula(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "Bula" in data["message"]
        assert data["version"] == API_VERSION

    def test_root_lists_locations(self):
        resp = client.get("/")
        data = resp.json()
        assert set(data["locations"]) == set(VALID_LOCATIONS)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["agent"] == "Lagi"
        assert "uptime_seconds" in data
        assert "timestamp" in data


class TestLocationsEndpoint:
    def test_lists_all_locations(self):
        resp = client.get("/locations")
        assert resp.status_code == 200
        assert set(resp.json()["locations"]) == set(VALID_LOCATIONS)


class TestForecastEndpoint:
    def test_valid_forecast(self):
        resp = client.post("/forecast", json={
            "location": "Suva",
            "temperature": 28.0,
            "rain_probability": 50.0,
            "wind": 15.0,
            "humidity": 75.0,
            "enso_phase": "Neutral",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["location"] == "Suva"
        assert "adjusted_forecast" in data
        assert "monte_carlo" in data

    def test_invalid_location(self):
        resp = client.post("/forecast", json={
            "location": "Auckland",
            "temperature": 20.0,
            "rain_probability": 30.0,
            "wind": 10.0,
        })
        assert resp.status_code == 400

    def test_invalid_enso_phase(self):
        resp = client.post("/forecast", json={
            "location": "Suva",
            "temperature": 28.0,
            "rain_probability": 50.0,
            "wind": 15.0,
            "enso_phase": "SuperNino",
        })
        assert resp.status_code == 400

    def test_rain_probability_range_validation(self):
        resp = client.post("/forecast", json={
            "location": "Suva",
            "temperature": 28.0,
            "rain_probability": 150.0,  # Invalid
            "wind": 15.0,
        })
        assert resp.status_code == 422  # Pydantic validation error


class TestValidationEndpoints:
    def test_get_validation_report(self):
        resp = client.get("/validation/report")
        assert resp.status_code == 200
        assert "LAGI VALIDATION REPORT" in resp.text

    def test_log_actual_valid(self):
        resp = client.post("/validation/log-actual", json={
            "date": "2026-04-01",
            "location": "Suva",
            "temperature": 29.0,
            "rain_probability": 70.0,
        })
        assert resp.status_code == 200
        assert "Vinaka" in resp.json()["message"]

    def test_log_actual_invalid_location(self):
        resp = client.post("/validation/log-actual", json={
            "date": "2026-04-01",
            "location": "Sydney",
            "temperature": 22.0,
            "rain_probability": 30.0,
        })
        assert resp.status_code == 400


class TestChatEndpoint:
    def test_chat_returns_reply(self):
        resp = client.post("/chat", json={
            "question": "Will it rain today?",
            "location": "Suva",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert "Bula" in data["reply"]
        assert "certainty" in data

    def test_chat_invalid_location(self):
        resp = client.post("/chat", json={
            "question": "Weather?",
            "location": "Mars",
        })
        assert resp.status_code == 400


class TestCertaintyEndpoint:
    def test_returns_certainty(self):
        resp = client.get("/certainty")
        assert resp.status_code == 200
        data = resp.json()
        assert "certainty_pct" in data
        assert 0 <= data["certainty_pct"] <= 100
