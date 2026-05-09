"""
Lagi -- Fiji Weather Guardian: Centralized Configuration
=========================================================
All constants, thresholds, API URLs, and hyperparameters in one place.
Secrets are loaded from environment variables — never hardcoded.
"""

import os
import secrets
from pathlib import Path

# Load .env file if present (for local development)
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

PREDICTIONS_LOG = DATA_DIR / "predictions_log.jsonl"
ACTUALS_LOG = DATA_DIR / "actuals_log.jsonl"
DYNAMIC_COEFFICIENTS_FILE = OUTPUT_DIR / "dynamic_coefficients.json"
STATIC_COEFFICIENTS_FILE = OUTPUT_DIR / "coefficients.json"
ERROR_DISTRIBUTIONS_FILE = OUTPUT_DIR / "error_distributions.pkl"
PERFORMANCE_LOG = DATA_DIR / "performance_log.jsonl"
USERS_FILE = DATA_DIR / "users.jsonl"

# ---------------------------------------------------------------------------
# Secrets (from environment variables — NEVER hardcode)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SIGNUP_SECRET = os.getenv("LAGI_SIGNUP_SECRET", secrets.token_urlsafe(32))
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")

# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------
VERIFICATION_CODE_EXPIRY = 600  # 10 minutes
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "lagi@renles.com")

# ---------------------------------------------------------------------------
# Locations — Fiji cities with coordinates and station IDs
# ---------------------------------------------------------------------------
LOCATIONS = {
    "Suva": {"lat": -18.1416, "lon": 178.4419, "ghcn_id": "FJM00091680", "wmo_id": "91680"},
    "Nadi": {"lat": -17.7765, "lon": 177.9640, "ghcn_id": "FJM00091680", "wmo_id": "91685"},
    "Labasa": {"lat": -16.4167, "lon": 179.3667, "ghcn_id": "FJ000091670", "wmo_id": "91670"},
    "Lautoka": {"lat": -17.6065, "lon": 177.4540, "ghcn_id": "FJ000091684", "wmo_id": "91684"},
}

VALID_LOCATIONS = list(LOCATIONS.keys())
VALID_SEASONS = ["DJF", "MAM", "JJA", "SON"]
VALID_ENSO = ["Nina", "Neutral", "Nino"]

# ---------------------------------------------------------------------------
# Season & ENSO encoding
# ---------------------------------------------------------------------------
SEASONS = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
SEASON_NAMES = {
    0: "wet season (DJF)",
    1: "transition (MAM)",
    2: "dry season (JJA)",
    3: "transition (SON)",
}
ENSO_PHASES = {"Nina": -1, "Neutral": 0, "Nino": 1}

# ---------------------------------------------------------------------------
# Inference defaults
# ---------------------------------------------------------------------------
MONTE_CARLO_SAMPLES = 1000

DEFAULT_STATIC_COEFFICIENTS = {
    "temp": {"beta_0": 0.5, "beta_1": -0.02, "beta_2": 0.1, "beta_3": 0.05, "r_squared": 0.0},
    "precip": {"gamma_0": 5.0, "gamma_1": -0.05, "gamma_2": 0.02, "gamma_3": -0.1, "r_squared": 0.0},
}

DEFAULT_TEMP_SIGMA = 1.2
DEFAULT_PRECIP_INTERVAL = 15.0
DYNAMIC_R_THRESHOLD = 0.5
MIN_CORRELATION_SAMPLES = 5

# ---------------------------------------------------------------------------
# Data fetcher settings
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60.0
HTTP_TIMEOUT = 30
HTTP_MAX_RETRIES = 3
CACHE_TTL_SECONDS = 6 * 60 * 60

# ---------------------------------------------------------------------------
# Cyclone risk thresholds
# ---------------------------------------------------------------------------
CYCLONE_SEASON_MONTHS = [11, 12, 1, 2, 3, 4]

PRESSURE_NORMAL = 1010
PRESSURE_LOW = 1005
PRESSURE_VERY_LOW = 1000
PRESSURE_CRITICAL = 995

GUST_MODERATE = 40
GUST_ELEVATED = 60
GUST_HIGH = 90

PRECIP_MODERATE = 20
PRECIP_HEAVY = 50

CYCLONE_RISK_LEVELS = ["low", "elevated", "high", "extreme"]

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
API_VERSION = "2.0.0"
API_RATE_LIMIT = "30/minute"
ALLOWED_ORIGINS = [
    "https://renles.com",
    "https://www.renles.com",
    "http://localhost:8000",
    "http://localhost:3000",
]

# ---------------------------------------------------------------------------
# Groq LLM settings
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MAX_TOKENS = 512
GROQ_TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# Report branding colours
# ---------------------------------------------------------------------------
BRAND_COLORS = {
    "ocean_blue": "#0077B6",
    "deep_blue": "#023E8A",
    "coral": "#FF6B6B",
    "gold": "#FFD93D",
    "teal": "#00B4D8",
    "dark_bg": "#1a1a2e",
    "light_text": "#e0e0e0",
    "green": "#2ecc71",
    "orange": "#f39c12",
    "white": "#ffffff",
}

# ---------------------------------------------------------------------------
# Training hyperparameters (reference, used by train_lagi.py)
# ---------------------------------------------------------------------------
TRAINING = {
    "base_model": "Qwen/Qwen3.5-9B",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "optimizer": "paged_adamw_8bit",
    "quantization_bits": 4,
}
