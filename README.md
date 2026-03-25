# Lagi - Fiji Weather Guardian

AI-corrected weather forecasting for Fiji. Bula!

Lagi uses a fine-tuned LLM with QLoRA to correct weather forecast biases specific to Fiji's tropical climate, covering **Suva, Nadi, Labasa, and Lautoka**.

## How It Works

1. **5-Source Ensemble** — Aggregates forecasts from Fiji Met, Windy, AccuWeather, Wunderground, and BOM/Metvuw
2. **Bias Correction** — Learned regression coefficients correct systematic forecast errors using ENSO phase, seasonal patterns, humidity, and wind
3. **Dynamic Validation** — Pearson r correlation between predictions and actuals drives continuous coefficient refinement
4. **Monte Carlo Intervals** — 1,000 error realisations from historical distributions provide 95% credible intervals
5. **Natural Language Chat** — Ask Lagi weather questions in plain English and get warm, Fijian-tone responses
6. **Certainty Tracking** — Every response includes a certainty percentage derived from validation history

### Correction Formulas

```
T_adj = T_forecast + (β₀ + β₁·T_f + β₂·Season + β₃·ENSO)
P_adj = sigmoid(P_forecast + (γ₀ + γ₁·P_f + γ₂·Humidity + γ₃·Wind))
```

## Quick Start

### Training

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (resilient to SSH drops)
./scripts/run_training.sh --background

# Monitor progress
tail -f logs/nohup_training.log
```

### Deployment

```bash
# Start the API server
uvicorn deployment.api:app --host 0.0.0.0 --port 8000

# Open web UI at http://localhost:8000/ui
```

### API Usage

```bash
# Manual forecast correction
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Suva",
    "temperature": 28.5,
    "rain_probability": 65,
    "wind": 12,
    "humidity": 78,
    "enso_phase": "Neutral"
  }'

# Auto-fetch ensemble forecast
curl -X POST http://localhost:8000/forecast/auto \
  -H "Content-Type: application/json" \
  -d '{"location": "Suva", "enso_phase": "Neutral"}'

# Natural language chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Will it rain in Nadi today?", "location": "Nadi"}'
```

## Project Structure

```
lagi-weather/
├── scripts/
│   ├── train_lagi.py        # QLoRA fine-tuning with full mathematical core
│   ├── run_training.sh      # Resilient nohup training wrapper
│   ├── data_fetcher.py      # 5-source ensemble data fetching
│   └── validation.py        # Prediction vs actual validation + dynamic coefficients
├── deployment/
│   ├── api.py               # FastAPI server with chat + validation endpoints
│   ├── inference.py          # Ensemble + bias correction + Monte Carlo engine
│   └── web_ui.html           # 3-tab web interface (Chat, Forecast, Info)
├── docs/
│   ├── lagi-portfolio.md     # Agent portfolio
│   ├── lagi-personality.md   # Full personality & phrase library
│   ├── user-guide.md         # Onboarding guide
│   └── patreon-entry-1.md    # Patreon journal entry
├── data/                     # Training data (CSV files)
├── output/                   # Trained model + coefficients
├── logs/                     # Training logs
└── requirements.txt
```

## Training Specs

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3.5-9B |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Quantisation | 4-bit (QLoRA) |
| Epochs | 3 |
| Batch Size | 1 (gradient accumulation 8) |
| Learning Rate | 2e-4 |
| Compute | 1x NVIDIA RTX 4090 (24 GB VRAM) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forecast` | POST | Correct a manual forecast input |
| `/forecast/auto` | POST | Auto-fetch 5 sources + ensemble + correct |
| `/chat` | POST | Natural language weather chat |
| `/certainty` | GET | Current prediction certainty level |
| `/validation/report` | GET | Prediction vs actual correlation report |
| `/validation/log-actual` | POST | Log observed weather for validation |
| `/validation/compute` | POST | Recompute dynamic coefficients |
| `/ui` | GET | Web interface |

## The 5 Rules of Lagi

1. **Respect and Honesty** — Cite sources, be transparent about limitations
2. **Humility** — Equal to all, superior to none
3. **Communication** — Clear, warm Fijian-English explanations
4. **First Principles** — Root problem-solving approach
5. **Have Fun** — Joyful, hopeful, Pacific-spirited

## Links

- **Live:** [renles.com](https://renles.com)
- **Patreon:** Journal entries with training metrics

## License

MIT

---

*Vinaka vakalevu! Lagi is here to keep Fiji safe and informed.*
