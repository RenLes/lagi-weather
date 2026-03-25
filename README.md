# Lagi - Fiji Weather Guardian

AI-corrected weather forecasting for Fiji. Bula!

Lagi uses a fine-tuned Qwen3.5-27B/32B model with QLoRA to correct weather forecast biases specific to Fiji's tropical climate, covering **Suva, Nadi, Labasa, and Lautoka**.

## How It Works

1. **Bias Correction** — Learned regression coefficients correct systematic forecast errors using ENSO phase, seasonal patterns, humidity, and wind
2. **Monte Carlo Intervals** — 1,000 error realisations from historical distributions provide 95% credible intervals
3. **Fine-Tuned LLM** — QLoRA-adapted model generates natural language explanations in warm Fijian-English

### Correction Formulas

```
T_adj = T_forecast + (β₀ + β₁·T_f + β₂·Season + β₃·ENSO)
P_adj = sigmoid(P_forecast + (γ₀ + γ₁·P_f + γ₂·Humidity + γ₃·Wind))
```

## Quick Start

### Training (Vast.ai RTX 4090)

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

# Open web UI at http://localhost:8000
```

### API Usage

```bash
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
```

## Project Structure

```
lagi-weather/
├── scripts/
│   ├── train_lagi.py        # QLoRA fine-tuning with full mathematical core
│   └── run_training.sh      # Resilient nohup training wrapper
├── deployment/
│   ├── api.py               # FastAPI server
│   ├── inference.py          # Bias correction + Monte Carlo engine
│   └── web_ui.html           # Web interface
├── docs/
│   ├── lagi-portfolio.md     # Agent portfolio
│   └── patreon-entry-1.md    # Patreon journal entry
├── data/                     # Training data (CSV files)
├── output/                   # Trained model + coefficients
├── logs/                     # Training logs
└── requirements.txt
```

## Training Specs

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3.5-27B/32B |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Quantisation | 4-bit (QLoRA) |
| Epochs | 3 |
| Batch Size | 4-8 |
| Learning Rate | 2e-4 |
| GPU | 1x RTX 4090 (24 GB) |
| Duration | 8-10 hours |
| Cost | < $5 (Vast.ai) |

## The 5 Rules of Lagi

1. **Respect and Honesty** — Cite sources, be transparent about limitations
2. **Humility** — Equal to all, superior to none
3. **Communication** — Clear, warm Fijian-English explanations
4. **First Principles** — Root problem-solving approach
5. **Have Fun** — Joyful, hopeful, Pacific-spirited

## Links

- **Live:** [renles.com](https://renles.com)
- **Patreon:** Journal entries with cost proof and training metrics

## License

MIT

---

*Vinaka vakalevu! Lagi is here to keep Fiji safe and informed.*
