# Patreon Journal Entry #1: Lagi is Born

**Date:** March 2026
**Agent:** Lagi -- Fiji Weather Guardian
**Author:** Renles

---

## Bula, Supporters!

Today I'm excited to share the birth of **Lagi**, our Fiji Weather Guardian -- an AI agent built to correct weather forecast biases specific to Fiji's tropical climate.

## What is Lagi?

Lagi is a fine-tuned large language model that takes raw weather forecasts for Suva, Nadi, Labasa, and Lautoka and applies learned correction coefficients to give you a more accurate prediction. It doesn't replace the Fiji Met Service -- it builds on top of existing forecasts to reduce systematic errors.

The name "Lagi" means "sky" or "heaven" in Fijian -- fitting for a weather guardian.

## Training Details

### Model & Method
- **Base Model:** Qwen3.5-9B (4-bit quantisation)
- **Method:** QLoRA (Low-Rank Adaptation with 4-bit quantisation)
- **LoRA Configuration:** Rank 16, Alpha 32
- **Epochs:** 3
- **Batch Size:** 1 with gradient accumulation of 8
- **Learning Rate:** 2e-4 with cosine schedule

### Compute

| Item | Detail |
|------|--------|
| GPU | 1x NVIDIA RTX 4090 (24 GB VRAM) |
| Quantisation | 4-bit (QLoRA) |
| Batch Size | 1 (gradient accumulation 8) |

### Training Data
- ~5,500 daily forecast-vs-actual pairs (2010-2025)
- Sources: Fiji Meteorological Service, Wunderground, CRU, ERA5
- Locations: Suva, Nadi, Labasa, Lautoka
- Includes ENSO phase, seasonal encoding, humidity, and wind data

## The Mathematical Core

What makes Lagi special is the bias correction system trained on 15 years of Fiji weather data:

**Temperature Correction:**
```
DeltaT = beta_0 + beta_1 * T_forecast + beta_2 * Season + beta_3 * ENSO
T_adjusted = T_forecast + DeltaT
```

**Precipitation Correction:**
```
DeltaP = gamma_0 + gamma_1 * P_forecast + gamma_2 * Humidity + gamma_3 * Wind
P_adjusted = sigmoid(P_forecast + DeltaP)
```

Every prediction comes with **95% credible intervals** computed from 1,000 Monte Carlo samples drawn from historical error distributions (Gaussian mixture models).

## Karpathy Research Loop

During training, Lagi ran 25 autonomous micro-experiments:
- Each experiment tweaked the correction coefficients slightly
- Ran a short LoRA training epoch
- Evaluated against a 20% held-out test set
- Kept improvements, reverted failures

This is inspired by Andrej Karpathy's approach to autonomous research -- letting the model explore the solution space systematically.

## What's Next

- **Live deployment** on renles.com with API + web UI
- **Quarterly retraining** with fresh weather data via Adam & Eve framework
- **Premium features** including extended forecasts and custom location alerts
- **Community feedback** -- we want to hear from Fijians about forecast accuracy

## Links

- **GitHub:** [renles/lagi-weather](https://github.com/renles/lagi-weather) (full source code including training scripts)
- **Live:** renles.com (coming soon)

## Compute Details

Training was completed on a single RTX 4090. Training logs and metrics are available in the GitHub repository under `logs/`.

---

*Vinaka vakalevu for your support! Every patron helps us build AI that serves Pacific communities.*

*-- Renles*
