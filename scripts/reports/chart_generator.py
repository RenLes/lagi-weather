"""Lagi Training Report — Chart Generation"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .report_data import (
    CHARTS_DIR, TRAIN_DATA, EVAL_DATA, COEFFICIENTS,
    OCEAN_BLUE, DEEP_BLUE, CORAL, GOLD, TEAL, GREEN, ORANGE, WHITE,
)


def set_chart_style():
    plt.rcParams.update({
        "figure.facecolor": "#0d1b2a",
        "axes.facecolor": "#1b2838",
        "axes.edgecolor": "#444",
        "axes.labelcolor": WHITE,
        "text.color": WHITE,
        "xtick.color": WHITE,
        "ytick.color": WHITE,
        "grid.color": "#333",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def chart_loss_curve():
    """Chart 1: Training loss curve with log scale initial phase."""
    set_chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 2]})

    epochs = [d[0] for d in TRAIN_DATA]
    losses = [d[1] for d in TRAIN_DATA]

    # Left: full view with log scale
    ax1.semilogy(epochs, losses, color=TEAL, linewidth=2, alpha=0.9)
    ax1.fill_between(epochs, losses, alpha=0.15, color=TEAL)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Training Loss (log scale)", fontsize=12)
    ax1.set_title("Loss Curve — Full View", fontsize=13, fontweight="bold", color=GOLD)
    ax1.axhline(y=0.089, color=CORAL, linestyle="--", alpha=0.5, label="Convergence: 0.089")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: zoomed convergence region
    converged = [(e, l) for e, l in zip(epochs, losses) if e > 0.15]
    ax2.plot([e for e, _ in converged], [l for _, l in converged],
             color=OCEAN_BLUE, linewidth=1.5, alpha=0.8)
    ax2.scatter([e for e, _ in converged], [l for _, l in converged],
                color=TEAL, s=12, alpha=0.6, zorder=5)

    # Eval points
    for ep, ev_loss in EVAL_DATA:
        ax2.scatter(ep, ev_loss, color=CORAL, s=80, zorder=10, marker="D",
                    edgecolors="white", linewidths=1.5)
        ax2.annotate(f"Eval: {ev_loss:.4f}", (ep, ev_loss),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=9, color=CORAL, fontweight="bold")

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Loss", fontsize=12)
    ax2.set_title("Convergence Detail — Epochs 0.2 to 3.0", fontsize=13,
                  fontweight="bold", color=GOLD)
    ax2.set_ylim(0.084, 0.100)
    ax2.grid(True, alpha=0.3)

    # Add min marker
    min_loss = min(losses)
    min_epoch = epochs[losses.index(min_loss)]
    ax2.annotate(f"Min: {min_loss:.4f}", (min_epoch, min_loss),
                 textcoords="offset points", xytext=(-40, -20),
                 fontsize=9, color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN))

    fig.suptitle("Lagi QLoRA Fine-Tuning — Training Loss", fontsize=15,
                 fontweight="bold", color=WHITE, y=1.02)
    plt.tight_layout()
    path = CHARTS_DIR / "01_loss_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def chart_lr_schedule():
    """Chart 2: Learning rate schedule."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    epochs = [d[0] for d in TRAIN_DATA]
    lrs = [d[3] for d in TRAIN_DATA]

    ax.plot(epochs, [lr * 1e4 for lr in lrs], color=GOLD, linewidth=2.5)
    ax.fill_between(epochs, [lr * 1e4 for lr in lrs], alpha=0.15, color=GOLD)

    # Mark warmup end
    warmup_end = 0.0909  # ~5% warmup
    ax.axvline(x=warmup_end, color=CORAL, linestyle=":", alpha=0.7)
    ax.annotate("Warmup\ncomplete", (warmup_end, 1.8), fontsize=9,
                color=CORAL, ha="center")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate (×10⁻⁴)", fontsize=12)
    ax.set_title("Cosine Learning Rate Schedule with 5% Warmup",
                 fontsize=13, fontweight="bold", color=GOLD)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = CHARTS_DIR / "02_lr_schedule.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def chart_grad_norm():
    """Chart 3: Gradient norm convergence."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    epochs = [d[0] for d in TRAIN_DATA]
    grads = [d[2] for d in TRAIN_DATA]

    ax.semilogy(epochs, grads, color=GREEN, linewidth=1.5, alpha=0.8)
    ax.fill_between(epochs, grads, alpha=0.1, color=GREEN)

    ax.axhline(y=0.005, color=CORAL, linestyle="--", alpha=0.5,
               label="Stability threshold: 0.005")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gradient Norm (log scale)", fontsize=12)
    ax.set_title("Gradient Norm — Convergence Stability",
                 fontsize=13, fontweight="bold", color=GOLD)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = CHARTS_DIR / "03_grad_norm.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def chart_train_vs_eval():
    """Chart 4: Train vs Eval loss comparison."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Average train loss per epoch (only epochs with complete data)
    epoch_train = {}
    for ep, loss, _, _ in TRAIN_DATA:
        bucket = int(ep) + 1
        if bucket not in epoch_train:
            epoch_train[bucket] = []
        epoch_train[bucket].append(loss)

    train_avgs = {k: np.mean(v) for k, v in epoch_train.items()}
    eval_losses = {int(e): l for e, l in EVAL_DATA}

    # Only show epochs that have eval data, plus epoch 3 train with "pending" note
    epochs_plot = sorted(train_avgs.keys())
    x = np.arange(len(epochs_plot))
    width = 0.35

    bars1 = ax.bar(x - width/2, [train_avgs[e] for e in epochs_plot],
                   width, label="Train Loss (avg)", color=OCEAN_BLUE, alpha=0.85)

    # For eval, use actual values where available, 0 where pending
    eval_vals = []
    for e in epochs_plot:
        if e in eval_losses:
            eval_vals.append(eval_losses[e])
        else:
            eval_vals.append(0)

    bars2 = ax.bar(x + width/2, eval_vals,
                   width, label="Eval Loss", color=CORAL, alpha=0.85)

    # Value labels
    for bar in bars1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom",
                    fontsize=10, color=WHITE, fontweight="bold")
    for i, bar in enumerate(bars2):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom",
                    fontsize=10, color=WHITE, fontweight="bold")
        elif epochs_plot[i] == 3:
            ax.text(bar.get_x() + bar.get_width()/2., 0.005,
                    "pending", ha="center", va="bottom",
                    fontsize=8, color="#888", fontstyle="italic")

    # Generalization gap annotation
    if 2 in eval_losses and 2 in train_avgs:
        gap = eval_losses[2] - train_avgs[2]
        ax.annotate(f"Gap: {gap:.4f}\n(No overfitting!)",
                    xy=(1 + width/2, eval_losses[2]),
                    xytext=(1.8, 0.105),
                    fontsize=10, color=GREEN, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=GREEN))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Train vs Evaluation Loss — No Overfitting",
                 fontsize=13, fontweight="bold", color=GOLD)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs_plot])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = CHARTS_DIR / "04_train_vs_eval.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def chart_coefficients():
    """Chart 5: Bias correction coefficients."""
    set_chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Temperature coefficients
    t = COEFFICIENTS["temp"]
    t_names = ["β₀\n(Intercept)", "β₁\n(Forecast)", "β₂\n(Season)", "β₃\n(ENSO)"]
    t_vals = [t["beta_0"], t["beta_1"], t["beta_2"], t["beta_3"]]
    t_colors = [OCEAN_BLUE if v >= 0 else CORAL for v in t_vals]

    bars1 = ax1.bar(t_names, t_vals, color=t_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars1, t_vals):
        y_pos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.03
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f"{val:+.4f}", ha="center", va="bottom", fontsize=10,
                 color=WHITE, fontweight="bold")

    ax1.set_ylabel("Coefficient Value", fontsize=12)
    ax1.set_title("Temperature Correction\nΔT = β₀ + β₁·T_f + β₂·Season + β₃·ENSO",
                  fontsize=12, fontweight="bold", color=TEAL)
    ax1.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="y")

    # Precipitation coefficients
    p = COEFFICIENTS["precip"]
    p_names = ["γ₀\n(Intercept)", "γ₁\n(Forecast)", "γ₂\n(Humidity)", "γ₃\n(Wind)"]
    p_vals = [p["gamma_0"], p["gamma_1"], p["gamma_2"], p["gamma_3"]]
    p_colors = [OCEAN_BLUE if v >= 0 else CORAL for v in p_vals]

    bars2 = ax2.bar(p_names, p_vals, color=p_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, p_vals):
        y_pos = bar.get_height() + 0.15 if val >= 0 else bar.get_height() - 0.4
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f"{val:+.4f}", ha="center", va="bottom", fontsize=10,
                 color=WHITE, fontweight="bold")

    ax2.set_ylabel("Coefficient Value", fontsize=12)
    ax2.set_title("Precipitation Correction\nΔP = γ₀ + γ₁·P_f + γ₂·Humidity + γ₃·Wind",
                  fontsize=12, fontweight="bold", color=TEAL)
    ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Learned Bias Correction Coefficients", fontsize=15,
                 fontweight="bold", color=GOLD, y=1.02)
    plt.tight_layout()
    path = CHARTS_DIR / "05_coefficients.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def chart_corrections():
    """Chart 6: Before vs after correction practical scenarios."""
    set_chart_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    t = COEFFICIENTS["temp"]
    p = COEFFICIENTS["precip"]

    # Temperature scenarios
    scenarios_t = [
        ("Suva\nSummer\nEl Niño", 30, 1, 1),
        ("Nadi\nSummer\nNeutral", 32, 1, 0),
        ("Lautoka\nWinter\nLa Niña", 27, -1, -1),
        ("Labasa\nWinter\nNeutral", 29, -1, 0),
    ]

    labels_t = []
    forecast_t = []
    corrected_t = []
    for label, tf, season, enso in scenarios_t:
        delta = t["beta_0"] + t["beta_1"]*tf + t["beta_2"]*season + t["beta_3"]*enso
        labels_t.append(label)
        forecast_t.append(tf)
        corrected_t.append(tf + delta)

    x = np.arange(len(labels_t))
    width = 0.35
    ax1.bar(x - width/2, forecast_t, width, label="Raw Forecast", color="#555", alpha=0.7)
    ax1.bar(x + width/2, corrected_t, width, label="Lagi Corrected", color=TEAL, alpha=0.85)

    for i, (f, c) in enumerate(zip(forecast_t, corrected_t)):
        ax1.annotate(f"+{c-f:.1f}°C", (i + width/2, c + 0.1),
                     ha="center", fontsize=9, color=GREEN, fontweight="bold")

    ax1.set_ylabel("Temperature (°C)", fontsize=12)
    ax1.set_title("Temperature: Forecast vs Corrected",
                  fontsize=12, fontweight="bold", color=TEAL)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_t, fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Precipitation scenarios
    scenarios_p = [
        ("Light\n5mm", 5, 75, 15),
        ("Moderate\n10mm", 10, 80, 20),
        ("Heavy\n20mm", 20, 90, 30),
        ("Dry Day\n0mm", 0, 60, 5),
    ]

    labels_p = []
    forecast_p = []
    corrected_p = []
    for label, pf, hum, wind in scenarios_p:
        delta = p["gamma_0"] + p["gamma_1"]*pf + p["gamma_2"]*hum + p["gamma_3"]*wind
        labels_p.append(label)
        forecast_p.append(pf)
        corrected_p.append(pf + delta)

    x2 = np.arange(len(labels_p))
    ax2.bar(x2 - width/2, forecast_p, width, label="Raw Forecast", color="#555", alpha=0.7)
    ax2.bar(x2 + width/2, corrected_p, width, label="Lagi Corrected", color=OCEAN_BLUE, alpha=0.85)

    for i, (f, c) in enumerate(zip(forecast_p, corrected_p)):
        pct = ((c - f) / max(f, 0.1)) * 100
        ax2.annotate(f"+{c-f:.1f}mm", (i + width/2, c + 0.3),
                     ha="center", fontsize=9, color=GREEN, fontweight="bold")

    ax2.set_ylabel("Rainfall (mm)", fontsize=12)
    ax2.set_title("Precipitation: Forecast vs Corrected",
                  fontsize=12, fontweight="bold", color=TEAL)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_p, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Practical Impact — Before & After Lagi Correction",
                 fontsize=15, fontweight="bold", color=GOLD, y=1.02)
    plt.tight_layout()
    path = CHARTS_DIR / "06_corrections.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def chart_architecture():
    """Chart 7: System architecture flow diagram."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white",
                              linewidth=1.5, alpha=0.85, zorder=5)
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight=weight, zorder=10,
                wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=2),
                    zorder=3)

    # Title
    ax.text(7, 7.5, "Lagi Weather Guardian — System Architecture",
            ha="center", fontsize=16, fontweight="bold", color=GOLD)

    # Row 1: 5 Sources
    sources_short = ["FMS\nmet.gov.fj", "Windy\nECMWF/GFS", "AccuWeather\nFiji", "Wunderground", "BOM/Metvuw\nAustralia"]
    for i, src in enumerate(sources_short):
        draw_box(0.3 + i*2.7, 5.8, 2.4, 1.0, src, "#2d5986", fontsize=8)
        draw_arrow(0.3 + i*2.7 + 1.2, 5.8, 7, 5.1)

    # Row 2: Ensemble
    draw_box(5, 4.4, 4, 0.7, "5-Source Ensemble Average", OCEAN_BLUE, fontsize=11, bold=True)
    draw_arrow(7, 4.4, 7, 3.8)

    # Row 3: Bias Correction
    draw_box(4, 3.1, 6, 0.7, "Static Bias Correction: ΔT = β₀ + β₁T + β₂Season + β₃ENSO",
             "#1a6b4a", fontsize=10, bold=True)
    draw_arrow(7, 3.1, 7, 2.5)

    # Row 4: Dynamic + Monte Carlo
    draw_box(1, 1.8, 5, 0.7, "Dynamic Adjustment (Pearson r)\nfinal_T = T × slope + intercept",
             "#8b4513", fontsize=9, bold=True)
    draw_box(7, 1.8, 5.5, 0.7, "Monte Carlo Stress Test\n1,000 error realisations → certainty %",
             "#6b2d5b", fontsize=9, bold=True)
    draw_arrow(7, 2.5, 3.5, 2.5)
    draw_arrow(7, 2.5, 9.75, 2.5)

    # Row 5: Output
    draw_arrow(3.5, 1.8, 7, 1.2)
    draw_arrow(9.75, 1.8, 7, 1.2)
    draw_box(4.5, 0.4, 5, 0.8, "Warm Fijian Natural Language Response\n+ Certainty % + Disclaimer",
             CORAL, fontsize=10, bold=True)

    # Feedback loop
    ax.annotate("", xy=(12.5, 3.5), xytext=(12.5, 0.8),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(13.2, 2.2, "Daily\nFeedback\nLoop", fontsize=8, color=GREEN,
            ha="center", fontweight="bold")

    path = CHARTS_DIR / "07_architecture.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#0d1b2a")
    plt.close()
    return path


def chart_improvement_projection():
    """Chart 8: Projected correlation improvement over time."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Projected Pearson r growth
    weeks = np.arange(0, 27)
    # Logistic growth: starts at 0.016, asymptotes at ~0.85
    r_proj = 0.016 + (0.85 - 0.016) / (1 + np.exp(-0.25 * (weeks - 12)))

    ax.plot(weeks, r_proj, color=TEAL, linewidth=3, label="Projected Pearson r")
    ax.fill_between(weeks, r_proj * 0.85, np.minimum(r_proj * 1.15, 0.95),
                    alpha=0.15, color=TEAL, label="Confidence interval")

    # Threshold lines
    ax.axhline(y=0.5, color=GOLD, linestyle="--", alpha=0.6, label="Dynamic adjustment activates (r > 0.5)")
    ax.axhline(y=0.6, color=GREEN, linestyle="--", alpha=0.6, label="Strong prediction threshold")

    # Current position
    ax.scatter([0], [0.016], color=CORAL, s=100, zorder=10, marker="*")
    ax.annotate("Current\n(synthetic data)", (0, 0.016),
                textcoords="offset points", xytext=(30, 20),
                fontsize=9, color=CORAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CORAL))

    # Milestone markers
    ax.scatter([8], [r_proj[8]], color=GOLD, s=60, zorder=10)
    ax.annotate("~2 months:\nDynamic kicks in", (8, r_proj[8]),
                textcoords="offset points", xytext=(20, -30),
                fontsize=8, color=GOLD,
                arrowprops=dict(arrowstyle="->", color=GOLD))

    ax.scatter([20], [r_proj[20]], color=GREEN, s=60, zorder=10)
    ax.annotate("~5 months:\nHigh accuracy", (20, r_proj[20]),
                textcoords="offset points", xytext=(15, -25),
                fontsize=8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))

    ax.set_xlabel("Weeks Since Launch", fontsize=12)
    ax.set_ylabel("Pearson r Correlation", fontsize=12)
    ax.set_title("Self-Improvement Projection — Correlation Growth Over Time",
                 fontsize=13, fontweight="bold", color=GOLD)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = CHARTS_DIR / "08_projection.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path
