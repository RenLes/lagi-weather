"""Lagi Training Report — PDF Document Builder"""
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

from .report_data import (
    REPORTS_DIR, COEFFICIENTS, HYPERPARAMS, WEATHER_SOURCES,
    TRAIN_DATA, EVAL_DATA,
    OCEAN_BLUE, DEEP_BLUE, CORAL, GOLD, TEAL,
)


def build_pdf(charts):
    pdf_path = REPORTS_DIR / "Lagi_Training_Report.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        title="Lagi Weather Guardian — Training Report",
        author="renles / David",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        "MainTitle", parent=styles["Title"],
        fontSize=28, textColor=HexColor(OCEAN_BLUE),
        spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "SubTitle", parent=styles["Normal"],
        fontSize=14, textColor=HexColor("#666666"),
        spaceAfter=12, alignment=TA_CENTER, fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=18, textColor=HexColor(OCEAN_BLUE),
        spaceBefore=24, spaceAfter=12, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=14, textColor=HexColor(DEEP_BLUE),
        spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=11, leading=15, textColor=black,
        spaceAfter=8, alignment=TA_JUSTIFY, fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        "BulletItem", parent=styles["Normal"],
        fontSize=11, leading=15, textColor=black,
        spaceAfter=4, leftIndent=20, bulletIndent=10,
        fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        "Footer", parent=styles["Normal"],
        fontSize=9, textColor=HexColor("#888888"),
        alignment=TA_CENTER, fontName="Helvetica-Oblique",
    ))

    story = []

    # ── Title Page ──
    story.extend([Spacer(1, 5*cm)])
    story.append(Paragraph("LAGI WEATHER GUARDIAN", styles["MainTitle"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("QLoRA Fine-Tuning Report<br/>Training Results &amp; Technical Analysis", styles["SubTitle"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "<i>First AI Weather Agent Purpose-Built for Fiji and the Pacific Islands</i>",
        ParagraphStyle("TagLine", parent=styles["Normal"], fontSize=13,
                       textColor=HexColor(TEAL), alignment=TA_CENTER)
    ))
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(
        f"March 2026 | renles.com | github.com/renles/lagi-weather",
        styles["Footer"]
    ))
    story.append(PageBreak())

    # Helper to add chart image
    def add_chart(key, width=15*cm):
        if key in charts:
            story.append(Spacer(1, 0.3*cm))
            story.append(RLImage(str(charts[key]), width=width,
                                 height=width * 0.45))
            story.append(Spacer(1, 0.3*cm))

    # ── 1. Executive Summary ──
    story.append(Paragraph("1. Executive Summary", styles["H1"]))
    story.append(Paragraph(
        "Lagi is the first artificial intelligence weather agent purpose-built for Fiji "
        "and the Pacific Islands. Named after the Fijian word for weather and sky, Lagi "
        "addresses a critical gap: global weather models consistently underperform for "
        "small island developing states, systematically under-predicting both temperature "
        "and rainfall across the Pacific.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "This report presents the results of Lagi's initial QLoRA fine-tuning run on "
        "Qwen3.5-9B, a 9-billion parameter large language model. The training achieved "
        "a <b>98.8% reduction in loss</b> (7.53 → 0.089) with zero overfitting, and produced "
        "bias correction coefficients that quantify — for the first time — exactly how "
        "much global models miss for Fiji.",
        styles["Body"]
    ))

    story.append(Paragraph("Key Findings", styles["H2"]))
    findings = [
        "Global models under-predict Fiji temperatures by <b>+0.58°C</b> on average",
        "Global models under-predict Fiji rainfall by <b>+7.6mm</b> on average",
        "ENSO phase introduces a measurable <b>+0.010°C</b> correction",
        "Training loss converged to <b>0.089</b> with eval loss of <b>0.095</b> — gap of only 0.006",
        "5-source ensemble + Monte Carlo simulation delivers calibrated certainty percentages",
    ]
    for f in findings:
        story.append(Paragraph(f"• {f}", styles["BulletItem"]))

    story.append(PageBreak())

    # ── 2. The Problem ──
    story.append(Paragraph("2. The Problem: Why Pacific Islands Need Better Forecasts", styles["H1"]))
    story.append(Paragraph(
        "Global numerical weather prediction models operate on grid cells of 10–25 km resolution. "
        "Fiji's total land area (18,274 km²) is represented by just a handful of grid points "
        "surrounded by ocean. This creates systematic biases in temperature (+0.58°C) and "
        "rainfall (+7.6mm) that affect agriculture, disaster preparedness, and tourism — "
        "the three pillars of Fiji's economy.",
        styles["Body"]
    ))

    story.append(PageBreak())

    # ── 3. Training Methodology ──
    story.append(Paragraph("3. Training Methodology", styles["H1"]))
    story.append(Paragraph(
        "We used QLoRA (Quantised Low-Rank Adaptation) to fine-tune Qwen3.5-9B. This technique "
        "freezes the base model, quantizes to 4-bit precision, and trains small adapter matrices — "
        "reducing memory by ~75% while preserving quality.",
        styles["Body"]
    ))

    # Hyperparams table
    hp_data = [["Parameter", "Value"]]
    for k, v in list(HYPERPARAMS.items())[:12]:
        hp_data.append([k, str(v)])

    hp_table = Table(hp_data, colWidths=[7*cm, 8*cm])
    hp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(OCEAN_BLUE)),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f0f8ff")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(hp_table)

    story.append(PageBreak())

    # ── 4. Training Results ──
    story.append(Paragraph("4. Training Results", styles["H1"]))
    story.append(Paragraph("4.1 Loss Convergence", styles["H2"]))
    story.append(Paragraph(
        "The model achieved rapid convergence, reducing training loss by 98.8% within "
        "the first 100 steps, then continued refining for 1,500+ additional steps.",
        styles["Body"]
    ))
    add_chart("loss_curve", 16*cm)

    story.append(Paragraph("4.2 Learning Rate Schedule", styles["H2"]))
    add_chart("lr_schedule", 14*cm)

    story.append(Paragraph("4.3 Gradient Stability", styles["H2"]))
    story.append(Paragraph(
        "Gradient norms dropped from 4.78 to 0.003, confirming stable convergence.",
        styles["Body"]
    ))
    add_chart("grad_norm", 14*cm)

    story.append(PageBreak())

    story.append(Paragraph("4.4 Overfitting Analysis", styles["H2"]))
    story.append(Paragraph(
        "Train-eval gap of only 0.006 confirms strong generalisation.",
        styles["Body"]
    ))
    add_chart("train_vs_eval", 12*cm)

    story.append(PageBreak())

    # ── 5. Coefficients ──
    story.append(Paragraph("5. Bias Correction Coefficients", styles["H1"]))
    add_chart("coefficients", 16*cm)

    story.append(Paragraph("5.1 Practical Impact", styles["H2"]))
    add_chart("corrections", 16*cm)

    story.append(PageBreak())

    # ── 6. Architecture ──
    story.append(Paragraph("6. System Architecture", styles["H1"]))
    story.append(Paragraph(
        "Lagi consumes forecasts from 5 independent weather services, applies learned "
        "bias corrections, runs Monte Carlo uncertainty quantification, and delivers "
        "warm Fijian natural-language responses with calibrated certainty percentages.",
        styles["Body"]
    ))
    add_chart("architecture", 16*cm)

    story.append(PageBreak())

    # ── 7. Self-Improvement ──
    story.append(Paragraph("7. Self-Improvement Engine", styles["H1"]))
    story.append(Paragraph(
        "Lagi's most powerful feature: she learns from her own mistakes daily. Every prediction "
        "is logged, compared against actuals, and used to refine correction coefficients. "
        "Weekly full retraining ensures the model continuously improves.",
        styles["Body"]
    ))
    add_chart("projection", 14*cm)

    story.append(PageBreak())

    # ── 8. Market Opportunity ──
    story.append(Paragraph("8. Market Opportunity", styles["H1"]))

    tier_data = [
        ["Tier", "Quota", "Price", "Audience"],
        ["Free", "1,000/day", "Free", "Any registered user"],
        ["Boosted Free", "5,000/day", "Free", "Pacific communities"],
        ["Starter", "50,000/mo", "$19/mo", "Developers, small apps"],
        ["Growth", "500,000/mo", "$49/mo", "Resorts, tourism"],
        ["Enterprise", "Unlimited", "Custom", "Government, airlines"],
    ]
    tier_table = Table(tier_data, colWidths=[3*cm, 3*cm, 2.5*cm, 5*cm])
    tier_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(OCEAN_BLUE)),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f0f8ff")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tier_table)

    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "Lagi's architecture is replicable to Tonga, Samoa, Vanuatu, Solomon Islands, and "
        "more — creating a network of AI weather agents across the most climate-vulnerable "
        "region on Earth.",
        styles["Body"]
    ))

    story.append(Spacer(1, 2*cm))

    # Footer
    story.append(HRFlowable(width="80%", color=HexColor("#cccccc")))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Lagi Weather Guardian — Built with care for Fiji and the Pacific<br/>"
        "Part of the '100 Agents in 100 Days' initiative | renles.com<br/>"
        f"Report generated: {datetime.now().strftime('%d %B %Y')}",
        styles["Footer"]
    ))

    doc.build(story)
    print(f"PDF saved: {pdf_path}")
    return pdf_path
