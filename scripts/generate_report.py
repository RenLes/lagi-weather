#!/usr/bin/env python3
"""
Lagi Weather Guardian — Training Report Generator
==================================================
Produces comprehensive Word (.docx) and PDF reports with charts.

This is the entry point. The implementation is split across:
  - reports/report_data.py     — Training data, constants, shared config
  - reports/chart_generator.py — All chart generation functions
  - reports/docx_builder.py    — Word document builder
  - reports/pdf_builder.py     — PDF document builder
"""

from reports.report_data import CHARTS_DIR
from reports.chart_generator import (
    chart_loss_curve, chart_lr_schedule, chart_grad_norm,
    chart_train_vs_eval, chart_coefficients, chart_corrections,
    chart_architecture, chart_improvement_projection,
)
from reports.docx_builder import build_docx
from reports.pdf_builder import build_pdf


def main():
    print("=" * 60)
    print("  LAGI TRAINING REPORT GENERATOR")
    print("=" * 60)

    # Generate all charts
    print("\nGenerating charts...")
    charts = {}

    print("  [1/8] Loss curve...")
    charts["loss_curve"] = chart_loss_curve()

    print("  [2/8] Learning rate schedule...")
    charts["lr_schedule"] = chart_lr_schedule()

    print("  [3/8] Gradient norm...")
    charts["grad_norm"] = chart_grad_norm()

    print("  [4/8] Train vs eval...")
    charts["train_vs_eval"] = chart_train_vs_eval()

    print("  [5/8] Coefficients...")
    charts["coefficients"] = chart_coefficients()

    print("  [6/8] Corrections...")
    charts["corrections"] = chart_corrections()

    print("  [7/8] Architecture...")
    charts["architecture"] = chart_architecture()

    print("  [8/8] Improvement projection...")
    charts["projection"] = chart_improvement_projection()

    print(f"\n  All charts saved to {CHARTS_DIR}")

    # Generate documents
    print("\nGenerating Word document...")
    docx_path = build_docx(charts)

    print("Generating PDF...")
    pdf_path = build_pdf(charts)

    print("\n" + "=" * 60)
    print("  REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Word: {docx_path}")
    print(f"  PDF:  {pdf_path}")
    print(f"  Charts: {CHARTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
