"""Basic end-to-end and data-integrity tests.

Run from the repository root with:  python -m pytest -q
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_data_file_integrity():
    df = pd.read_csv(ROOT / "data" / "synthetic_survival_data.csv")
    required = {"time_months", "event", "biomarker_positive", "age", "stage"}
    assert required.issubset(df.columns)
    assert (df["time_months"] > 0).all()
    assert set(df["event"].unique()) <= {0, 1}
    assert set(df["biomarker_positive"].unique()) <= {0, 1}
    assert df["stage"].isin(["III", "IV"]).all()
    assert len(df) >= 50  # enough events for a stable Cox fit


def test_script_runs_and_produces_outputs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "survival_analysis.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

    km_fig = ROOT / "figures" / "km_survival_by_biomarker.png"
    cox_csv = ROOT / "results" / "cox_model_summary.csv"
    assert km_fig.exists() and km_fig.stat().st_size > 0
    assert cox_csv.exists()


def test_cox_model_output_is_sane():
    cox_csv = ROOT / "results" / "cox_model_summary.csv"
    if not cox_csv.exists():
        test_script_runs_and_produces_outputs()
    summary = pd.read_csv(cox_csv, index_col=0)
    assert {"coef", "exp(coef)", "p"}.issubset(summary.columns)
    assert {"biomarker_positive", "age", "stage_IV"} <= set(summary.index)
    assert (summary["exp(coef)"] > 0).all()   # hazard ratios must be positive and finite
    assert summary["coef"].notna().all()
    assert summary["p"].between(0, 1).all()
