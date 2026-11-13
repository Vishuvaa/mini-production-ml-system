import pytest

from src.monitor import _numeric_metrics, psi
import pandas as pd


def test_psi_is_zero_for_identical_distributions():
    distribution = {"No": 0.75, "Yes": 0.25}
    assert psi(distribution, distribution) == pytest.approx(0.0)


def test_numeric_drift_reports_missing_and_invalid_rows():
    metrics = _numeric_metrics(
        [0.0, 1.0, 2.0, 3.0],
        pd.Series([0.0, 1.0, "bad", None], dtype="object"),
    )
    assert metrics["rows_with_invalid_values"] == 1
    assert metrics["missing_or_blank_rows"] == 1
    assert metrics["ks_pvalue"] >= 0
