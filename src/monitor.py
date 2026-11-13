from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "artifacts" / "monitoring_reference.json"
OUTPUT_PATH = ROOT / "outputs" / "drift_report.json"
EPSILON = 1e-6


def psi(expected: dict, actual: dict) -> float:
    keys = set(expected) | set(actual)
    expected_total = sum(expected.values()) or 1
    actual_total = sum(actual.values()) or 1
    score = 0.0
    for key in keys:
        expected_share = max(expected.get(key, 0) / expected_total, EPSILON)
        actual_share = max(actual.get(key, 0) / actual_total, EPSILON)
        score += (actual_share - expected_share) * np.log(
            actual_share / expected_share
        )
    return float(score)


def _numeric_metrics(reference: list[float], batch: pd.Series) -> dict:
    source = batch.astype("string").str.strip()
    blank = batch.isna() | source.eq("")
    numeric = pd.to_numeric(source.mask(blank), errors="coerce")
    invalid = (~blank & numeric.isna()).sum()
    current = numeric.dropna().to_numpy(dtype=float)
    baseline = np.asarray(reference, dtype=float)
    if len(current) == 0:
        return {
            "error": "no numeric observations",
            "rows_with_invalid_values": int(invalid),
            "missing_or_blank_rows": int(blank.sum()),
        }

    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    expected_counts, _ = np.histogram(baseline, bins=edges)
    actual_counts, _ = np.histogram(current, bins=edges)
    value = psi(dict(enumerate(expected_counts)), dict(enumerate(actual_counts)))
    ks_statistic, ks_pvalue = cast(
        tuple[float, float], ks_2samp(baseline, current)
    )
    return {
        "psi": value,
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "rows_with_invalid_values": int(invalid),
        "missing_or_blank_rows": int(blank.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare an input CSV with the training feature reference."
    )
    parser.add_argument("--input", required=True, help="New batch CSV.")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help=f"Report JSON path (default: {OUTPUT_PATH}).",
    )
    args = parser.parse_args()

    if not REFERENCE_PATH.is_file():
        raise SystemExit(
            f"Monitoring reference not found at {REFERENCE_PATH}; train the model first."
        )
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")
    batch = pd.read_csv(input_path)
    if batch.empty:
        raise SystemExit("Input CSV contains no rows; no drift result can be computed.")
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    report = {
        "rows": int(len(batch)),
        "reference": "training partition sample",
        "features": {},
    }

    for column, spec in reference.items():
        if column not in batch:
            report["features"][column] = {"error": "missing feature column"}
            continue
        if spec["type"] == "numeric":
            metrics = _numeric_metrics(spec["values"], batch[column])
        else:
            source = batch[column].astype("string").str.strip()
            normalized = source.mask(source.eq("")).fillna("__MISSING__").astype(str)
            actual = normalized.value_counts().to_dict()
            metrics = {
                "psi": psi(spec["distribution"], actual),
                "missing_or_blank_rows": int(
                    (batch[column].isna() | source.eq("")).sum()
                ),
            }
        if "psi" in metrics:
            value = metrics["psi"]
            status = "alert" if value > 0.20 else ("investigate" if value >= 0.10 else "low")
            if (
                metrics.get("ks_pvalue") is not None
                and metrics["ks_pvalue"] < 0.01
                and status == "low"
            ):
                status = "investigate"
            metrics["status"] = status
        report["features"][column] = metrics

    statuses = [
        item.get("status", "error") for item in report["features"].values()
    ]
    if any(status in ("alert", "investigate", "error") for status in statuses):
        report["recommendation"] = (
            "Review data quality and route affected customers to manual review "
            "before relying on scores."
        )
    else:
        report["recommendation"] = (
            "No configured drift threshold was crossed; continue monitoring."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
