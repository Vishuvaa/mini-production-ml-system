from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORT_PATH = ROOT / "submission_report.pdf"
PAGE_SIZE = (8.27, 11.69)
NAVY = "#17324d"
BLUE = "#2878a5"
LIGHT_BLUE = "#eaf2f8"
GREY = "#52606d"


def _read_inputs() -> tuple[dict, pd.DataFrame, dict, dict]:
    required = [
        OUTPUTS / "metrics.json",
        OUTPUTS / "model_comparison.csv",
        OUTPUTS / "confusion_matrix.png",
        OUTPUTS / "precision_recall_curve.png",
        OUTPUTS / "benchmark.json",
        OUTPUTS / "drift_report.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Generate training, benchmark, and monitoring outputs before the report: "
            + ", ".join(missing)
        )
    return (
        json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8")),
        pd.read_csv(OUTPUTS / "model_comparison.csv"),
        json.loads((OUTPUTS / "benchmark.json").read_text(encoding="utf-8")),
        json.loads((OUTPUTS / "drift_report.json").read_text(encoding="utf-8")),
    )


def _new_page(
    pdf: PdfPages, page_number: int, title: str, subtitle: str = ""
) -> tuple[plt.Figure, float]:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    fig.text(0.08, 0.95, title, fontsize=20, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.08, 0.915, subtitle, fontsize=10, color=GREY)
    fig.add_artist(
        plt.Line2D([0.08, 0.92], [0.895, 0.895], color=BLUE, linewidth=1.2)
    )
    fig.text(
        0.08,
        0.035,
        f"BITS Pilani WILP | Machine Learning Model Engineering | Page {page_number} of 8",
        fontsize=8,
        color=GREY,
    )
    return fig, 0.86


def _paragraph(
    fig: plt.Figure,
    text: str,
    y: float,
    *,
    width: int = 102,
    fontsize: float = 10.5,
    color: str = "#202a33",
    line_height: float = 0.024,
) -> float:
    lines = textwrap.wrap(text, width=width, break_long_words=False)
    fig.text(
        0.09,
        y,
        "\n".join(lines),
        fontsize=fontsize,
        color=color,
        va="top",
        linespacing=1.35,
    )
    return y - max(1, len(lines)) * line_height


def _bullets(fig: plt.Figure, items: list[str], y: float) -> float:
    for item in items:
        y = _paragraph(fig, f"- {item}", y, width=96, fontsize=10.2)
        y -= 0.012
    return y


def _table(
    fig: plt.Figure,
    data: list[list[str]],
    headers: list[str],
    bounds: tuple[float, float, float, float],
    fontsize: float = 8.2,
) -> None:
    axis = fig.add_axes(bounds)
    axis.axis("off")
    table = axis.table(
        cellText=data,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d6dee5")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2:
            cell.set_facecolor(LIGHT_BLUE)


def _finish(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _cover(pdf: PdfPages, metrics: dict) -> None:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    fig.add_artist(
        FancyBboxPatch(
            (0.08, 0.63),
            0.84,
            0.25,
            boxstyle="round,pad=0.02",
            facecolor=LIGHT_BLUE,
            edgecolor=BLUE,
            linewidth=1.5,
        )
    )
    fig.text(
        0.12,
        0.80,
        "BITS PILANI WILP",
        fontsize=13,
        color=BLUE,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.745,
        "Mini Production ML System",
        fontsize=25,
        color=NAVY,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.695,
        "Telecom Customer Churn Prediction",
        fontsize=17,
        color=NAVY,
    )
    fig.text(
        0.12,
        0.655,
        "Machine Learning Model Engineering | Academic Project Report",
        fontsize=10.5,
        color=GREY,
    )
    fig.text(0.12, 0.53, "Student name: [ADD YOUR NAME]", fontsize=12, color=NAVY)
    fig.text(0.12, 0.49, "Student ID: [ADD YOUR ID]", fontsize=12, color=NAVY)
    fig.text(
        0.12,
        0.45,
        "GitHub repository: [ADD YOUR FINAL PUBLIC REPOSITORY URL]",
        fontsize=11,
        color=NAVY,
    )
    fig.text(
        0.12,
        0.35,
        (
            f"Selected model: {metrics['model']} | Held-out test F1: "
            f"{metrics['f1']:.4f} | Test rows: {metrics['test_rows']}"
        ),
        fontsize=11,
        color=BLUE,
    )
    fig.text(
        0.08,
        0.035,
        "Prepared from the project code and measured outputs; student identifiers and repository URL require completion.",
        fontsize=8,
        color=GREY,
    )
    _finish(pdf, fig)


def _problem_page(pdf: PdfPages) -> None:
    fig, y = _new_page(
        pdf, 2, "1. Problem formulation", "Business context, target and intended use"
    )
    y = _paragraph(
        fig,
        "Customer churn can reduce recurring revenue and increase replacement costs. "
        "This project demonstrates a small scoring service that can help a retention "
        "analyst prioritize records for review, rather than automatically deciding "
        "which customers receive an intervention.",
        y,
    )
    y -= 0.03
    y = _paragraph(fig, "Formulation", y, fontsize=13, color=NAVY)
    y = _bullets(
        fig,
        [
            "Binary classification target: the sample's Churn label (Yes / No).",
            "Model output: estimated class probability plus a validation-selected threshold and predicted class.",
            "The customer identifier is not used as a predictor.",
            "The target is an observed label in a cross-sectional educational sample; the data does not establish a prediction timestamp.",
        ],
        y,
    )
    y -= 0.02
    y = _paragraph(fig, "Intended use and guardrails", y, fontsize=13, color=NAVY)
    _paragraph(
        fig,
        "A score is a prioritization aid for human review, not a causal estimate or "
        "verified forecast of future churn. Before deployment, the organization would "
        "need temporal validation, verified labels, cost-sensitive threshold analysis, "
        "calibration, privacy/legal review, and fairness assessment.",
        y,
    )
    _finish(pdf, fig)


def _data_page(pdf: PdfPages, metrics: dict) -> None:
    fig, y = _new_page(
        pdf, 3, "2. Data ingestion and validation", "IBM Telco Customer Churn sample"
    )
    y = _paragraph(
        fig,
        "The training command downloads the public CSV over HTTPS, verifies its "
        "independently measured SHA-256, validates the required schema and target, "
        "and fails closed if the source content changes.",
        y,
    )
    y -= 0.02
    _table(
        fig,
        [
            ["Rows / source columns", "7,043 / 21"],
            ["Target", "Churn: 1,869 Yes / 5,174 No"],
            ["Predictors", "19 (4 numeric, 15 categorical)"],
            ["ID handling", "customerID checked for blanks/duplicates; excluded"],
            ["TotalCharges blanks", "11 converted to numeric missing values"],
            ["Dataset SHA-256", metrics["dataset_sha256"][:24] + "..."],
        ],
        ["Validation item", "Observed result"],
        (0.09, 0.54, 0.82, 0.24),
        fontsize=8.5,
    )
    y = _paragraph(
        fig,
        "Categorical blanks use most-frequent imputation; numeric missing values use "
        "training-partition medians. Negative charges, invalid numeric values, "
        "out-of-range tenure, duplicate/blank identifiers, and invalid target labels "
        "are rejected rather than silently coerced into valid records.",
        0.49,
    )
    _paragraph(
        fig,
        "Source: IBM telco-customer-churn-on-icp4d repository. The direct CSV currently "
        "uses the repository's master URL and is protected by the recorded content hash; "
        "upstream changes require explicit re-verification.",
        y - 0.02,
    )
    _finish(pdf, fig)


def _architecture_page(pdf: PdfPages) -> None:
    fig, y = _new_page(
        pdf, 4, "3. Architecture and leakage prevention", "Reproducible training and serving flow"
    )
    axis = fig.add_axes([0.07, 0.33, 0.86, 0.50])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.02, 0.69, 0.19, 0.19, "IBM CSV\nHTTPS + SHA-256"),
        (0.28, 0.69, 0.20, 0.19, "Schema / data\nquality checks"),
        (0.55, 0.69, 0.19, 0.19, "Stratified\n70 / 15 / 15 split"),
        (0.29, 0.34, 0.22, 0.20, "Train-only\npreprocessing +\nmodels"),
        (0.59, 0.34, 0.20, 0.20, "Validation model\nand threshold\nselection"),
        (0.80, 0.34, 0.18, 0.20, "Held-out test\nmetrics + plots"),
        (0.04, 0.03, 0.24, 0.18, "Serialized pipeline\n+ threshold"),
        (0.39, 0.03, 0.24, 0.18, "FastAPI + batch\nscoring"),
        (0.73, 0.03, 0.24, 0.18, "PSI / KS drift\nreport"),
    ]
    for x, box_y, width, height, label in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, box_y),
                width,
                height,
                boxstyle="round,pad=0.012",
                facecolor=LIGHT_BLUE,
                edgecolor=BLUE,
                linewidth=1.2,
            )
        )
        axis.text(
            x + width / 2,
            box_y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=8.6,
            color=NAVY,
        )
    arrows = [
        ((0.21, 0.785), (0.28, 0.785)),
        ((0.48, 0.785), (0.55, 0.785)),
        ((0.65, 0.69), (0.43, 0.54)),
        ((0.51, 0.44), (0.59, 0.44)),
        ((0.79, 0.44), (0.80, 0.44)),
        ((0.70, 0.34), (0.20, 0.21)),
        ((0.28, 0.12), (0.39, 0.12)),
        ((0.63, 0.12), (0.73, 0.12)),
    ]
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1,
                color=GREY,
            )
        )
    _paragraph(
        fig,
        "Leakage control: the input is split before fitting; each imputer, scaler and "
        "encoder is inside the candidate model pipeline and is fitted only by "
        "pipeline.fit(X_train, y_train). Validation selects model and threshold; "
        "the test partition is evaluated once afterward.",
        y,
        width=100,
    )
    _finish(pdf, fig)


def _model_page(pdf: PdfPages, metrics: dict, comparison: pd.DataFrame) -> None:
    fig, _ = _new_page(
        pdf, 5, "4. Model development and evaluation", "Validation comparison and final test results"
    )
    columns = [
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    table_data = []
    for _, row in comparison[columns].iterrows():
        table_data.append(
            [str(row["model"])]
            + [f"{float(row[column]):.3f}" for column in columns[1:]]
        )
    _table(
        fig,
        table_data,
        ["Validation model"] + [name.upper() for name in columns[1:]],
        (0.06, 0.67, 0.88, 0.19),
        fontsize=7.6,
    )
    fig.text(0.09, 0.62, "Held-out test metrics — selected model", fontsize=12, color=NAVY, fontweight="bold")
    _table(
        fig,
        [
            [
                str(metrics["model"]),
                f"{metrics['threshold']:.4f}",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['pr_auc']:.4f}",
            ]
        ],
        ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        (0.06, 0.53, 0.88, 0.10),
        fontsize=7.5,
    )
    axis = fig.add_axes([0.18, 0.10, 0.64, 0.36])
    image = plt.imread(OUTPUTS / "confusion_matrix.png")
    axis.imshow(image)
    axis.axis("off")
    fig.text(
        0.09,
        0.48,
        (
            f"Selected {metrics['model']} at threshold {metrics['threshold']:.4f}, "
            f"chosen using validation F1. The held-out test contains {metrics['test_rows']} "
            "records and is not used for model selection."
        ),
        fontsize=9.4,
        color=GREY,
    )
    _finish(pdf, fig)


def _inference_page(pdf: PdfPages, metrics: dict) -> None:
    fig, y = _new_page(
        pdf, 6, "5. Production inference", "FastAPI, batch CLI and explicit failure behavior"
    )
    y = _bullets(
        fig,
        [
            "GET /health: readiness check; HTTP 503 when no model is loaded.",
            "POST /predict: Pydantic validates required fields, types and ranges; extra fields are forbidden.",
            "The fitted preprocessing and classifier are loaded from one project-relative joblib bundle at startup.",
            "The batch command uses the same serialized pipeline and threshold as online scoring.",
        ],
        y,
    )
    y -= 0.015
    y = _paragraph(fig, "Observed response from a live local API request", y, fontsize=12, color=NAVY)
    response = (
        '{"customerID":"LIVE-CHECK-001",'
        '"churn_probability":0.22991762071893426,'
        '"churn_prediction":0,'
        f'"threshold":{metrics["threshold"]},'
        f'"model":"{metrics["model"]}"' + "}"
    )
    _paragraph(fig, response, y, width=88, fontsize=9.3, color=BLUE)
    y -= 0.04
    y = _paragraph(fig, "Fallback and error handling", y, fontsize=12, color=NAVY)
    _paragraph(
        fig,
        "Missing artifacts stop application startup; an unavailable in-process model "
        "returns 503. Invalid request fields return 422. Batch inputs that are missing, "
        "empty, or lack required columns stop without writing predictions. Route affected "
        "customers to manual review; never substitute a dummy/default probability.",
        y,
    )
    _finish(pdf, fig)


def _monitoring_page(pdf: PdfPages, benchmark: dict, drift: dict) -> None:
    fig, y = _new_page(
        pdf, 7, "6. Benchmark and drift monitoring", "Measured local results and interpretation"
    )
    latency = benchmark["latency_ms"]
    _table(
        fig,
        [
            [
                str(benchmark["measured_requests"]),
                str(benchmark["errors"]),
                f"{latency['p50']:.2f} ms",
                f"{latency['p95']:.2f} ms",
                f"{latency['p99']:.2f} ms",
                f"{benchmark['throughput_requests_per_second']:.2f} req/s",
            ]
        ],
        ["Requests", "Errors", "p50", "p95", "p99", "Throughput"],
        (0.07, 0.70, 0.86, 0.13),
        fontsize=8.3,
    )
    figure_axis = fig.add_axes([0.15, 0.39, 0.70, 0.25])
    drift_features = drift.get("features", {})
    psi_values = {
        feature: item["psi"]
        for feature, item in drift_features.items()
        if "psi" in item
    }
    top = sorted(psi_values.items(), key=lambda item: item[1], reverse=True)[:6]
    figure_axis.bar(
        [feature for feature, _ in top],
        [value for _, value in top],
        color=BLUE,
    )
    figure_axis.axhline(0.10, color="#e39b2f", linestyle="--", label="Investigate: PSI 0.10")
    figure_axis.axhline(0.20, color="#bd3d3a", linestyle=":", label="Alert: PSI 0.20")
    figure_axis.set_ylabel("Population Stability Index")
    figure_axis.set_title("Highest feature PSI: training reference vs. test split")
    figure_axis.tick_params(axis="x", labelrotation=25)
    figure_axis.legend(fontsize=7)
    figure_axis.grid(axis="y", alpha=0.2)
    y = _paragraph(
        fig,
        (
            f"Benchmark environment: {benchmark['platform']}; Python "
            f"{benchmark['python_version']}. The run was single-client, sequential, "
            "local loopback HTTP; it is not a capacity or production SLA test."
        ),
        0.34,
        fontsize=9.4,
    )
    alert_count = sum(
        item.get("status") in ("alert", "investigate", "error")
        for item in drift_features.values()
    )
    _paragraph(
        fig,
        (
            f"Drift check: {drift['rows']} disjoint held-out test records compared with "
            f"the training reference; {alert_count} features crossed the configured PSI/KS "
            "investigation/alert rules. This is a split-to-split diagnostic, not live "
            "production monitoring or a claim that drift is absent."
        ),
        y - 0.01,
        fontsize=9.4,
    )
    _finish(pdf, fig)


def _limitations_page(pdf: PdfPages) -> None:
    fig, y = _new_page(
        pdf, 8, "7. Limitations and conclusion", "Responsible interpretation and next validation steps"
    )
    y = _paragraph(
        fig,
        "The project delivers a reproducible baseline-to-serving workflow with validated "
        "data ingestion, train-only preprocessing, a validation-selected classifier and "
        "threshold, held-out metrics, an API, batch inference, and drift diagnostics.",
        y,
    )
    y -= 0.02
    y = _bullets(
        fig,
        [
            "The IBM sample is cross-sectional and cannot establish future-period generalization or causal impact.",
            "A random stratified split is appropriate for this exercise but does not replace temporal evaluation.",
            "The threshold maximizes F1, not expected business utility; costs and intervention capacity were not supplied.",
            "Demographic/proxy features require fairness, legal-basis and privacy review before real use.",
            "The latency run is a local sequential check, not a production load test.",
            "Drift was checked on a held-out sample only; production monitoring requires new covariates, delayed labels and ongoing review.",
        ],
        y,
    )
    y -= 0.01
    y = _paragraph(fig, "References and submission fields", y, fontsize=12, color=NAVY)
    y = _paragraph(
        fig,
        "Dataset: IBM, telco-customer-churn-on-icp4d, "
        "https://github.com/IBM/telco-customer-churn-on-icp4d",
        y,
        fontsize=9.5,
    )
    y = _paragraph(fig, "Student name: [ADD YOUR NAME]", y, fontsize=9.5)
    y = _paragraph(fig, "Student ID: [ADD YOUR ID]", y, fontsize=9.5)
    _paragraph(
        fig,
        "Final public GitHub repository: [ADD YOUR VERIFIED REPOSITORY URL]",
        y,
        fontsize=9.5,
    )
    _finish(pdf, fig)


def main() -> None:
    metrics, comparison, benchmark, drift = _read_inputs()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(REPORT_PATH) as pdf:
        pdf.infodict()["Title"] = "Telecom Customer Churn Prediction — BITS Pilani WILP"
        pdf.infodict()["Author"] = "Student author to be supplied"
        pdf.infodict()["Subject"] = "Machine Learning Model Engineering project report"
        _cover(pdf, metrics)
        _problem_page(pdf)
        _data_page(pdf, metrics)
        _architecture_page(pdf)
        _model_page(pdf, metrics, comparison)
        _inference_page(pdf, metrics)
        _monitoring_page(pdf, benchmark, drift)
        _limitations_page(pdf)
    print(f"Generated 8-page report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
