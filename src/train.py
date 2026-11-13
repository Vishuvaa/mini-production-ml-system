from __future__ import annotations

import json
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_pipeline import (
    CATEGORICAL_FEATURES,
    DATA_URL,
    FEATURES,
    NUMERIC_FEATURES,
    ROOT,
    load_data,
    sha256_file,
    split_data,
)

ARTIFACTS = ROOT / "artifacts"
OUTPUTS = ROOT / "outputs"
SEED = 42


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def metric_row(
    name: str, y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float | str]:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "model": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def _save_evaluation_plots(
    model_name: str, y_test: pd.Series, probabilities: np.ndarray, threshold: float
) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"Test confusion matrix — {model_name}")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], labels=["No churn", "Churn"])
    axis.set_yticks([0, 1], labels=["No churn", "Churn"])
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(str(OUTPUTS / "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(
        recall,
        precision,
        label=f"Average precision = {average_precision_score(y_test, probabilities):.3f}",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Held-out test precision–recall curve")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(OUTPUTS / "precision_recall_curve.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df, SEED)
    print(
        f"Rows={len(df)} | train={len(X_train)} valid={len(X_valid)} "
        f"test={len(X_test)}"
    )
    print(
        f"Churn rate: {y_train.mean():.3%} train / {y_valid.mean():.3%} validation / "
        f"{y_test.mean():.3%} test"
    )

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
    }
    fitted: dict[str, tuple[Pipeline, float]] = {}
    rows: list[dict[str, float | str]] = []
    training_seconds: dict[str, float] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [("preprocess", build_preprocessor()), ("model", estimator)]
        )
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_seconds[name] = time.perf_counter() - started
        validation_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        threshold = choose_threshold(y_valid, validation_probabilities)
        row = metric_row(name, y_valid, validation_probabilities, threshold)
        rows.append(row)
        fitted[name] = (pipeline, threshold)
        print(
            f"{name}: validation F1={row['f1']:.4f}; threshold={threshold:.3f}; "
            f"train={training_seconds[name]:.2f}s"
        )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    baseline_probabilities = np.repeat(float(y_train.mean()), len(y_valid))
    rows.append(metric_row("MajorityBaseline", y_valid, baseline_probabilities))

    comparison = pd.DataFrame(rows).sort_values(
        "f1", ascending=False, kind="stable"
    )
    comparison.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    if best_name == "MajorityBaseline":
        raise RuntimeError("The majority baseline ranked first; inspect the model pipeline.")
    best_pipeline, best_threshold = fitted[best_name]

    # The test partition is evaluated once, after model and threshold selection.
    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= best_threshold).astype(int)
    test_metrics: dict[str, object] = dict(
        metric_row(best_name, y_test, test_probabilities, best_threshold)
    )
    test_metrics.update(
        {
            "test_rows": int(len(y_test)),
            "test_churn_rate": float(y_test.mean()),
            "training_seconds": float(training_seconds[best_name]),
            "threshold_selected_on": "validation",
            "selection_metric": "validation F1",
            "split": "stratified 70/15/15; random_state=42",
            "dataset_rows": int(len(df)),
            "dataset_sha256": sha256_file(
                ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
            ),
            "dataset_url": DATA_URL,
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "classification_report": classification_report(
                y_test, test_predictions, output_dict=True, zero_division=0
            ),
        }
    )
    (OUTPUTS / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print("\nTEST METRICS")
    print(
        json.dumps(
            {
                key: value
                for key, value in test_metrics.items()
                if key != "classification_report"
            },
            indent=2,
        )
    )
    print("\nClassification report")
    print(classification_report(y_test, test_predictions, zero_division=0))
    _save_evaluation_plots(best_name, y_test, test_probabilities, best_threshold)

    bundle = {
        "model": best_pipeline,
        "threshold": best_threshold,
        "model_name": best_name,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    dump(bundle, ARTIFACTS / "churn_model.joblib")

    reference = {}
    reference_frame = X_train.sample(min(2000, len(X_train)), random_state=SEED)
    for column in FEATURES:
        series = reference_frame[column]
        if column in NUMERIC_FEATURES:
            values = pd.to_numeric(series, errors="coerce").dropna().tolist()
            reference[column] = {"type": "numeric", "values": values}
        else:
            counts = (
                series.fillna("__MISSING__")
                .astype(str)
                .value_counts(normalize=True)
            )
            reference[column] = {
                "type": "categorical",
                "distribution": counts.to_dict(),
            }
    (ARTIFACTS / "monitoring_reference.json").write_text(
        json.dumps(reference, indent=2)
    )
    print(f"\nSaved model to {ARTIFACTS / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
