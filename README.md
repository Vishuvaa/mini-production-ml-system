# Mini Production ML System: Telecom Customer Churn

**BITS Pilani WILP — Machine Learning Model Engineering**

## Project summary

This project builds a reproducible classification and inference workflow to help a customer-retention analyst prioritize review of telecom customers whose records resemble those labelled as churned in the IBM Telco sample. The model returns a churn probability and a threshold-based class.

**Scope limitation:** this is a cross-sectional educational dataset, not a time-ordered record of customers observed before churn. The result is not a validated forecast of future churn, and it does not establish that a retention intervention prevents churn.

## Dataset and data quality

- **Source:** [IBM Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d), downloaded from the repository's [CSV URL](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
- **Shape:** 7,043 rows and 21 columns.
- **Target:** `Churn` (`Yes` / `No`), with 1,869 churn and 5,174 non-churn records.
- **Identifiers:** `customerID` is validated for blanks and duplicates, but excluded from model features.
- **Predictors (19):** numeric `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`; categorical `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, and `PaymentMethod`.
- `TotalCharges` is text in the source; 11 blank values are converted to missing numeric values and imputed within the fitted pipeline.
- Invalid target/numeric values, missing identifiers, duplicate IDs, out-of-range tenure, and negative charges fail validation. Categorical missing values are imputed; unseen inference categories are tolerated by the fitted one-hot encoder.
- The raw CSV is not committed. Its SHA-256 was independently calculated from the downloaded 970,457-byte file on 2026-11-12: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. Downloads are rejected if the content changes; verify a new checksum and source revision before intentionally updating it.

## Architecture and workflow

1. Download and checksum-check the source; validate the schema and data quality.
2. Make a reproducible, stratified 70/15/15 train/validation/test split (`random_state=42`).
3. Fit median imputation/scaling for numeric features and most-frequent imputation/one-hot encoding for categorical features inside each model pipeline.
4. Compare a majority-class baseline, balanced Logistic Regression, and balanced Random Forest on validation data. Choose the classifier and maximize validation F1 to select its threshold.
5. Evaluate the selected classifier once on the held-out test partition. Save the serialized model pipeline, threshold, metrics, training reference, confusion matrix, and precision-recall plot.
6. Serve the same fitted preprocessing and classifier through FastAPI or the batch CLI. Compare incoming features with the training reference using PSI and, for numeric features, the two-sample KS test.

The preprocessing is part of the serialized scikit-learn `Pipeline`; it is fitted only during `.fit(X_train, y_train)`. Validation is used for model/threshold selection, and test data is reserved for final evaluation. This prevents preprocessing leakage, but the random split is not a temporal validation strategy.

## Measured model results

The values below were produced by the included run on Python 3.11.4 and scikit-learn 1.5.2. The selected model is **Random Forest**, because it had the highest validation F1 (0.6353) at threshold 0.3722. Its validation F1 edge over Logistic Regression is small; Logistic Regression had higher validation ROC-AUC and PR-AUC.

### Validation comparison

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.3722 | 0.7434 | 0.5097 | 0.8429 | 0.6353 | 0.8353 | 0.6106 |
| Logistic Regression | 0.5397 | 0.7528 | 0.5220 | 0.8036 | 0.6329 | 0.8447 | 0.6308 |
| Majority baseline | 0.5000 | 0.7348 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2652 |

### Held-out test results for the selected model

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Test rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3722 (selected on validation) | 0.7512 | 0.5209 | 0.7972 | 0.6301 | 0.8362 | 0.6624 | 1,057 |

These are sample-specific results, not estimates of future-period performance. Detailed values are in `outputs/metrics.json` and `outputs/model_comparison.csv`.

## Setup and execution (Windows PowerShell)

Run commands from this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
python -m pytest -q
```

Training downloads the dataset if it is absent, verifies the checksum and schema, trains/evaluates the models, and creates `artifacts/churn_model.joblib`, `artifacts/monitoring_reference.json`, and the metrics/plots in `outputs/`. The raw data and model artifacts are ignored by Git and can be rebuilt locally.

To reproduce the PDF after training and after running the benchmark and drift workflows:

```powershell
python -m src.build_report
```

## Online inference API

Start the service in a separate terminal after training:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

- `GET /health` reports readiness; it returns HTTP 503 when no model is loaded.
- `POST /predict` accepts all 19 predictor fields and an optional `customerID`. Missing or extra fields, invalid types, and out-of-range values receive HTTP 422.
- Missing/corrupt model artifacts prevent startup with an explicit error. No default or fabricated prediction is returned.

PowerShell health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example request (the sample customer is illustrative):

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"customerID":"DEMO-001","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":65.5,"TotalCharges":786.0}'
```

Observed live response during verification:

```json
{"customerID":"LIVE-CHECK-001","churn_probability":0.22991762071893426,"churn_prediction":0,"threshold":0.3722444821018916,"model":"RandomForest"}
```

## Batch inference

The sample CSV contains two illustrative records and no labels:

```powershell
python -m serving.predict_batch --input data\sample_customers.csv --output outputs\sample_predictions.csv
```

Provide a CSV with the 19 predictor columns for a real batch; `customerID` is optional and preserved as the output `record_id`. Missing input files, required columns, or model artifacts fail explicitly without writing success-shaped predictions. Scoring/validation failures should be routed to analyst review.

## Benchmark

With the API running, execute:

```powershell
python -m serving.benchmark --requests 150 --url http://127.0.0.1:8000/predict
```

The measured run used 150 sequential requests (one warm-up excluded), with 0 errors:

| p50 latency | p95 latency | p99 latency | Throughput |
|---:|---:|---:|---:|
| 35.18 ms | 45.69 ms | 46.14 ms | 25.91 requests/second |

Environment: Windows 10 (build 10.0.26100), Python 3.11.4, local loopback API. This is a warm, single-client, sequential end-to-end HTTP measurement; it includes local HTTP overhead and says nothing about concurrent load, deployment hardware, network latency, cold starts, or production capacity. Exact results are written to `outputs/benchmark.json`.

## Drift monitoring and operational fallback

For a new batch containing predictor columns:

```powershell
python -m src.monitor --input path\to\new_batch.csv
```

The monitor compares numeric features with reference-quantile bins for PSI and a two-sample KS test; categorical features use PSI. Heuristics: PSI below 0.10 is low, 0.10–0.20 merits investigation, and above 0.20 is an alert; numeric KS p-values below 0.01 merit investigation. The JSON report is saved to `outputs/drift_report.json`.

The executed monitoring check compared the training reference with the disjoint, random held-out test partition (1,057 records), not live customer data. The highest PSI was 0.0235 (`TotalCharges`), and the smallest numeric KS p-value was 0.1116; no configured investigation threshold was crossed. This only describes this split comparison; it is not evidence that production data is stable.

If the model is unavailable, the service fails startup/readiness and clients must route cases to manual review. Invalid or unreliable input is rejected rather than scored; batch processing stops without fabricated outputs. If drift crosses configured thresholds or data quality checks fail, investigate the source and route affected customers to manual review. Do not retrain automatically on drift alone; require verified labels, governance review, and fresh validation.

## Tests and generated deliverables

```powershell
python -m pytest -q
```

The test suite covers pipeline validation/splitting, API predictions, missing/extra/invalid fields, missing values, unknown categories, unavailable models, and drift calculations. The completed run passed **14 tests**; one upstream Starlette/AnyIO deprecation warning remains.

Key files:

```text
mini-production-ml-system/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                     # Downloaded data; not committed
│   └── sample_customers.csv     # Illustrative batch input
├── artifacts/                   # Model/reference; not committed
├── outputs/                     # Metrics, plots, benchmark, drift report
├── serving/
│   ├── app.py
│   ├── schemas.py
│   ├── predict_batch.py
│   └── benchmark.py
├── src/
│   ├── data_pipeline.py
│   ├── train.py
│   ├── monitor.py
│   └── build_report.py
├── tests/
└── submission_report.pdf
```

## Limitations and references

This educational sample is cross-sectional, has no verified prediction timestamp, and cannot support claims of future churn prediction or causal retention impact. The random split can place related customers from the same source distribution in each partition. Threshold choice optimizes validation F1 rather than business cost; operational use would require cost-sensitive analysis, calibration, temporal evaluation, fairness/privacy review, and monitoring of delayed labels. The benchmark is local and sequential.

Dataset source: [IBM Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d). The final report includes a placeholder for the student's GitHub URL because no verified remote is configured.
