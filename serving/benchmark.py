import argparse
import json
import platform
from pathlib import Path
import time

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = {
    "customerID": "BENCH-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.5,
    "TotalCharges": 786.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure local sequential single-client API latency."
    )
    parser.add_argument("--requests", type=int, default=150)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "benchmark.json"),
        help="Path for the benchmark JSON report.",
    )
    args = parser.parse_args()
    if args.requests < 100:
        raise SystemExit("Use at least 100 measured requests.")

    latencies_ms: list[float] = []
    errors = 0
    with httpx.Client(timeout=10.0) as client:
        warmup = client.post(args.url, json=SAMPLE)
        warmup.raise_for_status()
        start_all = time.perf_counter()
        for _ in range(args.requests):
            started = time.perf_counter()
            try:
                response = client.post(args.url, json=SAMPLE)
                if response.status_code != 200:
                    errors += 1
            except httpx.RequestError:
                errors += 1
            latencies_ms.append((time.perf_counter() - started) * 1000)
        elapsed_seconds = time.perf_counter() - start_all

    values = np.asarray(latencies_ms)
    report = {
        "measured_requests": args.requests,
        "warmup_requests_excluded": 1,
        "errors": errors,
        "latency_ms": {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        },
        "elapsed_seconds": elapsed_seconds,
        "throughput_requests_per_second": float(args.requests / elapsed_seconds),
        "url": args.url,
        "measurement": "sequential, single client, end-to-end HTTP",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output_path}")
    if errors:
        raise SystemExit(
            f"{errors} of {args.requests} measured requests failed; "
            "the report was saved and must not be treated as an all-success benchmark."
        )


if __name__ == "__main__":
    main()
