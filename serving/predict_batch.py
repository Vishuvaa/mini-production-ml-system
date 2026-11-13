import argparse
from pathlib import Path

import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "artifacts" / "churn_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score unseen customer records in batch."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    if not BUNDLE_PATH.is_file():
        raise SystemExit(
            f"Model artifact not found at {BUNDLE_PATH}; route records to manual review."
        )
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    data = pd.read_csv(input_path)
    if data.empty:
        raise SystemExit("Input CSV contains no customer rows.")
    bundle = load(BUNDLE_PATH)
    missing = sorted(set(bundle["features"]) - set(data.columns))
    if missing:
        raise SystemExit(
            f"Input is missing required predictor columns: {missing}; "
            "no predictions were written."
        )

    X = data[bundle["features"]].copy()
    try:
        probabilities = bundle["model"].predict_proba(X)[:, 1]
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"Input could not be scored ({type(exc).__name__}); "
            "route these records to manual review."
        ) from exc

    output = pd.DataFrame(
        {
            "record_id": data["customerID"]
            if "customerID" in data
            else range(len(data)),
            "churn_probability": probabilities,
            "churn_prediction": (
                probabilities >= bundle["threshold"]
            ).astype(int),
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Scored {len(output)} records -> {output_path}")


if __name__ == "__main__":
    main()
