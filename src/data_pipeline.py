from __future__ import annotations

from io import BytesIO
import hashlib
import os
from pathlib import Path
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# Independently calculated from the 970,457-byte CSV downloaded on 2026-11-12.
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COL] + FEATURES + [TARGET]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Dataset contains no rows.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = frame[REQUIRED_COLUMNS].copy()
    customer_ids = df[ID_COL].astype("string").fillna("")
    if customer_ids.str.strip().eq("").any():
        raise ValueError("customerID contains missing or blank values.")
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate customerID values detected.")

    target = df[TARGET].astype("string").str.strip()
    df[TARGET] = target.map({"No": 0, "Yes": 1})
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values or values other than Yes/No.")

    for column in NUMERIC_FEATURES:
        raw = df[column]
        text = raw.astype("string").str.strip()
        blank = raw.isna() | text.eq("")
        numeric = pd.to_numeric(text.mask(blank), errors="coerce")
        invalid = ~blank & numeric.isna()
        if invalid.any():
            examples = text[invalid].head(3).tolist()
            raise ValueError(f"{column} contains invalid numeric values: {examples}")
        df[column] = numeric

    if not df["SeniorCitizen"].dropna().isin([0, 1]).all():
        raise ValueError("SeniorCitizen values must be 0, 1, or missing.")
    if not df["tenure"].dropna().between(0, 72).all():
        raise ValueError("tenure values must be between 0 and 72 months.")
    for column in ("MonthlyCharges", "TotalCharges"):
        if (df[column].dropna() < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    for column in CATEGORICAL_FEATURES:
        text = df[column].astype("string").str.strip()
        df[column] = text.mask(text.eq(""))

    return df


def download_dataset(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        actual = sha256_file(DATA_PATH)
        if actual != EXPECTED_SHA256:
            raise ValueError(
                f"Dataset checksum mismatch. Expected {EXPECTED_SHA256}, got {actual}. "
                "The existing file was not replaced; inspect it or rerun with force=True."
            )
        _validate_dataframe(pd.read_csv(DATA_PATH))
        return DATA_PATH

    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "mini-production-ml-system/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Dataset download returned HTTP {response.status}.")
        payload = response.read()

    actual = hashlib.sha256(payload).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(
            f"Downloaded dataset checksum mismatch. Expected {EXPECTED_SHA256}, "
            f"got {actual}. The downloaded file was not saved."
        )
    _validate_dataframe(pd.read_csv(BytesIO(payload)))

    temporary_path = DATA_PATH.with_suffix(".csv.tmp")
    temporary_path.write_bytes(payload)
    os.replace(temporary_path, DATA_PATH)
    return DATA_PATH


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else download_dataset()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    return _validate_dataframe(pd.read_csv(csv_path))


def split_data(df: pd.DataFrame, seed: int = 42):
    """Return a reproducible stratified 70/15/15 split (not a temporal split)."""
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test
