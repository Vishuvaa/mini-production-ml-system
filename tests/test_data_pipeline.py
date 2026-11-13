import pandas as pd
import pytest

from src.data_pipeline import (
    CATEGORICAL_FEATURES,
    REQUIRED_COLUMNS,
    load_data,
    split_data,
)


def _rows(count: int = 2) -> pd.DataFrame:
    rows = []
    for index in range(count):
        row = {
            "customerID": f"C-{index}",
            "SeniorCitizen": index % 2,
            "tenure": index % 73,
            "MonthlyCharges": 50 + index,
            "TotalCharges": " " if index == 0 else str(50 + index),
            "Churn": "Yes" if index % 2 else "No",
        }
        row.update({feature: "Yes" for feature in CATEGORICAL_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_load_data_converts_blank_total_charges_to_missing(tmp_path):
    source = tmp_path / "input.csv"
    _rows().to_csv(source, index=False)
    frame = load_data(source)
    assert pd.isna(frame.loc[0, "TotalCharges"])
    assert frame["Churn"].tolist() == [0, 1]


def test_load_data_rejects_duplicate_customer_ids(tmp_path):
    frame = _rows()
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Duplicate customerID"):
        load_data(source)


def test_load_data_rejects_invalid_numeric_values(tmp_path):
    frame = _rows()
    frame["MonthlyCharges"] = frame["MonthlyCharges"].astype(object)
    frame.loc[0, "MonthlyCharges"] = "not-a-number"
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    with pytest.raises(ValueError, match="MonthlyCharges contains invalid numeric values"):
        load_data(source)


def test_split_data_is_reproducible_and_stratified():
    frame = validate_frame(_rows(100))
    first = split_data(frame, seed=42)
    second = split_data(frame, seed=42)
    X_train, X_valid, X_test, y_train, y_valid, y_test = first
    assert (len(X_train), len(X_valid), len(X_test)) == (70, 15, 15)
    assert y_train.mean() == pytest.approx(0.5)
    for target in (y_train, y_valid, y_test):
        counts = target.value_counts()
        assert counts.max() - counts.min() <= 1
    for index in range(3):
        pd.testing.assert_frame_equal(first[index], second[index])
    for index in range(3, 6):
        pd.testing.assert_series_equal(first[index], second[index])


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    from src.data_pipeline import _validate_dataframe

    return _validate_dataframe(frame)
