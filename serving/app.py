from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load

from serving.schemas import PredictionRequest, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "artifacts" / "churn_model.joblib"
bundle: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    if not BUNDLE_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {BUNDLE_PATH}. Run `python -m src.train` first."
        )
    loaded_bundle = load(BUNDLE_PATH)
    required_keys = {
        "model",
        "threshold",
        "model_name",
        "features",
        "numeric_features",
        "categorical_features",
    }
    if not isinstance(loaded_bundle, dict) or not required_keys.issubset(loaded_bundle):
        raise RuntimeError("Model artifact is invalid or incomplete.")
    bundle = loaded_bundle
    yield
    bundle = None


app = FastAPI(
    title="Telecom Churn Scoring API",
    version="1.0.0",
    description="Educational churn classification inference service.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded; route customers to manual review.",
        )
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded; route this customer to manual review.",
        )

    row = payload.model_dump(exclude={"customerID"})
    features = bundle["features"]
    X = pd.DataFrame([row], columns=features)
    try:
        probability = float(bundle["model"].predict_proba(X)[0, 1])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="The supplied feature values could not be scored; route to manual review.",
        ) from exc

    threshold = float(bundle["threshold"])
    return PredictionResponse(
        customerID=payload.customerID,
        churn_probability=probability,
        churn_prediction=int(probability >= threshold),
        threshold=threshold,
        model=bundle["model_name"],
    )
