from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customerID: str | None = Field(default=None, min_length=1, strict=True)
    gender: str = Field(min_length=1, strict=True)
    SeniorCitizen: int = Field(ge=0, le=1, strict=True)
    Partner: str = Field(min_length=1, strict=True)
    Dependents: str = Field(min_length=1, strict=True)
    tenure: int = Field(ge=0, le=72, strict=True)
    PhoneService: str = Field(min_length=1, strict=True)
    MultipleLines: str = Field(min_length=1, strict=True)
    InternetService: str = Field(min_length=1, strict=True)
    OnlineSecurity: str = Field(min_length=1, strict=True)
    OnlineBackup: str = Field(min_length=1, strict=True)
    DeviceProtection: str = Field(min_length=1, strict=True)
    TechSupport: str = Field(min_length=1, strict=True)
    StreamingTV: str = Field(min_length=1, strict=True)
    StreamingMovies: str = Field(min_length=1, strict=True)
    Contract: str = Field(min_length=1, strict=True)
    PaperlessBilling: str = Field(min_length=1, strict=True)
    PaymentMethod: str = Field(min_length=1, strict=True)
    MonthlyCharges: float = Field(ge=0, allow_inf_nan=False)
    TotalCharges: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    customerID: str | None
    churn_probability: float
    churn_prediction: int
    threshold: float
    model: str
