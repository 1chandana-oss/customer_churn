
from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Locate saved model and threshold
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"

# Load model and threshold
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    THRESHOLD = json.load(file)["threshold"]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained XGBoost pipeline.",
    version="1.0.0"
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    customer_df = pd.DataFrame([customer.model_dump()])

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = probability >= THRESHOLD

    return {
        "churn_probability": round(probability, 4),
        "predicted_churn": "Yes" if prediction else "No",
        "decision_threshold": THRESHOLD
    }
