from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# ==============================
# 1. CONFIGURATION
# ==============================

CHURN_THRESHOLD = 0.35

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"

pipeline = joblib.load(MODEL_PATH)


# ==============================
# 2. INPUT SCHEMA
# ==============================


class Customer(BaseModel):
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Tenure_Months: int
    Monthly_Charges: float
    Total_Charges: float
    CLTV: int


# ==============================
# 3. FASTAPI APP
# ==============================

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "API de prédiction du risque de churn client " "avec seuil métier optimisé."
    ),
    version="1.0.0",
)


# ==============================
# 4. ROOT ENDPOINT
# ==============================


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "model_loaded": True,
        "threshold": CHURN_THRESHOLD,
    }


# ==============================
# 5. PREDICTION ENDPOINT
# ==============================


@app.post("/predict")
def predict(customer: Customer):

    client_df = pd.DataFrame(
        [
            {
                "Gender": customer.Gender,
                "Senior Citizen": customer.Senior_Citizen,
                "Partner": customer.Partner,
                "Dependents": customer.Dependents,
                "Phone Service": customer.Phone_Service,
                "Multiple Lines": customer.Multiple_Lines,
                "Internet Service": customer.Internet_Service,
                "Online Security": customer.Online_Security,
                "Online Backup": customer.Online_Backup,
                "Device Protection": customer.Device_Protection,
                "Tech Support": customer.Tech_Support,
                "Streaming TV": customer.Streaming_TV,
                "Streaming Movies": customer.Streaming_Movies,
                "Contract": customer.Contract,
                "Paperless Billing": customer.Paperless_Billing,
                "Payment Method": customer.Payment_Method,
                "Tenure Months": customer.Tenure_Months,
                "Monthly Charges": customer.Monthly_Charges,
                "Total Charges": customer.Total_Charges,
                "CLTV": customer.CLTV,
            }
        ]
    )

    probability = pipeline.predict_proba(client_df)[0][1]

    prediction = int(probability >= CHURN_THRESHOLD)

    if probability >= 0.60:
        risk_level = "high"

    elif probability >= CHURN_THRESHOLD:
        risk_level = "medium"

    else:
        risk_level = "low"

    return {
        "prediction": prediction,
        "churn_probability": round(
            float(probability),
            4,
        ),
        "risk_level": risk_level,
        "threshold": CHURN_THRESHOLD,
    }
