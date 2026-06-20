from fastapi import FastAPI
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

pipeline = joblib.load(
    "models/churn_pipeline.joblib"
)

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

app = FastAPI()

@app.get("/")
def home():

    return {
        "message": "Customer Churn API",
        "model_loaded": True
    }

@app.post("/predict")
def predict(customer: Customer):

    client_df = pd.DataFrame([{
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
        "CLTV": customer.CLTV
    }])

    prediction = int(
        pipeline.predict(client_df)[0]
    )

    probability = float(
        pipeline.predict_proba(client_df)[0][1]
    )

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4)
    }