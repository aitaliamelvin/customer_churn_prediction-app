import joblib
import pandas as pd

pipeline = joblib.load(
    "../models/churn_pipeline.joblib"
)

print("Pipeline chargé avec succès")

client = pd.DataFrame({
    "Gender": ["Female"],
    "Senior Citizen": ["No"],
    "Partner": ["Yes"],
    "Dependents": ["Yes"],
    "Phone Service": ["Yes"],
    "Multiple Lines": ["No"],
    "Internet Service": ["DSL"],
    "Online Security": ["Yes"],
    "Online Backup": ["Yes"],
    "Device Protection": ["Yes"],
    "Tech Support": ["Yes"],
    "Streaming TV": ["No"],
    "Streaming Movies": ["No"],
    "Contract": ["Two year"],
    "Paperless Billing": ["No"],
    "Payment Method": ["Bank transfer (automatic)"],
    "Tenure Months": [60],
    "Monthly Charges": [55],
    "Total Charges": [3300],
    "CLTV": [6000]
})

prediction = pipeline.predict(client)[0]

probability = pipeline.predict_proba(client)[0][1]

print("Prédiction :", prediction)
print("Probabilité de churn :", round(probability * 100, 2), "%")