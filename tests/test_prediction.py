import joblib
import pandas as pd


pipeline = joblib.load(
    "models/churn_pipeline.joblib"
)


def test_prediction_is_valid():

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
        "Payment Method": [
            "Bank transfer (automatic)"
        ],
        "Tenure Months": [60],
        "Monthly Charges": [55],
        "Total Charges": [3300],
        "CLTV": [6000]
    })

    prediction = pipeline.predict(client)[0]

    assert prediction in [0, 1]

def test_probability_between_0_and_1():

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
        "Payment Method": [
            "Bank transfer (automatic)"
        ],
        "Tenure Months": [60],
        "Monthly Charges": [55],
        "Total Charges": [3300],
        "CLTV": [6000]
    })

    probability = (
        pipeline.predict_proba(client)[0][1]
    )

    assert 0 <= probability <= 1