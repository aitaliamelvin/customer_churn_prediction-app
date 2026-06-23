from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Customer Churn API",
        "model_loaded": True
    }


def test_predict_endpoint():

    customer = {
        "Gender": "Female",
        "Senior_Citizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "Phone_Service": "Yes",
        "Multiple_Lines": "No",
        "Internet_Service": "DSL",
        "Online_Security": "Yes",
        "Online_Backup": "Yes",
        "Device_Protection": "Yes",
        "Tech_Support": "Yes",
        "Streaming_TV": "No",
        "Streaming_Movies": "No",
        "Contract": "Two year",
        "Paperless_Billing": "No",
        "Payment_Method": "Bank transfer (automatic)",
        "Tenure_Months": 60,
        "Monthly_Charges": 55.0,
        "Total_Charges": 3300.0,
        "CLTV": 6000
    }

    response = client.post(
        "/predict",
        json=customer
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "churn_probability" in data

    assert data["prediction"] in [0, 1]

    assert (
        0 <= data["churn_probability"] <= 1
    )