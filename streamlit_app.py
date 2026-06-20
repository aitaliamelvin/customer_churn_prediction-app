import streamlit as st
import requests

st.title("📉 Customer Churn Prediction")

st.write(
    "Prédisez le risque de départ d'un client."
)

st.header("📋 Informations client")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col2:

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

st.header("📡 Services")

col1, col2 = st.columns(2)

with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

col1, col2 = st.columns(2)

with col1:

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col2:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

st.header("💳 Facturation")

col1, col2 = st.columns(2)

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.header("📈 Historique")

col1, col2 = st.columns(2)

with col1:

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

with col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0
    )

    cltv = st.number_input(
        "CLTV",
        min_value=0,
        value=4000
    )


if st.button("Tester l'API"):

    payload = {
        "Gender": gender,
        "Senior_Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Phone_Service": phone_service,
        "Multiple_Lines": multiple_lines,
        "Internet_Service": internet_service,
        "Online_Security": online_security,
        "Online_Backup": online_backup,
        "Device_Protection": device_protection,
        "Tech_Support": tech_support,
        "Streaming_TV": streaming_tv,
        "Streaming_Movies": streaming_movies,
        "Contract": contract,
        "Paperless_Billing": paperless_billing,
        "Payment_Method": payment_method,
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "CLTV": cltv
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    result = response.json()

    probability = result["churn_probability"] * 100

    st.subheader("Résultat")

    st.metric(
    "Probabilité de churn",
    f"{probability:.2f}%"
)

    st.progress(min(probability / 100, 1.0))

    st.write(
        f"Prédiction du modèle : {result['prediction']}"
    )

    if probability >= 40:

        st.error(
            "⚠️ Risque élevé de départ"
        )

    elif probability >= 20:

        st.warning(
            "🟡 Risque modéré"
        )

    else:

        st.success(
            "✅ Client fidèle"
        )
    
  

