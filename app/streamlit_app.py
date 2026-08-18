import requests
import streamlit as st
from pathlib import Path
import pandas as pd

# ==============================
# 1. PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)


# ==============================
# 2. CUSTOM CSS
# ==============================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #5f6368;
        line-height: 1.6;
        max-width: 900px;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7e9ed;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    .risk-card {
        background: #ffffff;
        border: 1px solid #e7e9ed;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border: 1px solid #e5e7eb;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# 3. API URL
# ==============================

API_URL = "http://https://customer-churn-prediction-app-a18v.onrender.com/predict"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "Telco_customer_churn.xlsx"


@st.cache_data
def load_dataset():
    return pd.read_excel(DATA_PATH)


df = load_dataset()

# ==============================
# 4. HEADER
# ==============================

st.markdown(
    """
    <h1>📉 Customer Churn Prediction</h1>

    <div class="hero-subtitle">
        Cette application estime le risque qu'un client quitte le service
        à partir de ses caractéristiques contractuelles, comportementales
        et financières.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="badge">Machine Learning</span>
    <span class="badge">FastAPI</span>
    <span class="badge">Streamlit</span>
    <span class="badge">Business Analytics</span>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ==============================
# 5. SIDEBAR INPUTS
# ==============================

st.sidebar.markdown("## 👤 Profil client")

with st.sidebar.form("customer_form"):

    gender = st.selectbox(
        "Genre",
        ["Male", "Female"],
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"],
    )

    partner = st.selectbox(
        "Partenaire",
        ["No", "Yes"],
    )

    dependents = st.selectbox(
        "Personnes à charge",
        ["No", "Yes"],
    )

    phone_service = st.selectbox(
        "Service téléphonique",
        ["Yes", "No"],
    )

    multiple_lines = st.selectbox(
        "Lignes multiples",
        ["No", "Yes", "No phone service"],
    )

    internet_service = st.selectbox(
        "Service Internet",
        ["DSL", "Fiber optic", "No"],
    )

    online_security = st.selectbox(
        "Sécurité en ligne",
        ["No", "Yes", "No internet service"],
    )

    online_backup = st.selectbox(
        "Sauvegarde en ligne",
        ["No", "Yes", "No internet service"],
    )

    device_protection = st.selectbox(
        "Protection des appareils",
        ["No", "Yes", "No internet service"],
    )

    tech_support = st.selectbox(
        "Support technique",
        ["No", "Yes", "No internet service"],
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"],
    )

    streaming_movies = st.selectbox(
        "Streaming Films",
        ["No", "Yes", "No internet service"],
    )

    contract = st.selectbox(
        "Type de contrat",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
    )

    paperless_billing = st.selectbox(
        "Facturation dématérialisée",
        ["Yes", "No"],
    )

    payment_method = st.selectbox(
        "Mode de paiement",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    tenure_months = st.slider(
        "Ancienneté (mois)",
        0,
        72,
        12,
    )

    monthly_charges = st.number_input(
        "Charges mensuelles",
        min_value=0.0,
        value=70.0,
        step=1.0,
    )

    total_charges = st.number_input(
        "Charges totales",
        min_value=0.0,
        value=800.0,
        step=10.0,
    )

    cltv = st.number_input(
        "Customer Lifetime Value",
        min_value=0,
        value=3000,
        step=100,
    )

    predict_button = st.form_submit_button(
        "Analyser le risque",
        use_container_width=True,
    )


# ==============================
# 6. CALL API
# ==============================

if predict_button:

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
        "Tenure_Months": tenure_months,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "CLTV": cltv,
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        response.raise_for_status()

        result = response.json()

        st.session_state["result"] = result
        st.session_state["customer_payload"] = payload

    except requests.exceptions.RequestException as e:

        st.error("Impossible de contacter l'API FastAPI.")

        with st.expander("Détails techniques"):
            st.code(str(e))


# ==============================
# 7. RESULTS
# ==============================

if "result" in st.session_state:

    result = st.session_state["result"]
    customer = st.session_state["customer_payload"]

    probability = result["churn_probability"]
    risk_level = result["risk_level"]
    threshold = result["threshold"]
    prediction = result["prediction"]

    st.markdown(
        '<div class="section-title">📊 Résultat de l’analyse</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Probabilité de churn",
            f"{probability * 100:.1f} %",
        )

    with col2:
        st.metric(
            "Seuil métier",
            f"{threshold * 100:.0f} %",
        )

    with col3:
        st.metric(
            "CLTV",
            f"{customer['CLTV']:,.0f}",
        )

    st.divider()

    # ==============================
    # 8. RISK LEVEL
    # ==============================

    st.markdown(
        '<div class="section-title">⚠️ Niveau de risque</div>',
        unsafe_allow_html=True,
    )

    if risk_level == "high":

        st.error("🔴 Risque élevé — ce client présente une forte probabilité de churn.")

    elif risk_level == "medium":

        st.warning(
            "🟠 Risque modéré — ce client doit être surveillé et peut bénéficier d'une action de rétention."
        )

    else:

        st.success(
            "🟢 Risque faible — le profil présente actuellement un faible risque de churn."
        )

    # ==============================
    # EXECUTIVE SUMMARY
    # ==============================

    portfolio_churn_rate = df["Churn Value"].mean()

    difference_vs_portfolio = probability - portfolio_churn_rate

    st.markdown(
        '<div class="section-title">🧾 Synthèse</div>',
        unsafe_allow_html=True,
    )

    if risk_level == "high":

        summary = (
            f"Ce client présente un risque élevé de churn "
            f"({probability * 100:.1f} %), soit "
            f"{difference_vs_portfolio * 100:+.1f} points "
            f"par rapport au taux moyen du portefeuille. "
            "Une action de rétention rapide est recommandée."
        )

    elif risk_level == "medium":

        summary = (
            f"Ce client présente un risque modéré de churn "
            f"({probability * 100:.1f} %), soit "
            f"{difference_vs_portfolio * 100:+.1f} points "
            f"par rapport au taux moyen du portefeuille. "
            "Une action de fidélisation ciblée est recommandée."
        )

    else:

        summary = (
            f"Ce client présente un faible risque de churn "
            f"({probability * 100:.1f} %), soit "
            f"{difference_vs_portfolio * 100:+.1f} points "
            f"par rapport au taux moyen du portefeuille. "
            "Aucune action urgente n'est nécessaire."
        )

    st.markdown(
        f"""
        <div class="risk-card">
            {summary}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ==============================
    # FACTEURS DE RISQUE
    # ==============================

    st.markdown(
        '<div class="section-title">🔎 Facteurs de risque détectés</div>',
        unsafe_allow_html=True,
    )

    risk_factors = []
    protective_factors = []

    if customer["Contract"] == "Month-to-month":
        risk_factors.append("Contrat mensuel, généralement plus exposé au churn")
    else:
        protective_factors.append("Contrat longue durée")

    if customer["Internet_Service"] == "Fiber optic":
        risk_factors.append(
            "Service fibre associé à un profil de churn plus élevé dans le dataset"
        )

    if customer["Payment_Method"] == "Electronic check":
        risk_factors.append("Paiement par chèque électronique")

    if customer["Tenure_Months"] < 12:
        risk_factors.append("Faible ancienneté client")
    elif customer["Tenure_Months"] >= 24:
        protective_factors.append("Ancienneté client significative")

    if customer["Tech_Support"] == "No":
        risk_factors.append("Absence de support technique")
    elif customer["Tech_Support"] == "Yes":
        protective_factors.append("Support technique actif")

    if customer["Online_Security"] == "No":
        risk_factors.append("Absence de sécurité en ligne")

    col1, col2 = st.columns(2)

    with col1:

        if risk_factors:
            risk_html = "".join(f"<li>{factor}</li>" for factor in risk_factors)
        else:
            risk_html = "<li>Aucun facteur de risque majeur détecté.</li>"

        st.markdown(
            f"""
            <div class="risk-card">
                <h4>⚠️ Facteurs de risque</h4>
                <ul>
                    {risk_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        if protective_factors:
            protective_html = "".join(
                f"<li>{factor}</li>" for factor in protective_factors
            )
        else:
            protective_html = "<li>Aucun facteur protecteur majeur détecté.</li>"

        st.markdown(
            f"""
            <div class="risk-card">
                <h4>✅ Facteurs protecteurs</h4>
                <ul>
                    {protective_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ==============================
    # 9. BUSINESS RECOMMENDATION
    # ==============================

    st.markdown(
        '<div class="section-title">🎯 Recommandation business</div>',
        unsafe_allow_html=True,
    )

    if risk_level == "high":

        recommendation = (
            "Contacter rapidement le client et proposer une action de rétention ciblée : "
            "offre commerciale personnalisée, amélioration du contrat ou échange avec le support."
        )

    elif risk_level == "medium":

        recommendation = (
            "Mettre le client sous surveillance et envisager une campagne de fidélisation "
            "ou une offre adaptée avant que le risque n'augmente."
        )

    else:

        recommendation = (
            "Aucune action urgente nécessaire. Maintenir une expérience client de qualité "
            "et suivre l'évolution du profil."
        )

    st.markdown(
        f"""
        <div class="risk-card">
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==============================
    # VALEUR CLIENT EXPOSÉE
    # ==============================

    st.markdown(
        '<div class="section-title">💶 Valeur client exposée</div>',
        unsafe_allow_html=True,
    )

    expected_value_at_risk = customer["CLTV"] * probability

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "CLTV",
            f"{customer['CLTV']:,.0f}",
        )

    with col2:
        st.metric(
            "Probabilité de churn",
            f"{probability * 100:.1f} %",
        )

    with col3:
        st.metric(
            "Valeur pondérée à risque",
            f"{expected_value_at_risk:,.0f}",
        )

    st.caption(
        "La valeur pondérée à risque correspond ici à CLTV × probabilité de churn. "
        "Il s'agit d'un indicateur illustratif, pas d'une perte financière certaine."
    )

    st.divider()

    # ==============================
    # COMPARAISON AU PORTEFEUILLE
    # ==============================

    st.markdown(
        '<div class="section-title">📊 Comparaison au portefeuille</div>',
        unsafe_allow_html=True,
    )

    portfolio_churn_rate = df["Churn Value"].mean()

    difference_vs_portfolio = probability - portfolio_churn_rate

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Risque du client",
            f"{probability * 100:.1f} %",
        )

    with col2:
        st.metric(
            "Taux de churn du portefeuille",
            f"{portfolio_churn_rate * 100:.1f} %",
        )

    with col3:
        st.metric(
            "Écart",
            f"{difference_vs_portfolio * 100:+.1f} pts",
        )

    if probability > portfolio_churn_rate:
        st.warning(
            "Ce client présente un risque de churn supérieur à la moyenne observée dans le portefeuille."
        )
    else:
        st.success(
            "Ce client présente un risque de churn inférieur à la moyenne observée dans le portefeuille."
        )

    st.divider()

    # ==============================
    # 10. CUSTOMER PROFILE
    # ==============================

    st.markdown(
        '<div class="section-title">👤 Profil client</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(f"**Contrat :** {customer['Contract']}")

        st.write(f"**Ancienneté :** {customer['Tenure_Months']} mois")

        st.write(f"**Internet :** {customer['Internet_Service']}")

        st.write(f"**Mode de paiement :** {customer['Payment_Method']}")

    with col2:

        st.write(f"**Charges mensuelles :** {customer['Monthly_Charges']:.2f}")

        st.write(f"**Charges totales :** {customer['Total_Charges']:.2f}")

        st.write(f"**CLTV :** {customer['CLTV']}")

        st.write(f"**Prédiction :** {'Churn' if prediction == 1 else 'No churn'}")


# ==============================
# 11. INITIAL STATE
# ==============================

else:

    st.info(
        "👈 Renseignez les caractéristiques du client puis cliquez sur « Analyser le risque »."
    )


# ==============================
# 12. DISCLAIMER
# ==============================

st.divider()

st.caption(
    "Projet de démonstration Machine Learning. "
    "Le score de churn constitue une aide à la décision et non une certitude sur le comportement futur du client."
)
