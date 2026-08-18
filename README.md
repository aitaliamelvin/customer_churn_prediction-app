# 📉 Customer Churn Prediction

An end-to-end Machine Learning application designed to identify customers at risk of churn and support retention decisions with business-oriented insights.

The project combines a scikit-learn pipeline, FastAPI, Streamlit, MLflow, Docker, automated tests and CI.

## 🚀 Live Demo

👉 **Streamlit Application**
https://customerchurnmlprediction.streamlit.app/

👉 **FastAPI Documentation**
https://customer-churn-prediction-app-a18v.onrender.com/docs

---

## 📌 Project Overview

Customer churn prediction is a classification problem where the goal is to identify customers who are likely to leave a service.

This project goes beyond a simple binary prediction.

It provides:

* churn probability
* business decision threshold
* risk level
* risk factors
* protective factors
* customer value at risk
* portfolio comparison
* retention recommendations

The objective is to transform a Machine Learning prediction into an actionable business decision.

---

## ✨ Main Features

### Churn Prediction

The trained model estimates the probability that a customer will churn based on contractual, behavioral and financial information.

Features include:

* gender
* senior citizen status
* partner
* dependents
* phone service
* multiple lines
* internet service
* online security
* online backup
* device protection
* technical support
* streaming services
* contract type
* billing method
* payment method
* tenure
* monthly charges
* total charges
* customer lifetime value

---

## 🧠 Machine Learning Pipeline

The project uses a complete scikit-learn pipeline:

```text
Raw customer data
        ↓
ColumnTransformer
        ↓
Categorical encoding
        +
Numerical scaling
        ↓
Logistic Regression
        ↓
Churn probability
```

The preprocessing pipeline is stored together with the trained model using `joblib`.

This ensures consistent preprocessing between training and production inference.

---

## 🤖 Model

The production model is based on:

**Logistic Regression**

The model was selected because it offers:

* strong predictive performance
* interpretable behavior
* probability outputs
* efficient inference
* suitability for business classification problems

---

## 📊 Model Performance

Performance on the test dataset:

| Metric    |     Score |
| --------- | --------: |
| Accuracy  |     0.803 |
| Precision |     0.646 |
| Recall    |     0.575 |
| F1-score  |     0.608 |
| ROC-AUC   | **0.849** |

At the default classification threshold of `0.50`, the confusion matrix was:

```text
[[917 118]
 [159 215]]
```

This means:

* True positives: 215
* False positives: 118
* False negatives: 159
* True negatives: 917

---

## 🎯 Business Threshold Optimization

For churn prediction, missing a customer who is genuinely at risk can be more costly than contacting a customer who would not actually churn.

For this reason, several decision thresholds were evaluated.

| Threshold | Precision |    Recall |        F1 | False Negatives | False Positives |
| --------: | --------: | --------: | --------: | --------------: | --------------: |
|      0.50 |     0.646 |     0.575 |     0.608 |             159 |             118 |
|      0.45 |     0.606 |     0.628 |     0.617 |             139 |             153 |
|      0.40 |     0.579 |     0.668 |     0.620 |             124 |             182 |
|  **0.35** | **0.561** | **0.717** | **0.629** |         **106** |             210 |
|      0.30 |     0.532 |     0.743 |     0.620 |              96 |             245 |

The final business threshold was set to:

```text
0.35
```

This threshold provides the best F1-score among the tested values while significantly improving recall.

Compared with the default 0.50 threshold:

* Recall increases from **57.5% to 71.7%**
* False negatives decrease from **159 to 106**
* F1-score improves from **0.608 to 0.629**

This trade-off intentionally accepts more false positives in order to detect more customers genuinely at risk.

---

## ⚠️ Risk Levels

The application converts the churn probability into an operational risk level:

```text
Probability < 35%
→ Low risk

35% ≤ Probability < 60%
→ Medium risk

Probability ≥ 60%
→ High risk
```

This classification is used to generate business recommendations inside the dashboard.

---

## 🎯 Business Recommendations

The dashboard provides simple retention guidance depending on the detected risk level.

Examples:

### Low Risk

* no urgent action
* maintain customer experience
* continue monitoring

### Medium Risk

* monitor the account
* consider a loyalty campaign
* propose a personalized offer

### High Risk

* prioritize customer contact
* trigger a retention action
* review contract and service conditions

---

## 🔎 Risk Factors

The Streamlit dashboard also highlights simple business-oriented risk indicators such as:

* month-to-month contract
* electronic check payment
* low tenure
* absence of technical support
* absence of online security
* fiber optic service

It can also surface protective factors such as:

* longer-term contracts
* significant customer tenure
* active technical support

These indicators are heuristic business explanations and should not be interpreted as exact feature attribution from the Machine Learning model.

---

## 💶 Customer Value at Risk

The dashboard estimates an illustrative value-at-risk metric:

```text
Customer Value at Risk
=
CLTV × Churn Probability
```

Example:

```text
CLTV: 3,000
Churn probability: 36.2%

Weighted value at risk:
≈ 1,086
```

This metric is intended as a decision-support indicator, not as a guaranteed financial loss.

---

## 📊 Portfolio Comparison

The application compares the predicted customer risk with the churn rate observed across the dataset.

Example:

```text
Customer churn risk: 36.2%
Portfolio churn rate: 26.5%

Difference:
+9.7 percentage points
```

This helps identify whether a customer is materially more exposed than the average customer profile.

---

## ⚡ FastAPI

The trained pipeline is exposed through a REST API.

### Endpoint

```text
POST /predict
```

Example request:

```json
{
  "Gender": "Male",
  "Senior_Citizen": "No",
  "Partner": "No",
  "Dependents": "No",
  "Phone_Service": "Yes",
  "Multiple_Lines": "No",
  "Internet_Service": "DSL",
  "Online_Security": "No",
  "Online_Backup": "No",
  "Device_Protection": "No",
  "Tech_Support": "No",
  "Streaming_TV": "No",
  "Streaming_Movies": "No",
  "Contract": "Month-to-month",
  "Paperless_Billing": "Yes",
  "Payment_Method": "Electronic check",
  "Tenure_Months": 12,
  "Monthly_Charges": 70.0,
  "Total_Charges": 800.0,
  "CLTV": 3000
}
```

Example response:

```json
{
  "prediction": 1,
  "churn_probability": 0.6171,
  "risk_level": "high",
  "threshold": 0.35
}
```

---

## 🖥️ Streamlit Dashboard

The Streamlit interface provides:

* churn probability
* optimized business threshold
* customer lifetime value
* risk classification
* executive summary
* risk factors
* protective factors
* retention recommendation
* weighted customer value at risk
* comparison with portfolio churn
* customer profile summary

---

## 🧪 Testing

The project includes automated tests for:

* model prediction
* probability output
* API root endpoint
* prediction endpoint

Run tests with:

```bash
pytest
```

---

## 🔬 MLflow

MLflow is used to track:

* model parameters
* model metrics
* training runs
* saved model artifacts

Tracked metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Local MLflow artifacts are excluded from the Git repository.

---

## 🐳 Docker

The project contains dedicated Docker configurations for:

* FastAPI
* Streamlit

Files:

```text
Dockerfile.api
Dockerfile.streamlit
docker-compose.yml
```

This allows the API and dashboard to be containerized independently or run together.

---

## ⚙️ Continuous Integration

The repository includes a GitHub Actions workflow.

It can be used to automatically run the project's tests whenever changes are pushed to the repository.

---

## 🗂️ Project Structure

```text
customer_churn_ml/
│
├── .github/
│   └── workflows/
│
├── app/
│   ├── main.py
│   └── streamlit_app.py
│
├── data/
│   └── raw/
│
├── models/
│   └── churn_pipeline.joblib
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   └── train.py
│
├── tests/
│
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* pandas
* NumPy
* scikit-learn
* joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit

### MLOps / Engineering

* MLflow
* Docker
* GitHub Actions
* pytest

### Deployment

* Render
* Streamlit Community Cloud
* GitHub

---

## ⚙️ Local Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project:

```bash
cd customer_churn_ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python src/train.py
```

The trained pipeline is saved to:

```text
models/churn_pipeline.joblib
```

---

## ⚡ Run FastAPI Locally

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🖥️ Run Streamlit Locally

Make sure FastAPI is running, then launch:

```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Possible Improvements

Future versions could include:

* SHAP feature explanations
* automated retention campaign recommendations
* cost-sensitive threshold optimization
* customer segmentation
* batch churn scoring
* database integration
* authentication
* scheduled prediction jobs
* monitoring of model drift
* automated retraining
* experiment comparison through hosted MLflow
* cloud-native Docker deployment

---

## 🎯 What This Project Demonstrates

This project demonstrates practical skills in:

* classification Machine Learning
* preprocessing pipelines
* business-oriented threshold optimization
* FastAPI development
* Streamlit dashboard development
* API integration
* MLOps
* automated testing
* Docker
* CI
* deployment
* business analytics

---

## 👤 Author

**Melvin Ait-Alia**

Founder of **Optymia**
Digital solutions, automation, data & AI

GitHub:
https://github.com/aitaliamelvin

Live application:
https://customerchurnmlprediction.streamlit.app/

---

## ⚠️ Disclaimer

This project is a technical and educational Machine Learning demonstration.

Predictions are statistical estimates and should not be interpreted as guarantees of future customer behavior.
