from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import clean_data

# ==============================
# 1. PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Telco_customer_churn.xlsx"

MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"


# ==============================
# 2. LOAD DATA
# ==============================

df = pd.read_excel(DATA_PATH)

print("Dataset chargé")


# ==============================
# 3. CLEAN DATA
# ==============================

df_ml = clean_data(df)

print("Dataset nettoyé")


# ==============================
# 4. FEATURES / TARGET
# ==============================

X = df_ml.drop(
    "Churn Value",
    axis=1,
)

y = df_ml["Churn Value"]


# ==============================
# 5. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# ==============================
# 6. COLUMN TYPES
# ==============================

categorical_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"Colonnes catégorielles : " f"{len(categorical_cols)}")

print(f"Colonnes numériques : " f"{len(numerical_cols)}")


# ==============================
# 7. PREPROCESSING
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
            categorical_cols,
        ),
        (
            "num",
            StandardScaler(),
            numerical_cols,
        ),
    ]
)


# ==============================
# 8. ML PIPELINE
# ==============================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
            ),
        ),
    ]
)


# ==============================
# 9. MLFLOW
# ==============================

mlflow.set_experiment("customer_churn_prediction")


# ==============================
# 10. TRAIN MODEL
# ==============================

with mlflow.start_run():

    pipeline.fit(
        X_train,
        y_train,
    )

    print("\nModèle entraîné")

    # ==============================
    # 11. PREDICTIONS
    # ==============================

    y_pred = pipeline.predict(X_test)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # ==============================
    # 12. METRICS
    # ==============================

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
    )

    recall = recall_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    # ==============================
    # 13. DISPLAY METRICS
    # ==============================

    print("\n--- Model performance ---")

    print(f"Accuracy : {accuracy:.3f}")

    print(f"Precision: {precision:.3f}")

    print(f"Recall   : {recall:.3f}")

    print(f"F1-score : {f1:.3f}")

    print(f"ROC-AUC  : {roc_auc:.3f}")

    # ==============================
    # 14. CLASSIFICATION REPORT
    # ==============================

    print("\n--- Classification report ---")

    print(
        classification_report(
            y_test,
            y_pred,
        )
    )

    # ==============================
    # 15. CONFUSION MATRIX
    # ==============================

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    print("\n--- Confusion matrix ---")

    print(matrix)

    tn, fp, fn, tp = matrix.ravel()

    print("\n--- Business errors ---")

    print(f"True positives : {tp}")

    print(f"False positives: {fp}")

    print(f"False negatives: {fn}")

    print(f"True negatives : {tn}")

    # ==============================
    # 16. THRESHOLD ANALYSIS
    # ==============================

    thresholds = [
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
    ]

    print("\n--- Threshold analysis ---")

    for threshold in thresholds:

        y_pred_threshold = (y_proba >= threshold).astype(int)

        precision_t = precision_score(
            y_test,
            y_pred_threshold,
        )

        recall_t = recall_score(
            y_test,
            y_pred_threshold,
        )

        f1_t = f1_score(
            y_test,
            y_pred_threshold,
        )

        tn_t, fp_t, fn_t, tp_t = confusion_matrix(
            y_test,
            y_pred_threshold,
        ).ravel()

        print(
            f"Threshold {threshold:.2f} | "
            f"Precision: {precision_t:.3f} | "
            f"Recall: {recall_t:.3f} | "
            f"F1: {f1_t:.3f} | "
            f"FN: {fn_t} | "
            f"FP: {fp_t}"
        )

    # ==============================
    # 17. MLFLOW LOGGING
    # ==============================

    mlflow.log_param(
        "model",
        "LogisticRegression",
    )

    mlflow.log_param(
        "max_iter",
        5000,
    )

    mlflow.log_metric(
        "accuracy",
        accuracy,
    )

    mlflow.log_metric(
        "precision",
        precision,
    )

    mlflow.log_metric(
        "recall",
        recall,
    )

    mlflow.log_metric(
        "f1_score",
        f1,
    )

    mlflow.log_metric(
        "roc_auc",
        roc_auc,
    )

    mlflow.sklearn.log_model(
        pipeline,
        "model",
    )

    # ==============================
    # 18. SAVE MODEL
    # ==============================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print(f"\nModèle sauvegardé : " f"{MODEL_PATH}")


print("\nEntraînement terminé ✅")
