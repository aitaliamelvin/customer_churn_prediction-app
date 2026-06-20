import pandas as pd
import joblib

from preprocessing import clean_data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.read_excel(
    "../data/raw/Telco_customer_churn.xlsx"
)

print("Dataset chargé")

df_ml = clean_data(df)

X = df_ml.drop(
    "Churn Value",
    axis=1
)

y = df_ml["Churn Value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

categorical_cols = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_cols = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_cols
        ),
        (
            "num",
            StandardScaler(),
            numerical_cols
        )
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                random_state=42
            )
        )
    ]
)

pipeline.fit(
    X_train,
    y_train
)

print("Modèle entraîné")

joblib.dump(
    pipeline,
    "../models/churn_pipeline.joblib"
)

print("Modèle sauvegardé")