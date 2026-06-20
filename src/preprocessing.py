import pandas as pd


def clean_data(df):

    df_clean = df.copy()

    columns_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "Churn Label",
        "Churn Score",
        "Churn Reason"
    ]

    existing_cols = [
        col for col in columns_to_drop
        if col in df_clean.columns
    ]

    df_clean = df_clean.drop(columns=existing_cols)

    if "Total Charges" in df_clean.columns:
        df_clean["Total Charges"] = pd.to_numeric(
            df_clean["Total Charges"],
            errors="coerce"
        )

        df_clean["Total Charges"] = (
            df_clean["Total Charges"]
            .fillna(0)
        )

    columns_to_drop_v2 = [
        "City",
        "Zip Code",
        "Latitude",
        "Longitude",
        "Lat Long"
    ]

    existing_cols_v2 = [
        col for col in columns_to_drop_v2
        if col in df_clean.columns
    ]

    df_clean = df_clean.drop(
        columns=existing_cols_v2
    )

    return df_clean