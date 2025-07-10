import joblib
import pandas as pd
from pathlib import Path

# Define paths according to your structure
MODEL_PATH = Path("models/model.joblib")
SCALER_PATH = Path("models/scaler.joblib")
COLUMNS_PATH = Path("models/columns.joblib")

# Load the model, scaler, and encoded columns
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
columns = joblib.load(COLUMNS_PATH)

def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Ensure numeric conversion for TotalCharges
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # One-hot encode
    df_encoded = pd.get_dummies(df)

    # Add missing columns (if any)
    for col in columns:
        if col not in df_encoded:
            df_encoded[col] = 0

    # Reorder columns to match training
    df_encoded = df_encoded[columns]

    # Scale
    df_scaled = scaler.transform(df_encoded)

    return df_scaled

def predict_churn(data: dict):
    X = preprocess_input(data)
    proba = model.predict_proba(X)[0][1]
    churn = bool(proba > 0.5)
    return churn, round(proba, 4)