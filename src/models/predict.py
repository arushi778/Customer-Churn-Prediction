import pandas as pd
from src.utils.save_load import load_obj
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def predict(data_csv="data/processed/test.csv", model_path="models/model.joblib", scaler_path="models/scaler.joblib", columns_path="models/columns.joblib", out_csv="data/output/predictions.csv"):
    df = pd.read_csv(data_csv)
    X = df.drop(columns=["Churn"])

    model = load_obj(model_path)
    scaler = load_obj(scaler_path)
    columns = load_obj(columns_path)

    X_encoded = pd.get_dummies(X)

    missing_cols = set(columns) - set(X_encoded.columns)
    missing_df = pd.DataFrame(0, index=X_encoded.index, columns=list(missing_cols))
    X_encoded = pd.concat([X_encoded, missing_df], axis=1)

    # Remove extra columns: any that appear in test but not in train
    extra_cols = set(X_encoded.columns) - set(columns)
    X_encoded.drop(columns=extra_cols, inplace=True)

    X_encoded = X_encoded[columns]
    X_scaled = scaler.transform(X_encoded)
    preds = model.predict(X_scaled)

    Path("data/output").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ "prediction": preds}).to_csv(out_csv, index=False)

    logger.info(f"Predictions saved to {out_csv}")

if __name__ == "__main__":
    predict()
