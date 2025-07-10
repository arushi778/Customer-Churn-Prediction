import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import mlflow
from src.utils.save_load import load_obj
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate(test_csv="data/processed/test.csv", model_path="models/model.joblib", scaler_path="models/scaler.joblib"):
    mlflow.set_experiment("churn_evaluation")

    df = pd.read_csv(test_csv)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_encoded = pd.get_dummies(X)

    columns = load_obj("models/columns.joblib")
    model = load_obj(model_path)
    scaler = load_obj(scaler_path)

    # Add missing columns: any that were in train but not in test
    missing_cols = set(columns) - set(X_encoded.columns)
    missing_df = pd.DataFrame(0, index=X_encoded.index, columns=list(missing_cols))
    X_encoded = pd.concat([X_encoded, missing_df], axis=1)

    # Remove extra columns: any that appear in test but not in train
    extra_cols = set(X_encoded.columns) - set(columns)
    X_encoded.drop(columns=extra_cols, inplace=True)

    X_encoded = X_encoded[columns]
    X_scaled = scaler.transform(X_encoded)
    preds = model.predict(X_scaled)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    with mlflow.start_run(run_name="evaluate_model"):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1", f1)

        logger.info(f"Test accuracy: {acc:.4f}, F1-score: {f1:.4f}")

if __name__ == "__main__":
    evaluate()
