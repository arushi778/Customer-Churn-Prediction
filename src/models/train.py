import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path
import mlflow
from src.utils.save_load import save_obj
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train(train_csv="data/processed/train.csv", model_dir="models"):
    mlflow.set_experiment("churn_training")

    df = pd.read_csv(train_csv)
    X = df.drop(columns=["Churn","customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_encoded = pd.get_dummies(X)
    columns = X_encoded.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')

    with mlflow.start_run(run_name="train_model"):
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)
        acc = accuracy_score(y, preds)
        
        mlflow.log_metric("train_accuracy", acc)
        mlflow.sklearn.log_model(model, "random_forest_model")
        logger.info(f"Train accuracy: {acc:.4f}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        save_obj(model, f"{model_dir}/model.joblib")
        save_obj(scaler, f"{model_dir}/scaler.joblib")
        save_obj(columns, "models/columns.joblib")

        logger.info(f"Model and scaler saved to {model_dir}")

    # in train.py (after loading data)
    print(y.value_counts())

if __name__ == "__main__":
    train()
