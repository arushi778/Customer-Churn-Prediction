import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import mlflow
from src.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess(input_csv="data/processed/processed_data.csv", output_dir="data/processed"):
    mlflow.set_experiment("churn_preprocessing")

    with mlflow.start_run(run_name="preprocess_data"):
        df = pd.read_csv(input_csv)
        logger.info(f"Raw data shape: {df.shape}")

        df = df.dropna() 

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train_df.to_csv(f"{output_dir}/train.csv", index=False)
        test_df.to_csv(f"{output_dir}/test.csv", index=False)

        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("output_train_rows", train_df.shape[0])
        mlflow.log_param("output_test_rows", test_df.shape[0])

        logger.info(f"Saved processed train/test data to {output_dir}")

if __name__ == "__main__":
    preprocess()
