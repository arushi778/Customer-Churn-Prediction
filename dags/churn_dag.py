import sys
sys.path.append('/opt/airflow')  
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.data.preprocess import preprocess
from src.models.train import train
from src.models.evaluate import evaluate
from src.models.predict import predict


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='churn_prediction_pipeline',
    default_args=default_args,
    description='ML pipeline for churn prediction',
    schedule_interval=None,  
    catchup=False,
    tags=['churn', 'mlops']
) as dag:

    # Step 1: Preprocessing
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess
    )

    # Step 2: Train model
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train
    )

    # Step 3: Evaluate model
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate
    )

    # Step 4: Batch predict
    batch_predict_task = PythonOperator(
        task_id='batch_predict',
        python_callable=predict
    )

    # Step 5: Notify FastAPI (future; e.g., HTTP request)
    def notify_fastapi():
        print("Pretend to notify FastAPI to reload new model")

    notify_api_task = PythonOperator(
        task_id='notify_fastapi',
        python_callable=notify_fastapi
    )

    # Define pipeline flow
    preprocess_task >> train_task >> evaluate_task >> batch_predict_task >> notify_api_task
