FROM apache/airflow:2.8.0

COPY requirements.txt .
COPY dags /opt/airflow/dags
COPY src /opt/airflow/src

USER airflow

RUN pip install --no-cache-dir -r requirements.txt
