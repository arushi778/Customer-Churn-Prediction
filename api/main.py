from fastapi import FastAPI
from api.schema import ChurnRequest
from api.inference import predict_churn
from prometheus_client import Counter, Summary, generate_latest
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total number of API requests')
LATENCY = Summary('prediction_latency_seconds', 'Time spent on predictions')

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")
@LATENCY.time()
def predict(data: ChurnRequest):
    REQUEST_COUNT.inc()
    churn, proba = predict_churn(data.dict())
    return {"churn": churn, "probability": proba}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest()
