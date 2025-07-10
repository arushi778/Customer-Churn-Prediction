from fastapi import FastAPI, Request
from prometheus_client import Counter, Summary, Histogram, generate_latest
from starlette.responses import Response
import time

app = FastAPI()

REQUEST_COUNT = Counter("api_requests_total", "Total number of API requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency for prediction endpoint")

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_COUNT.inc()
    PREDICTION_LATENCY.observe(duration)
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
