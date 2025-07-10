# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy API code
COPY ./api /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn prometheus-client joblib scikit-learn pandas

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

