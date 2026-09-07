FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

# Build the synthetic corpus + DB + reranker at image build time so the
# container starts ready to serve (no cold-start data generation).
RUN PYTHONPATH=/app python3 scripts/run_pipeline.py

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
