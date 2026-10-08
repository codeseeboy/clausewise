# Build context is the repository root (see docker-compose.yml).
FROM python:3.11-slim AS app
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10

# CPU build of torch keeps the image small; the embedding and reranker models are small.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Same image plus pytest and the test files.
FROM app AS test
RUN pip install --no-cache-dir pytest==9.1.1
COPY pyproject.toml .
COPY tests ./tests
COPY evaluation ./evaluation
COPY data ./data
CMD ["pytest", "-q", "--junitxml=results/test_report.xml"]
