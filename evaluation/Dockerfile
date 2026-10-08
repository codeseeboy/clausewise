FROM python:3.11-slim
WORKDIR /srv
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
COPY evaluation/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY evaluation ./evaluation
CMD ["python", "-m", "evaluation.run"]
