FROM python:3.11-slim
WORKDIR /srv
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
