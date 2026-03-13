FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application code
COPY envs/ envs/
COPY agents/ agents/
COPY utils/ utils/
COPY app.py .
COPY dashboard.py .

# Copy data and models
COPY data/ data/
COPY results/ results/

EXPOSE 8000 8501

# Default: run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]