FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    netcat-openbsd \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY sentiment_analyzer/ ./sentiment_analyzer/
COPY main.py .
COPY .env.example .env
COPY docker-entrypoint.sh /usr/local/bin/

RUN mkdir -p /app/data /app/logs \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DATABASE_URL=sqlite:///./data/sentiment.db \
    REDIS_URL=redis://redis:6379/0 \
    WAIT_FOR_DB=false \
    WAIT_FOR_REDIS=false \
    RUN_MIGRATIONS=false \
    INIT_DATA=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

ENTRYPOINT ["docker-entrypoint.sh"]

CMD ["python", "-m", "streamlit", "run", "sentiment_analyzer/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
