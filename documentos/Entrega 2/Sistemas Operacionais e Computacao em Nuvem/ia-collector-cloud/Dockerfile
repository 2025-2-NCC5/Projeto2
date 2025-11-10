# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    APP_DIR=/app \
    DB_PATH=/app/data/metrics.db \
    INTERVAL_SEC=5 \
    WINDOW_MINUTES=60 \
    REPORT_DIR=/app/report \
    DEMO_SEED_DATA=1 \
    MPLBACKEND=Agg

WORKDIR ${APP_DIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libfreetype6-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    flask==3.0.3 \
    gunicorn==21.2.0 \
    psutil==6.0.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    matplotlib==3.8.4

COPY app.py ${APP_DIR}/app.py
RUN mkdir -p ${APP_DIR}/data ${APP_DIR}/report

EXPOSE ${PORT}
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "app:app"]
