# 3.11 rather than the host's interpreter: ray publishes wheels in the range AutoGluon accepts
# only up to 3.12, and without ray AutoGluon fits every bagged fold sequentially.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=8

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src ./src
COPY scripts ./scripts

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "scripts/train_autogluon.py"]
