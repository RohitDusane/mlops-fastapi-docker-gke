# syntax=docker/dockerfile:1.7

#########################################
# Stage 1 : Builder
#########################################

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# COPY requirements.txt requirements-dev.txt .

RUN pip install --upgrade pip setuptools wheel --no-cache-dir && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# RUN pip install --prefix=/install -r requirements.txt && \
#     pip install --prefix=/install -r requirements-dev.txt

RUN find /install -type f -name "*.pyc" -delete && \
    find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

#########################################
# Stage 2 : Runtime
#########################################

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG VERSION=1.0.0
ARG BUILD_DATE
ARG GIT_SHA

LABEL maintainer="rohit@yourdomain.com" \
      org.opencontainers.image.vendor="Rohit K Naik" \
      org.opencontainers.image.url="https://your-portfolio.com" \
      org.opencontainers.image.documentation="https://github.com/<your-username>/mlops-fastapi/blob/main/README.md" \
      org.opencontainers.image.title="Diabetes Risk Prediction API (UAE)" \
      org.opencontainers.image.description="Production FastAPI inference service" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/<username>/mlops-fastapi" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user — REQUIRED for EKS security policies
RUN groupadd -r fastapi && \
    useradd -r -g fastapi fastapi

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code only (NOT training code, NOT notebooks)
COPY --chown=fastapi:fastapi app ./app
COPY --chown=fastapi:fastapi configs ./configs
# COPY --chown=fastapi:fastapi artifacts ./artifacts

# Model artifacts are NOT baked into image
# They are pulled from S3 at container startup (see entrypoint)
COPY --chown=fastapi:fastapi scripts/fetch_model.sh ./scripts/
RUN chmod +x ./scripts/fetch_model.sh

# RUN chown -R fastapi:fastapi /app && \
#     chmod -R 755 /app
RUN mkdir -p artifacts logs && chown -R fastapi:fastapi artifacts logs

USER fastapi

EXPOSE 8000

# Kubernetes readiness probe calls /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
    # CMD wget --no-verbose --tries=1 --spider http://localhost:8000/docs || exit 1

# Gunicorn + UvicornWorker = production-grade async serving
CMD ["gunicorn", "app.main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "2", "--bind", "0.0.0.0:8000", "--graceful-timeout", "30", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]