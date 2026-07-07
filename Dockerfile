# syntax=docker/dockerfile:1.7

#########################################
# Stage 1 : Builder
#########################################

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Required only while installing packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a separate location
RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    -r requirements.txt


#########################################
# Stage 2 : Runtime
#########################################

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user
RUN groupadd -r fastapi && \
    useradd -r -g fastapi fastapi

COPY --from=builder /install /usr/local

COPY app ./app
COPY artifacts ./artifacts

RUN chown -R fastapi:fastapi /app

USER fastapi

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]










# # Dockerfile
# FROM python:3.10
# WORKDIR /app
# COPY . /app
# RUN pip install -r requirements.txt
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
