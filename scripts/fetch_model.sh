#!/bin/bash
# scripts/fetch_model.sh
# Called by K8s initContainer to pull model from S3 before API starts

set -e

MODEL_S3_PATH=${MODEL_S3_PATH:-"s3://your-mlops-bucket-uae/models/latest/model.pkl"}
LOCAL_PATH=${MODEL_PATH:-"/app/artifacts/model.pkl"}

echo "Fetching model from S3: ${MODEL_S3_PATH}"
mkdir -p "$(dirname ${LOCAL_PATH})"
aws s3 cp "${MODEL_S3_PATH}" "${LOCAL_PATH}"
echo "Model ready at ${LOCAL_PATH}"