#!/bin/bash
# backup-manifests.sh — snapshot the live cluster state as a disaster-recovery
# reference. NOTE: `kubectl get -o yaml` on a *live* object includes runtime
# fields (resourceVersion, uid, status, creationTimestamp) that will cause
# `kubectl apply` to fail or behave oddly if you try to reapply this file
# as-is later — this is a reference snapshot, not a substitute for the
# actual source-controlled k8s/*.yaml files in your repo. If you ever need
# to restore from one of these, strip those fields first, or better: just
# use the real files already in git (k8s/deployment.yaml etc.) — they're
# the actual source of truth, this backup is a secondary safety net.
set -euo pipefail

BACKUP_DIR="$HOME/k3s-backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

kubectl get deployment diabetes-api -o yaml > "$BACKUP_DIR/diabetes-api-deployment.yaml"
kubectl get svc diabetes-api-service -o yaml > "$BACKUP_DIR/diabetes-api-service.yaml"
kubectl get hpa -o yaml > "$BACKUP_DIR/hpa.yaml"
kubectl get all -n mlflow -o yaml > "$BACKUP_DIR/mlflow-all.yaml"
kubectl get pvc -n mlflow -o yaml > "$BACKUP_DIR/mlflow-pvc.yaml"

tar -czf "$HOME/k3s-backup-$(date +%Y%m%d-%H%M%S).tar.gz" -C "$BACKUP_DIR" .
echo "Backup saved: $HOME/k3s-backup-*.tar.gz"
echo "Download it with (from your laptop, via SSM, not SCP — no SSH port open):"
echo "  aws ssm start-session --target <instance-id> --document-name AWS-StartPortForwardingSession ..."
echo "  or simpler: aws s3 cp it to a bucket you control, then pull from there."