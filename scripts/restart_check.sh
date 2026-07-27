#!/bin/bash
# restart_check.sh — run after any EC2 reboot, or anytime to sanity-check state.
# Assumes bootstrap.sh already ran — kubeconfig is already permanent, no export needed.

echo "====================== K3S SERVICE STATUS ======================"
systemctl status k3s --no-pager | head -5
systemctl is-enabled k3s docker amazon-ssm-agent refresh-ecr-secret.timer

echo ""
echo "====================== NODES ======================"
kubectl get nodes

echo ""
echo "====================== PODS (all namespaces) ======================"
kubectl get pods -A

echo ""
echo "====================== SERVICES ======================"
kubectl get svc

echo ""
echo "====================== HPA ======================"
kubectl get hpa

echo ""
echo "====================== HEALTH CHECK ======================"
# Your actual service is type=LoadBalancer on port 80 (k3s ServiceLB), not NodePort.
curl -sf http://localhost/health && echo "" || echo "❌ /health did not respond"

echo ""
echo "====================== MLFLOW REGISTRY ======================"
kubectl exec -n mlflow -it deploy/mlflow -- python -c \
  "from mlflow import MlflowClient; c=MlflowClient(tracking_uri='http://localhost:5000'); [print(m.name, m.aliases) for m in c.search_registered_models()]" \
  2>/dev/null || echo "⚠️  MLflow not reachable — check pod status above"