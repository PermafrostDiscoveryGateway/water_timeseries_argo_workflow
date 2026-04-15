#!/bin/bash
# update-toml-config.sh

NAMESPACE="argo"
CONFIG_FILE="config.toml"

# Create/update ConfigMap from TOML file
kubectl create configmap download-dw-config -n $NAMESPACE \
  --from-file=config.toml=$CONFIG_FILE \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify it was created
echo "ConfigMap contents:"
kubectl get configmap download-dw-config -n $NAMESPACE -o jsonpath='{.data.config\.toml}' | cat