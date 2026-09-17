# 3. Whitelisting (Autopilot only) and Port Forwarding

## Whitelist your IP (Autopilot only)

GKE Autopilot clusters used here have master-authorized-networks enabled, so `kubectl` commands
will hang until your current IP is whitelisted. On most networks:

```bash
gcloud container clusters update autopilot-cluster-2 --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $(curl -s ifconfig.me)/32
```

On some networks, force IPv4 resolution instead:

```bash
gcloud container clusters update autopilot-cluster-2 --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $(curl -4 -s ifconfig.me)/32
```

If you're on a standard (non-Autopilot) cluster with authorized networks enabled, the same
pattern applies, substituting your cluster's name, e.g.:

```bash
gcloud container clusters update water-cluster --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $MY_IP/32
```

If your cluster does not restrict access by authorized networks, you can skip whitelisting
entirely.

## Port forward to view the Argo UI

To access the Argo Workflows UI from your local machine:

```bash
kubectl -n argo port-forward deployment/argo-server 2746:2746
```

Then open [https://localhost:2746](https://localhost:2746) in your browser.

Continue to [4. Setting up secrets](04-setting-up-secrets.md).
