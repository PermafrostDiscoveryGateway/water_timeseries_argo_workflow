# 1. Create a Namespace in the Cluster

All resources for this pipeline (Argo server, workflows, cron jobs, secrets, volumes) live in the
`argo` namespace.

## Connect to the cluster

If you haven't already, authenticate with gcloud and fetch cluster credentials:

```bash
gcloud auth application-default login
```

**(Autopilot only)** add the autopilot cluster's credentials:

```bash
gcloud container clusters get-credentials autopilot-2 \
    --region=us-west1 \
    --project=pdg-project-406720
```

If you are using a standard (non-Autopilot) cluster instead, get credentials for that cluster
name instead, e.g.:

```bash
gcloud container clusters get-credentials water-cluster --region=us-west1
```

Confirm you're pointed at the right cluster context:

```bash
kubectx
```

## Create the namespace

```bash
kubectl create namespace argo
```

Verify it exists (it will have no pods yet, until Argo is installed):

```bash
kubectl get pods -n argo
```

Continue to [2. Install Argo](02-install-argo.md).
