# Secrets Setup

The `lake_drainage_cron.yaml` file uses a number of secrets that you will need to set up in order for the workflow to run. If you have run the workflow locally, you will likely already have the necessary credentials or tokens on your local machine.

These secrets are:

- `earth-engine-creds` — Earth Engine credentials
- `gcp-personal-gcp-creds` — gcloud credentials
- `ghcr-secret` — GitHub Container Registry credentials

Each is explained below. For the Argo workflow, creation of the volume is handled within the yaml file.

## gcloud credentials setup

You will need to set up Google Cloud credentials for use in the code.

On your machine, run this command to create an application secrets file (follow the prompts) — see the [PDG gcloud setup documentation](https://github.com/PermafrostDiscoveryGateway/pdg-tech/blob/master/gcloud/gcloud-setup.md):

```bash
gcloud init
gcloud auth application-default login
```

This creates a file at `~/.config/gcloud/application_default_credentials.json`.

Add this as a secret to your Kubernetes cluster:

```bash
kubectl create secret generic personal-gcp-creds -n argo \
    --from-file=key.json=$HOME/.config/gcloud/application_default_credentials.json
```

Check that the secret was created:

```bash
kubectl get secret personal-gcp-creds -n argo -o jsonpath='{.data}' | jq 'keys'
```

Expected output:

```json
["key.json"]
```

## Credentials for Google Cloud Artifact Registry

!!! note "In progress"
    These instructions are in progress and will be updated soon. Currently the service account does not have permission, so you need to use your personal credentials to push images to the registry. You will need to regenerate them before any workflow run.

```bash
gcloud auth configure-docker us-west1-docker.pkg.dev
```

Delete any existing secret:

```bash
kubectl delete secret artifact-registry-pull-secret -n argo --ignore-not-found=true
```

Generate a new secret:

```bash
kubectl create secret docker-registry artifact-registry-pull-secret \
    --namespace argo \
    --docker-server=https://us-west1-docker.pkg.dev \
    --docker-username=oauth2accesstoken \
    --docker-password="$(gcloud auth application-default print-access-token)" \
    --dry-run=client -o yaml | kubectl apply -f -
```

!!! warning
    These secrets expire quickly, so you will need to regenerate them before any workflow run.

## GitHub Container Registry setup

First install `gh` on the command line and get your token. Make sure it works locally.

```bash
export GH_TOKEN=$(gh auth status --show-token | grep "Token:" | awk '{print $3}')
```

Create the secret in the Kubernetes namespace:

```bash
kubectl create secret docker-registry ghcr-secret \
  --namespace=argo \
  --docker-server=ghcr.io \
  --docker-username=<your-github-username> \
  --docker-password="${GH_TOKEN}" \
  --docker-email=<your-email>
```

Check it exists:

```bash
kubectl get secret ghcr-secret -n argo -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d | jq
```

## Earth Engine setup

```bash
ls ~/.config/earthengine/
```

```bash
kubectl create secret generic earth-engine-creds \
  --from-file=credentials=$HOME/.config/earthengine/credentials \
  -n argo
```

Expected output:

```
secret/earth-engine-creds created
```

With secrets in place, continue to [Argo Workflows (NRT Pipeline)](argo_workflows.md).
