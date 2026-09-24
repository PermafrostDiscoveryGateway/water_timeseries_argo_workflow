# 4. Setting Up Secrets

The workflow YAML files (e.g. `nrt-pipeline-cron.yaml`) reference a number of Kubernetes secrets
that must exist in the `argo` namespace before any workflow will run successfully. If you've
already run the pipeline locally, you likely already have the underlying credentials on your
machine, and just need to load them into the cluster as secrets.

The required secrets are:

- `personal-gcp-creds` — gcloud application default credentials
- `earth-engine-creds` — Google Earth Engine credentials
- `ghcr-secret` — GitHub Container Registry pull credentials
- `artifact-registry-pull-secret` — Google Cloud Artifact Registry pull credentials (in progress, see below)

## Gcloud credentials

On your machine, create an application-default credentials file:

```bash
gcloud init
```

Follow the prompts (see also the [PDG gcloud setup docs](https://github.com/PermafrostDiscoveryGateway/pdg-tech/blob/master/gcloud/gcloud-setup.md)).

```bash
gcloud auth application-default login
```

This creates a file at `~/.config/gcloud/application_default_credentials.json`.

Add it to the cluster as a secret:

```bash
kubectl create secret generic personal-gcp-creds -n argo \
    --from-file=key.json=$HOME/.config/gcloud/application_default_credentials.json
```

Check that it was created:

```bash
kubectl get secret personal-gcp-creds -n argo -o jsonpath='{.data}' | jq 'keys'
```

Expected output:

```
[
  "key.json"
]
```

## Google Cloud Artifact Registry credentials (in progress)

These instructions are in progress and will be updated. Currently the service account does not
have permission to pull images, so personal credentials are used instead. **These credentials
expire quickly and must be regenerated before any workflow run.**

```bash
gcloud auth configure-docker us-west1-docker.pkg.dev
```

Delete any existing secret:

```bash
kubectl delete secret artifact-registry-pull-secret -n argo --ignore-not-found=true
```

Generate a new one:

```bash
kubectl create secret docker-registry artifact-registry-pull-secret \
    --namespace argo \
    --docker-server=https://us-west1-docker.pkg.dev \
    --docker-username=oauth2accesstoken \
    --docker-password="$(gcloud auth application-default print-access-token)" \
    --dry-run=client -o yaml | kubectl apply -f -
```

## GitHub Container Registry (ghcr) setup

Install `gh` on the command line and confirm it works locally, then grab a token:

```bash
export GH_TOKEN=$(gh auth status --show-token | grep "Token:" | awk '{print $3}')
```

Create the secret in the `argo` namespace:

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

Confirm you have local Earth Engine credentials:

```bash
ls ~/.config/earthengine/
```

Create the secret:

```bash
kubectl create secret generic earth-engine-creds \
    --from-file=credentials=$HOME/.config/earthengine/credentials \
    -n argo
```

Expected output:

```
secret/earth-engine-creds created
```

Continue to [5. Setting up filestore and testing (Autopilot only)](05-filestore-setup-and-testing.md).
