# Getting Started

## Note on the environment

This assumes you have already cloned and installed [water-timeseries-v2](https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2).

Use that environment, or create a new one and install that repo using:

```bash
uv add git+https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2
```

Also install the dependencies in `requirements.txt`.

## Adding the Autopilot cluster

You will need to add the Autopilot cluster. First, authenticate using `gcloud`:

```bash
gcloud auth application-default login
```

Add the cluster:

```bash
gcloud container clusters get-credentials autopilot-2 \
    --region=us-west1 \
    --project=pdg-project-406720
```

Confirm you are using that context:

```bash
kubectx
```

## Connecting to the network

`kubectl` commands will likely hang. On any network, you will need to run this command to whitelist your IP for access:

```bash
gcloud container clusters update autopilot-cluster-2 --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $(curl -s ifconfig.me)/32
```

or, on some networks:

```bash
gcloud container clusters update autopilot-cluster-2 --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $(curl -4 -s ifconfig.me)/32
```

## Port-forwarding to the Argo UI

```bash
kubectl -n argo port-forward deployment/argo-server 2746:2746
```

## The download VM

A VM was created to handle some tasks. Useful commands:

```bash
gcloud compute instances stop download-vm --zone=us-west1-a
gcloud compute instances start download-vm --zone=us-west1-a
gcloud compute ssh download-vm --zone=us-west1-a
```

## Next steps

Once you can reach the cluster, continue to [Storage Setup](storage_setup.md).
