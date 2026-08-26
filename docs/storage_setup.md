# Storage Setup

Do this before other steps.

## Create a filestore

Create a filestore. This is the first step. You can also use an existing filestore if one is already created — an existing filestore has already been created for this project.

```bash
gcloud filestore instances create argo-filestore \
    --zone=us-west1-c \
    --tier=BASIC_HDD \
    --file-share=name="argo_share",capacity=1TB \
    --network=name="pdg-network-1"
```

Get the filestore IP:

```bash
NEW_FILESTORE_IP=$(gcloud filestore instances describe argo-filestore \
    --zone=us-west1-c \
    --format="value(networks[0].ipAddresses[0])")

echo "New Filestore IP: $NEW_FILESTORE_IP"
```

!!! warning "Special note on network"
    If you create the filestore, the network matters. This IP needs to go in `filestore-pv.yaml` under `spec.nfs.server`.

Get the full details to confirm:

```bash
gcloud filestore instances describe argo-filestore \
    --zone=us-west1-c \
    --format="yaml(name, state, fileShares, networks)"
```

Set firewall rules for the filestore:

```bash
gcloud compute firewall-rules create allow-filestore \
    --network=pdg-network-1 \
    --allow=tcp:111,udp:111,tcp:2049,udp:2049 \
    --source-ranges=10.0.0.0/8
```

## Creating a cluster

This pipeline will exceed the specs for the Autopilot cluster, so after filestore is set up (or verified), create a standard cluster:

```bash
gcloud container clusters create water-cluster \
    --region=us-west1 \
    --network=pdg-network-1 \
    --subnetwork=pdg-subnet-us-west1 \
    --machine-type=n2-highmem-4 \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=2 \
    --max-nodes=10 \
    --enable-ip-alias
```

Output of `kubectx` should show:

```
gke_pdg-project-406720_us-west1_water-cluster
```

Whitelist the cluster (see [Getting Started](getting_started.md) for how to get `$MY_IP`):

```bash
gcloud container clusters update water-cluster --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $MY_IP/32
```

## Create the Argo namespace and install the Argo server

```bash
kubectl create namespace argo
kubectl get pods -n argo
```

If you see no pods, install Argo Workflows:

```bash
kubectl apply -n argo -f "https://github.com/argoproj/argo-workflows/releases/download/v3.6.5/quick-start-minimal.yaml"
```

To access the Argo Workflow UI, port-forward the Argo server to your local machine:

```bash
kubectl -n argo port-forward deployment/argo-server 2746:2746
```

## Set up PV and PVC

These files are under the `/storage_setup` directory. After setting up filestore and adding the IP to `filestore-pv.yaml`, create a PersistentVolume:

```bash
kubectl apply -f filestore-pv.yaml
```

Next, create the PersistentVolumeClaim:

```bash
kubectl -n argo apply -f filestore-pvc.yaml
```

## Tests

Test the network (check the IP in the yaml file first):

```bash
kubectl -n argo apply -f network-test-v2.yaml
```

Test mounting to the PVC:

```bash
kubectl -n argo apply -f final-test.yaml
```

Test connecting to filestore:

```bash
kubectl -n argo apply -f test-filestore-connect.yaml
```

Remove the test pods when done:

```bash
kubectl -n argo delete pod network-test-v2
kubectl -n argo delete pod final-nfs-test
kubectl -n argo delete pod nfs-test
```

## Connecting to the PVC

This script creates 3 pods. You can exec into any of them to view what is in the PVC under `/data`:

```bash
kubectl -n argo apply -f multi-pvc-inspectors.yaml

kubectl -n argo exec -it pvc-inspector-1 -- /bin/sh
kubectl -n argo exec -it pvc-inspector-2 -- /bin/sh
```

To remove:

```bash
kubectl -n argo delete -f multi-pvc-inspectors.yaml
```

This completes the storage setup. Continue to [Secrets Setup](secrets.md).
