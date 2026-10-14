# 5. Setting Up Filestore and Testing (Autopilot only)

> **This step only applies when running on an Autopilot cluster.** Autopilot does not support the
> same node-attached storage options as a standard GKE cluster, so shared storage for the
> pipeline (input/output data, artifacts) is provided via a Google Cloud Filestore instance
> mounted as an NFS-backed PersistentVolume/PersistentVolumeClaim.

The YAML files referenced below live in [`storage_setup/`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/storage_setup).

## Create (or confirm) the filestore instance

This is the first step, before creating the cluster. You can reuse an existing filestore instance
if one has already been created for this project.

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

**Important:** if you (re-)create the filestore, the network matters. This IP must be set in
[`storage_setup/filestore-pv.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/storage_setup/filestore-pv.yaml) under `spec.nfs.server`.

Confirm the full details:

```bash
gcloud filestore instances describe argo-filestore \
    --zone=us-west1-c \
    --format="yaml(name, state, fileShares, networks)"
```

## Set firewall rules for filestore

```bash
gcloud compute firewall-rules create allow-filestore \
    --network=pdg-network-1 \
    --allow=tcp:111,udp:111,tcp:2049,udp:2049 \
    --source-ranges=10.0.0.0/8
```

## Create the PV and PVC

After the filestore IP is set in `filestore-pv.yaml`, create the PersistentVolume:

```bash
kubectl apply -f storage_setup/filestore-pv.yaml
```

Then create the PersistentVolumeClaim:

```bash
kubectl -n argo apply -f storage_setup/filestore-pvc.yaml
```

## Testing the storage setup

Test the network connection (check the IP in the yaml file first):

```bash
kubectl -n argo apply -f storage_setup/network-test-v2.yaml
```

Test mounting to the PVC:

```bash
kubectl -n argo apply -f storage_setup/pvc-test.yaml
```

Test connecting to filestore directly:

```bash
kubectl -n argo apply -f storage_setup/test-filestore-connect.yaml
```

Remove the test pods when done:

```bash
kubectl -n argo delete pod network-test-v2
kubectl -n argo delete pod final-nfs-test
kubectl -n argo delete pod nfs-test
```

## Inspecting the contents of the PVC

To browse what's actually stored in the PVC (under `/data`), you can start inspector pods and
`exec` into them:

```bash
kubectl -n argo apply -f storage_setup/python-inspector.yaml
kubectl -n argo apply -f storage_setup/light_inspector.yaml
```

```bash
kubectl -n argo exec -it pvc-inspector-1 -- /bin/sh
```

Remove them when finished:

```bash
kubectl -n argo delete -f storage_setup/python-inspector.yaml
kubectl -n argo delete -f storage_setup/light_inspector.yaml
```

This completes the storage setup. Continue to [6. How the pipeline works](06-how-the-pipeline-works.md).
