# setting up the storage

Before other steps, use this command 

# create filestore

Create a filestore. This is the first step. You can also use an existing filestore if it is already created.

`gcloud filestore instances create argo-filestore \
    --zone=us-west1-c \
    --tier=BASIC_HDD \
    --file-share=name="argo_share",capacity=1TB \
    --network=name="pdg-network-1"`

Get the filestore IP

`NEW_FILESTORE_IP=$(gcloud filestore instances describe argo-filestore \
    --zone=us-west1-c \
    --format="value(networks[0].ipAddresses[0])")`

`echo "New Filestore IP: $NEW_FILESTORE_IP"`

# SPECIAL NOTE ON NETWORK

if you create the filestore, the network matters

This IP would be in filestore-pv.yaml under spec, nfs, server

# Also get the full details to confirm

`gcloud filestore instances describe argo-filestore \
    --zone=us-west1-c \
    --format="yaml(name, state, fileShares, networks)"`

# firewall rules for filestore

`gcloud compute firewall-rules create allow-filestore \
    --network=pdg-network-1 \
    --allow=tcp:111,udp:111,tcp:2049,udp:2049 \
    --source-ranges=10.0.0.0/8`

# creating the cluster

This pipeline will exceed the specs for the autopilot cluster. So after filestore is set up (or verified)

cluster creation

`gcloud container clusters create water-cluster \
    --region=us-west1 \
    --network=pdg-network-1 \
    --subnetwork=pdg-subnet-us-west1 \
    --machine-type=n2-highmem-4 \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=2 \
    --max-nodes=10 \
    --enable-ip-alias`

output of `kubectx`

`gke_pdg-project-406720_us-west1_water-cluster
`

# whitelist cluster

`gcloud container clusters update water-cluster --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $MY_IP/32`

# create argo namespace

`kubectl create namespace argo`
kubectl get pods -n argo`

If you see no pods, run the following command to install argo workflow

`kubectl apply -n argo -f "https://github.com/argoproj/argo-workflows/releases/download/v3.6.5/quick-start-minimal.yaml"
`

In order to access the argo workflow UI, you need to port forward the argo server to your local machine using this command

`kubectl -n argo port-forward deployment/argo-server 2746:2746`



# PV and PVC

After setting up filestore and adding the IP to filestore-pv.yaml, create a PersistentVolume with this command

`kubectl apply -f filestore-pv.yaml`

Next, create the PersistentVolumeClaim using this

`kubectl -n argo apply -f filestore-pvc.yaml`

# Tests

Use this script to test the network (check the IP in the yaml file)

`kubectl -n argo apply -f network-test-v2.yaml`

Test mounting to the PVC

`kubectl -n argo final-test.yaml
`
Test connecting to filestore

`kubectl -n argo apply -f test-filestore-connect.yaml`

Remove tests

`kubectl -n argo delete pod network-test-v2
`
`kubectl -n argo delete pod final-nfs-test
`
`kubectl -n argo delete pod nfs-test`

# connecting to the PVC

this script creates 3 pods. You can exec into any of them to view what is in the PVC under /data

`kubectl -n argo apply -f multi-pvc-inspectors.yaml
`

`kubectl -n argo exec -it pvc-inspector-1  -- /bin/sh`


`kubectl -n argo exec -it pvc-inspector-2  -- /bin/sh`

to remove

`kubectl -n argo delete -f multi-pvc-inspectors.yaml`

This sets up the storage.