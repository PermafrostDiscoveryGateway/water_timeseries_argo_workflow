# setting up the storage

Before other steps, use this command 

# create filestore

Create a filestore. This is the first step. You can also use an existing filestore.

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

Once these steps are done, you are ready to run the 