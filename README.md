# water_timeseries_argo_workflow
Argo workflows for running the water time series pipeline. 

# connecting to network

need to whitelist your IP

`gcloud container clusters update autopilot-cluster-2 --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $(curl -s ifconfig.me)/32`

# port forward to view the argo UI interface

`kubectl -n argo port-forward deployment/argo-server 2746:2746`

# a VM was created for to handle some tasks. Commands below

gcloud compute instances stop download-vm --zone=us-west1-a
gcloud compute instances start download-vm --zone=us-west1-a
gcloud compute ssh download-vm --zone=us-west1-a

# NOTE ON THE ENVIRONMENT

This assumes that you have already cloned and installed 

https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2

Use that environment, or create a new one and install that repo using

uv add git+https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2

also install dependencies in requirements.txt
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


### Important note - setting up secrets

The .yaml file (lake_drainage_cron.yaml) uses a number of secrets that you will need to set up 
in order for the workflow to run. If you have run the workflow locally, then you will likely already have 
the necessary credentials or tokens on your local machine.

These secrets are 

earth-engine-creds (earth engine credentials)

gcp-personal-gcp-creds (gcloud credentials)

ghcr-secret (github container registry credentials)

Each of these will be explained in a section below. 

For the argo workflow, the creation of the volume is handled within the yaml file. 

### Gcloud Credentials Setup

You will need to setup google cloud credentials for use in the code the following way.

On your machine, run this command to create a application secrets file.

`glcoud init` (follow the prompts) PDG Documention https://github.com/PermafrostDiscoveryGateway/pdg-tech/blob/master/gcloud/gcloud-setup.md

`gcloud auth application-default login`

This will create a file at `~/.config/gcloud/application_default_credentials.json`

Now, you will want to add this secret to your kubernetes cluster using this command:

`kubectl create secret generic personal-gcp-creds -n argo --from-file=key.json=$HOME/.config/gcloud/application_default_credentials.json
`

Check that the secret was created using this command:

` kubectl get secret personal-gcp-creds -n argo -o jsonpath='{.data}' | jq 'keys'
`

expect to see the following output:
`[
  "key.json"
]`


## Setup Credentials for Google Cloud Artifact Registry (in progress)

These instructions are 'in progress' and will be updated soon. Currently the service account does not have permission,
so you will need to use your personal credentials to push images to the registry. You will need to regenerate them before any workflow run.

Run this command first

`gcloud auth configure-docker us-west1-docker.pkg.dev`

Now, delete any existing secret 

`kubectl delete secret artifact-registry-pull-secret -n argo --ignore-not-found=true
`

Generate a new secret with this command:

`kubectl create secret docker-registry artifact-registry-pull-secret \
    --namespace argo \
    --docker-server=https://us-west1-docker.pkg.dev \
    --docker-username=oauth2accesstoken \
    --docker-password="$(gcloud auth application-default print-access-token)" \
    --dry-run=client -o yaml | kubectl apply -f -
`

NOTE - these secrets will expire quickly, so you will need to regenerate them before any workflow run.

## Github container registry setup

First install gh on the command line. This gets your token. Make sure it works locally.

`export GH_TOKEN=$(gh auth status --show-token | grep "Token:" | awk '{print $3}')
`

This creates the secret on the kubernetes namespace

`kubectl create secret docker-registry ghcr-secret \
  --namespace=argo \
  --docker-server=ghcr.io \
  --docker-username=tcnichol \
  --docker-password="${GH_TOKEN}" \
  --docker-email=tcnichol@illinois.edu`

Checks it exists

`kubectl get secret ghcr-secret -n argo -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d | jq
`

## earthengine setup


`ls ~/.config/earthengine/`


`kubectl create secret generic earth-engine-creds \
  --from-file=credentials=$HOME/.config/earthengine/credentials \
  -n argo`

`secret/earth-engine-creds created
