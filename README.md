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

# NOTE ON THE ENVIRONMENT

This assumes that you have already cloned and installed 

https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2

Use that environment, or create a new one and install that repo using

uv add git+https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2

also install dependencies in requirements.txt