# water_timeseries_argo_workflow
Argo workflows for running the water time series pipeline. 

# connecting to network

need to whitelist your IP

`gcloud container clusters update autopilot-cluster-2 --region us-west1 \
    --enable-master-authorized-networks \
    --master-authorized-networks $(curl -s ifconfig.me)/32`

# port forward to view the argo UI interface

`kubectl -n argo port-forward deployment/argo-server 2746:2746`