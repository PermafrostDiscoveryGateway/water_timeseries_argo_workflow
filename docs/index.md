# Water Time Series Argo Workflow

Documentation for setting up and running the water time series (lake drainage / dynamic world)
pipeline as Argo Workflows on a Google Kubernetes Engine (GKE) cluster.

This pipeline downloads dynamic world data, merges it, runs the NRT ("near real time")
breakpoint processing step, and combines results into `.zarr` archives, syncing results to a
Google Cloud Storage bucket along the way.

> **Note on Autopilot:** this project has been run on both a standard GKE cluster and a GKE
> **Autopilot** cluster. Autopilot handles node scaling automatically and can lower costs, but it
> also comes with additional constraints (whitelisting, no self-managed filestore networking,
> resource limits per pod, etc). Steps that only apply when you are using an Autopilot cluster
> are clearly labeled **(Autopilot only)** throughout these docs. If you are running on a
> standard (non-Autopilot) cluster, you can skip those sections.

## Contents

1. [Create a namespace in the cluster](01-create-namespace.md)
2. [Install Argo](02-install-argo.md)
3. [Whitelisting (Autopilot only) and port forwarding](03-whitelist-and-port-forward.md)
4. [Setting up secrets](04-setting-up-secrets.md)
5. [Setting up filestore and testing (Autopilot only)](05-filestore-setup-and-testing.md)
6. [How the pipeline works](06-how-the-pipeline-works.md)
7. [Running the testing pipeline](07-running-the-testing-pipeline.md)
8. [Running the full pipeline](08-running-the-full-pipeline.md)

## Source READMEs

These docs consolidate information from the following files in the repository, which remain the
canonical, most detailed reference for their respective areas:

- [`README.md`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/README.md) — cluster/storage/secrets setup
- [`argo_workflows/README.md`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/README.md) — explanation of the NRT pipeline steps
- [`storage_setup/README.md`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/storage_setup/README.md) — filestore/PV/PVC setup
- [`Lake_Drainage_Overview.md`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/Lake_Drainage_Overview.md) — high level pipeline overview
