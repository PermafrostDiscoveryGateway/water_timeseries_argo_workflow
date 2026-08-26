# water-timeseries-argo-workflow

Argo Workflows for running the [water-timeseries-v2](https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2) lake drainage pipeline on a Google Kubernetes Engine (GKE) cluster.

## Overview of the Lake Drainage Pipeline

This repository runs the lake drainage pipeline on the Google Cloud Kubernetes cluster using Argo Workflows. Various steps are handled by different Argo Workflow cron jobs.

## Inputs and outputs

For input, the pipeline starts with Dynamic World data — both a historical netCDF file, and new netCDF files downloaded per grid tile, region, and month. The pipeline merges this new data into an updated netCDF file.

Output is written as a `.zarr` dataset. Each region/date run gets its own Zarr dataset.

## Steps in the pipeline

To allow more parallel processing, the Argo Workflow is divided into multiple parts:

1. **Downloading** monthly data from Dynamic World (regions run in parallel). Before downloading, this checks whether the download previously ran, and if it finished. Download is marked complete once over 99% of expected grid tiles are downloaded, since a few grid tiles occasionally fail to download.
2. **Merging** newly downloaded data into a new netCDF file (one job runs). This checks the most recent netCDF file, then checks the ids and dates of the newly downloaded data to see if it has already been merged. If not, it merges that region.
3. **Running the NRT Breakpoint** for the latest date (regions run in parallel). Merged filenames follow a set pattern; this job looks for that file and confirms it is no longer being written to before processing each region.
4. **Syncing** the Google Cloud bucket with the new output results (a periodic cron job). This runs as its own cron job since running it at the end of processing sometimes failed.

!!! note
    Every job type is set up so that a new run will not start if a previous run of the same job is still in progress. For example, if data for Canada is still downloading when the next scheduled time arrives, the new job will not start. This avoids overlapping runs while also not waiting too long to retry a step that failed.

## Following jobs

Jobs can be followed using the Argo web interface. You will see the cron jobs under **Crons**. See [Getting Started](getting_started.md) for how to port-forward to the Argo UI.

## Running on Autopilot

Autopilot handles node scaling, which lowers costs. However, Autopilot comes with various constraints in terms of resources — see [Getting Started](getting_started.md) for creating a standard (non-Autopilot) cluster when the pipeline exceeds Autopilot's limits.

## Where is the data

Output is stored in Google Cloud Storage — see [Argo Workflows: Where to Find New Output](argo_workflows.md#where-to-find-new-output) for bucket paths.

## Next steps

- [Getting Started](getting_started.md) — cluster access, networking, and environment setup
- [Storage Setup](storage_setup.md) — Filestore, PersistentVolume/PersistentVolumeClaim setup
- [Secrets Setup](secrets.md) — credentials required for the workflow to run
- [Argo Workflows (NRT Pipeline)](argo_workflows.md) — how the near-real-time pipeline is structured and run
