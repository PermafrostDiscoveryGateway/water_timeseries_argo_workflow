# 8. Running the Full Pipeline

Once secrets, storage, and namespace/Argo setup are complete, and you've validated the DAG shape
with the [testing pipeline](07-running-the-testing-pipeline.md), you can run the full pipeline
against all production regions.

Make sure the [region lake polygons have been generated](07-running-the-testing-pipeline.md#generate-the-region-lake-polygons)
before applying the cron workflow below — the download jobs will fail without them.

## Files

- Template: [`argo_workflows/near_real_time/workflow_templates/nrt-pipeline-templates.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/near_real_time/workflow_templates/nrt-pipeline-templates.yaml)
- Pipeline cron workflow: [`argo_workflows/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron.yaml)
- Cloud sync cron workflow: [`argo_workflows/near_real_time/cron_jobs/upload/upload.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/near_real_time/cron_jobs/upload/upload.yaml)
- Debug cleanup cron workflow: [`argo_workflows/near_real_time/cron_jobs/pipeline/clean-up-debug-cron.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/near_real_time/cron_jobs/pipeline/clean-up-debug-cron.yaml)

## Apply the workflow template

```bash
kubectl -n argo apply -f argo_workflows/near_real_time/workflow_templates/nrt-pipeline-templates.yaml
```

## Apply the pipeline cron workflow

```bash
kubectl -n argo apply -f argo_workflows/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron.yaml
```

This `CronWorkflow` runs on a daily schedule (`suspend: false` by default) and fans out one
`region-pipeline` (download → merge/backfill → process) per production region:
`ALASKA`, `CANADA1-4`, `EURASIA1-3`, plus `TEST`. Once every region's pipeline finishes, it runs
`create-historical-zarr-archive`, then `create-historical-file`.

Cron jobs are created in their configured suspend state — check the Argo UI (**Cron Workflows**)
to confirm `nrt-pipeline-cron` is not suspended, or resume it manually if needed:

```bash
argo cron resume -n argo nrt-pipeline-cron
```

You can also manually trigger a run outside of its schedule from the Argo UI or CLI:

```bash
argo submit -n argo --from cronworkflow/nrt-pipeline-cron
```

## Apply the supporting cron jobs

These run independently of the main pipeline DAG and should also be applied so results get synced
to the cloud and debug output doesn't grow unbounded:

```bash
kubectl -n argo apply -f argo_workflows/near_real_time/cron_jobs/upload/upload.yaml
kubectl -n argo apply -f argo_workflows/near_real_time/cron_jobs/pipeline/clean-up-debug-cron.yaml
```

`upload.yaml` (the `cloud-sync-cron` workflow) runs every 4 hours and syncs the output directory
in the PVC with the Google Cloud Storage bucket.

## Where to find output

- Output by region and date:
  `gs://pdg-storage-default/water-timeseries-v2/data/output`
- Dynamic world input data:
  `gs://pdg-storage-default/water-timeseries-v2/near-real-time/dynamic_world_data`
- Combined NRT zarr archive:
  `gs://pdg-storage-default/water-timeseries-v2/near-real-time/output`

## Following jobs

Jobs can be followed in the Argo UI under **Cron Workflows** — see
[6. How the pipeline works](06-how-the-pipeline-works.md#following-jobs) for details on accessing
the UI and inspecting individual step logs.

## Notes on retries and secrets expiry

- Every job type checks whether a previous run for the same region/step is still in progress
  before starting a new one, so overlapping runs of the same stage are avoided.
- Some secrets (e.g. the Artifact Registry pull secret) expire quickly and must be regenerated
  before each workflow run — see [4. Setting up secrets](04-setting-up-secrets.md).
