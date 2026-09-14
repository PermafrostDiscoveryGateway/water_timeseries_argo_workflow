# 7. Running the Testing Pipeline

The test pipeline exercises the full DAG shape (download → merge/backfill → process → combine)
against a small, fast-running set of regions (`TEST` and `EURASIA3`), and writes to test-only
input/output paths, so you can validate changes before pointing anything at production data or
the full region list.

## Files

- Template: [`argo_workflows/near_real_time/workflow_templates/nrt-pipeline-templates-test.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/near_real_time/workflow_templates/nrt-pipeline-templates-test.yaml)
- Cron workflow: [`argo_workflows/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron-test.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron-test.yaml)

The test `CronWorkflow` is **suspended by default** (`suspend: true`) — it is meant to be run
manually while testing, not on its own schedule.

## Apply the templates

```bash
kubectl -n argo apply -f argo_workflows/near_real_time/workflow_templates/nrt-pipeline-templates-test.yaml
```

## Apply the cron workflow

```bash
kubectl -n argo apply -f argo_workflows/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron-test.yaml
```

Since it's created suspended, submit it manually to trigger a run — either from the Argo UI
(**Cron Workflows** → `nrt-pipeline-cron-test` → **Submit**), or from the CLI if you have the
`argo` CLI installed:

```bash
argo cron resume -n argo nrt-pipeline-cron-test
argo submit -n argo --from cronworkflow/nrt-pipeline-cron-test
```

## What the test DAG runs

1. `download` — one `download-region-with-fallback` task per test region, in parallel.
2. `merge-and-backfill` — merges all test regions, checks completeness, and retries
   backfill+re-merge for any region still below the completeness threshold, up to
   `max-backfill-attempts` (5, in the test config).
3. `process` — one `process-region` task per test region, in parallel, once merging is done.
4. `create-historical-zarr-archive` — combines all test regions' output into one zarr archive.
5. `create-historical-file` — merges the new test data into a new historical netCDF file.

You can follow progress and inspect logs for each step in the Argo UI (see
[6. How the pipeline works](06-how-the-pipeline-works.md#following-jobs)).

## Prerequisites

Before running the test pipeline, make sure you've completed:

- [1. Create a namespace](01-create-namespace.md)
- [2. Install Argo](02-install-argo.md)
- [4. Setting up secrets](04-setting-up-secrets.md)
- [5. Setting up filestore and testing](05-filestore-setup-and-testing.md) (Autopilot only)

Continue to [8. Running the full pipeline](08-running-the-full-pipeline.md).
