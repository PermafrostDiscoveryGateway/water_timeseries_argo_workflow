# 6. How the Pipeline Works

This pipeline (the "NRT", or near-real-time, pipeline) processes dynamic world data for lake
drainage detection, region by region, and combines the results into `.zarr` archives.

## Inputs and outputs

- **Input:** dynamic world data, both a historical netCDF file and newly downloaded netCDF tiles
  per grid tile / region / month.
- **Output:** a `.zarr` dataset, one per region/date run.

## Steps

To allow for more parallel processing, the pipeline is split into multiple stages, run as an Argo
DAG per region, followed by combination stages once every region is done:

1. **Download** dynamic world data for the last complete month, per region, in parallel.
   Before downloading, each region checks whether it already ran and finished; a download is
   considered complete once over 99% of expected grid tiles are downloaded (a small number of
   tiles occasionally fail to download and are treated as an acceptable loss).
2. **Merge** the newly downloaded per-tile files for a region into a single, date-stamped netCDF
   file. This checks the most recent merged netCDF file, compares it against the IDs/dates of the
   newly downloaded data, and only merges if there's new data not yet merged.
3. **Process** using the NRT Breakpoint, once the merged file for a region/date is confirmed
   finished and no longer being written to. Regions run in parallel — a region that finishes
   downloading and merging early does not wait on slower regions.
4. **Combine** each region's process output into a single zarr archive for that run.
5. **Combine** the current run's zarr archive with all historical archives into one combined
   historical zarr archive.
6. **Create a new historical file** that merges the existing historical dynamic world netCDF data
   with the newly downloaded data, so that the next run's "historical" input is up to date. This
   runs after processing, to avoid different steps reading different generations of the input
   file.
7. **Sync** the Google Cloud Storage bucket with the output directory. This runs as its own cron
   job on a periodic schedule (independent of the main pipeline DAG), since combining this into
   the end of the process step proved unreliable.

Every job type is written so that a new run will not start while a previous run for the same
region/step is still in progress — e.g. if Canada's download is still running when the next
scheduled time arrives, the new one is skipped rather than started concurrently. This avoids long
waits for retries while also avoiding duplicate/overlapping work.

## Regions

Large regions are split into smaller sub-regions so that download and processing time stays
reasonable when run in parallel:

- Canada → `CANADA1`, `CANADA2`, `CANADA3`, `CANADA4`
- Eurasia → `EURASIA1`, `EURASIA2`, `EURASIA3`

## Orchestration: templates vs. cron jobs

Rather than creating and monitoring many separate cron jobs by hand, the DAG logic (dependencies,
retries, per-stage resource requests) is defined once as a reusable
[`WorkflowTemplate`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/near_real_time/workflow_templates/nrt-pipeline-templates.yaml),
and a lightweight [`CronWorkflow`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/argo_workflows/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron.yaml)
references it via `templateRef` and supplies the per-region parameters (chunk size, save interval,
`n_jobs`, etc).

There is a parallel **test** version of both files — `nrt-pipeline-templates-test.yaml` and
`nrt-pipeline-cron-test.yaml` — that runs against a restricted set of regions and test
input/output paths. See [7. Running the testing pipeline](07-running-the-testing-pipeline.md).

## Following jobs

Jobs can be followed in the Argo web UI (see
[3. Whitelisting and port forwarding](03-whitelist-and-port-forward.md) to access it) under the
**Cron Workflows** tab. From there, you can inspect the DAG for the running/most recent workflow,
view logs per step, and manually trigger ("submit") a cron workflow outside of its schedule.

## Autopilot considerations

Autopilot handles node scaling automatically, which can lower costs, but it also comes with
constraints on a per-pod basis (e.g. resource requests/limits must be specified, and some
storage/networking options are unavailable — see
[5. Setting up filestore and testing](05-filestore-setup-and-testing.md)).

Continue to [7. Running the testing pipeline](07-running-the-testing-pipeline.md).
