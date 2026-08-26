# Argo Workflows (NRT Pipeline)

This page explains the near-real-time (NRT) pipeline.

## Running the Near Real Time pipeline

Running the near real time pipeline involves the following steps:

1. During summer months, download data for the last complete month for each region.
2. Merge the partial download files for each region into a merged netCDF file.
3. Once the merged file for each region and date is finished, process using the NRT Breakpoint.
4. Periodically sync the Google Cloud bucket with the results in the output directory.
5. Create a new historical file that includes the new data.

## Notes on the order

Downloading data is a time-consuming step. For this reason, regions download in parallel, and a merged netCDF file for the region and date is created as soon as that region finishes downloading. Once that file exists, we can process and run the NRT breakpoint for that region and the latest month — this lets the NRT breakpoint run on a region as soon as it is downloaded and merged, even if other regions are still downloading.

## Running using the NRT pipeline templates

To avoid needing to create and monitor many separate cron jobs, we have created the following files under `argo_workflows`:

- `/workflow_templates/nrt-pipeline-templates.yaml`
- `/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron.yaml`

`nrt-pipeline-templates.yaml` defines all steps needed, provides a DAG for dependencies, and includes logic for retrying missing steps. To run:

```bash
kubectl -n argo apply -f nrt-pipeline-templates.yaml
kubectl -n argo apply -f nrt-pipeline-cron.yaml
```

The cron jobs are created paused — you will need to resume or submit them.

## Notes on the regions

Canada is a large region with many lakes. For this reason, like Eurasia, we split Canada into Canada1, 2, 3, and 4. Running both downloading and processing for Canada in parallel with smaller regions takes much less time.

### Notes on the order

1. Download Dynamic World data (in parallel).
2. Merge each region's new Dynamic World data into a single date-stamped file.
3. Process using the NRT Breakpoint for each region.
4. Combine the results into a single Zarr archive.
5. Combine the results and all historical ones together into one Zarr archive.
6. Combine the historical Dynamic World data and the newly downloaded data into a new netCDF file.
7. Upload both the input (Dynamic World data) and the output, both by region and by file.

## Where to find new output

The new output is currently located in this Google Cloud bucket:

| Contents | Path |
| --- | --- |
| Output by region and date | `gs://pdg-storage-default/water-timeseries-v2/data/output` |
| Dynamic World data | `gs://pdg-storage-default/water-timeseries-v2/near-real-time/dynamic_world_data` |
| Zarr archive (complete NRT results) | `gs://pdg-storage-default/water-timeseries-v2/near-real-time/output` |

## Deprecated: individual YAML files

The steps below describe the previous method of applying individual yaml files. These may be removed in the future, as the pipeline templates above supersede them, but are documented here for reference.

### Sync bucket with the latest output

```bash
kubectl -n argo apply -f upload/upload.yaml
```

Make sure this is running first — it checks the output directory at regular intervals and syncs the bucket with new contents.

### Downloading

```bash
kubectl -n argo apply -f download/download_alaska.yaml
kubectl -n argo apply -f download/download_test.yaml
kubectl -n argo apply -f download/download_canada1.yaml
kubectl -n argo apply -f download/download_canada2.yaml
kubectl -n argo apply -f download/download_canada3.yaml
kubectl -n argo apply -f download/download_canada4.yaml
kubectl -n argo apply -f download/download_eurasia1.yaml
kubectl -n argo apply -f download/download_eurasia2.yaml
kubectl -n argo apply -f download/download_eurasia3.yaml
```

These continue to run and retry on downloading failures for any region. Some zones get stuck showing 99% completion, so the rule is: if after a retry a zone is still 99% complete, it is considered finished.

### Merging the partial results for a region into a single netCDF file

```bash
kubectl -n argo apply -f merge/merge_new_region_results.yaml
```

This merges once downloading is finished.

### Running the NRT pipeline

```bash
kubectl -n argo apply -f process/process_alaska.yaml
kubectl -n argo apply -f process/process_test.yaml
kubectl -n argo apply -f process/process_canada1.yaml
kubectl -n argo apply -f process/process_canada2.yaml
kubectl -n argo apply -f process/process_canada3.yaml
kubectl -n argo apply -f process/process_canada4.yaml
kubectl -n argo apply -f process/process_eurasia1.yaml
kubectl -n argo apply -f process/process_eurasia2.yaml
kubectl -n argo apply -f process/process_eurasia3.yaml
```

These jobs check for the new merged netCDF file containing all new data for a region and the latest summer month. If that file exists and is verified to contain all downloaded data, it processes (if not already finished). This also restarts and retries on failure, saving intermediate progress.

The output is the Zarr dataset for the region/month.

### Create new historical file

```bash
kubectl -n argo apply -f merge/create_new_historical_file.yaml
```

This combines the existing historical dynamic world file with all newly downloaded data, once downloading and merging are done. It is meant to run after processing, so scripts do not end up using different input files.
