# 9. Data access

How to get data into and out of the pipeline's shared volume on the software-dev Kubernetes
cluster, and where copies of the results end up.

## Where the data lives

Every Argo pod mounts the `argo-workflows-share` PersistentVolumeClaim (namespace `argo`,
storage class `nfs-taiga`, see `storage_setup/ncsa-setup.yaml`) at `/data`. The pipeline's files
are under `/data/water_timeseries`:

| Path in the pods | What it holds |
| --- | --- |
| `/data/water_timeseries/input` | The global lake vector file (`Nitze_etal_Lakes_filtered_full_set_V2d.parquet`) |
| `/data/water_timeseries/region_lake_polygons` | Per-region lake polygons generated from it |
| `/data/water_timeseries/dynamic_world_data` | Historical `lakes_dw_*.nc` files, plus `downloads/` and merge scratch |
| `/data/water_timeseries/output` | Per-region results (breakpoint zarr, `drain_*.parquet`) |
| `/data/water_timeseries/combined_zarr_datasets` | Combined NRT zarr archives |
| `/data/water_timeseries/test/...` | The same layout for the test pipeline |

The volume is backed by NCSA's Taiga file system, so the same files are also visible from NCSA
systems that mount Taiga (see [From Delta and other NCSA systems](#from-delta-and-other-ncsa-systems)).

Pick a method by what you need:

| You want to... | Use |
| --- | --- |
| Look around, check sizes, open a NetCDF file | [Inspector pod](#browsing-the-volume-with-a-pvc-inspector-pod) |
| Copy a few files or a small directory to your machine | [`kubectl cp`](#downloading-with-kubectl-cp) |
| Copy a large directory to your machine | [rsync server](#downloading-large-directories-with-rsync) |
| Put files onto the volume from your machine | [WebDAV upload server](#uploading-with-the-webdav-server) |
| Read the data on Delta | [Taiga path](#from-delta-and-other-ncsa-systems) |
| Get published results without cluster access | [Google Cloud Storage](#google-cloud-storage) |

All `kubectl` commands need cluster access; if they hang, whitelist your IP first
(see [3. Whitelist & port forward](03-whitelist-and-port-forward.md)).

## Browsing the volume with a pvc-inspector pod

The pvc-inspector pods let you browse the contents of the PVC (mounted at `/data`) without
having to spin up a full pipeline run. Manifests live in `storage_setup/`:

- `storage_setup/python-inspector.yaml` -> pod `pvc-inspector-python` (python:3.12-slim + netcdf4)
- `storage_setup/light_inspector.yaml` -> pod `pvc-inspector-light` (alpine + ncdump/tree/file)

Check if one is already running:

```bash
kubectl -n argo get pods | grep pvc-inspector
```

If nothing is there, start one (pick whichever image suits what you need to do):

```bash
kubectl -n argo apply -f storage_setup/python-inspector.yaml
kubectl -n argo apply -f storage_setup/light_inspector.yaml
```

Wait for it to be Running:

```bash
kubectl -n argo get pod pvc-inspector-python -w
```

### Exec into the pod

```bash
kubectl -n argo exec -it pvc-inspector-python -- sh
kubectl -n argo exec -it pvc-inspector-light -- sh
```

Once inside, the PVC is mounted at `/data`, e.g.:

```bash
ls -la /data/water_timeseries/dynamic_world_data/
find /data -name '*.nc' | head -20
du -sh /data/water_timeseries/*
```

### Cleaning up

Remove the inspector pod(s) when you're done with them:

```bash
kubectl -n argo delete -f storage_setup/python-inspector.yaml
kubectl -n argo delete -f storage_setup/light_inspector.yaml
```

## Downloading with kubectl cp

Once you have an inspector pod up (see above), you can pull data off the PVC with `kubectl cp`.
This is the quickest way to get what you need onto your local machine to run the pipeline outside
the cluster.

Download the dynamic world data:

```bash
kubectl -n argo cp pvc-inspector-python:/data/water_timeseries/dynamic_world_data ./data/dynamic_world
```

Download the contents of output:

```bash
kubectl -n argo cp pvc-inspector-python:/data/water_timeseries/output ./output
```

Two more things you'll typically need alongside those:

Download the input vector lake file (the lake polygons parquet that everything else is keyed off
of):

```bash
kubectl -n argo cp pvc-inspector-python:/data/water_timeseries/input ./data/input
```

Download the per-region lake polygon files (used via `region_lake_polygons_dir` - these are
generated from the vector file above by `utils/generate_region_lake_polygons.py`, but it's much
faster to just copy the already-generated ones off the PVC):

```bash
kubectl -n argo cp pvc-inspector-python:/data/water_timeseries/region_lake_polygons ./data/region_lake_polygons
```

(Swap in `pvc-inspector-light` if that's the pod you started instead.) These paths match what
`dynamic_world_dir`/`dynamic_world_data`, `output_dir`, `vector_lake_file`, and
`region_lake_polygons_dir` point to locally in the example `.env` files (see
`historical_run/example.env` and `near_real_time/example.env`) - copy the data into the paths
those variables reference and you should be able to run the pipeline locally against it.

`dynamic_world_downloads` (a subdir under `dynamic_world_data`) is also used by the live
pipeline (download/merge scripts), but it's scratch space the pipeline populates itself as it
runs - you don't need to pre-populate it.

`kubectl cp` streams a tar through the Kubernetes API server and can be slow, or drop, for large
directories. For a big pull, `du -sh` the source directory first (see above) so you know what
you're in for, and consider the rsync server below instead.

## Downloading large directories with rsync

`storage_setup/rsync-server.yaml` starts a pod (`pvc-rsync-server`) that serves the whole PVC as
a **read-only** rsync module named `data`. Over a port-forward, `rsync` can resume a partial
transfer instead of starting over.

Start it and wait for it to be Running:

```bash
kubectl -n argo apply -f storage_setup/rsync-server.yaml
```

```bash
kubectl -n argo get pod pvc-rsync-server -w
```

In a second terminal, forward a local port to it and leave it running:

```bash
kubectl -n argo port-forward pod/pvc-rsync-server 8873:873
```

Then pull a directory (paths are relative to `/data`):

```bash
rsync -az --partial --progress rsync://localhost:8873/data/water_timeseries/output/ ./output/
```

Port-forwards tend to drop during long transfers. `storage_setup/download.sh` wraps the rsync in
a loop that restarts the port-forward and retries until the transfer completes. It reads its
settings from the environment and expects the first port-forward to be started by you, so run it
in place of the two commands above:

```bash
export NAMESPACE=argo POD=pvc-rsync-server LOCAL_PORT=8873 DEST=./pvc-copy/
kubectl -n $NAMESPACE port-forward $POD $LOCAL_PORT:873 & export PF_PID=$!
bash storage_setup/download.sh
```

`download.sh` copies the whole volume (`rsync://.../data/`) to `$DEST`. Edit the source path in it
to pull only part of the volume.

Remove the server when you're done:

```bash
kubectl -n argo delete -f storage_setup/rsync-server.yaml
```

## Uploading with the WebDAV server

`storage_setup/webdav-upload-server.yaml` runs `rclone serve webdav` against the PVC and exposes
it over HTTPS through the cluster's Traefik ingress at
`https://water-timeseries-upload.software-dev.ncsa.illinois.edu`. Unlike `kubectl cp` and
port-forwards, this doesn't go through the API server, so large uploads don't time out. The DAV
root is `/data`, so a file PUT to `.../water_timeseries/foo.nc` lands at
`/data/water_timeseries/foo.nc`.

This endpoint is **writable**. Create the login once (pick your own username and a strong
password; the secret is not stored in git):

```bash
kubectl create secret generic webdav-upload-creds -n argo --from-literal=username=<username> --from-literal=password=<password>
```

Start the server:

```bash
kubectl -n argo apply -f storage_setup/webdav-upload-server.yaml
```

Upload a file:

```bash
curl -u <username> -T localfile.nc https://water-timeseries-upload.software-dev.ncsa.illinois.edu/water_timeseries/localfile.nc
```

(`curl` prompts for the password when only the username is given, which keeps it out of your
shell history.) For whole directories, any WebDAV client works, e.g. `rclone copy` with a WebDAV
remote pointing at the same URL.

Don't leave a write-enabled endpoint running longer than needed. Tear it down when you're done:

```bash
kubectl -n argo delete -f storage_setup/webdav-upload-server.yaml
```

## From Delta and other NCSA systems

The PVC's backing directory on Taiga is:

```
/taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26
```

so `/data/water_timeseries` in the pods is
`/taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26/water_timeseries`
on Delta. No `kubectl` is needed: `ls`, `cp` and `rsync` it directly from a login node.

This is the live state of the Argo pipeline. Treat it as **read-only**: copy what you need into
your own space (e.g. `/work/hdd/biyc`) rather than writing into it. The Delta snakemake config
reads only the lake vectors from it (see [10. Running on Delta](10-running-on-delta.md)).

## Google Cloud Storage

Results are also published to the `pdg-storage-default` bucket (project `pdg-project-406720`),
which anyone with access to the bucket can read without cluster access.

**From the cluster to GCS.** The `cloud-sync-cron` CronWorkflow
(`argo_workflows/near_real_time/cron_jobs/upload/upload.yaml`) runs every 4 hours and runs
`google_cloud_utils/upload_to_cloud.py`, which uploads only new or changed files:

| From the PVC | To |
| --- | --- |
| `output/**/breakpoint_zarr` directories and `drain_*.parquet` files | `gs://pdg-storage-default/water-timeseries-v2/data/output/` |
| `combined_zarr_datasets/combined_historical_nrt_*.zarr` | `gs://pdg-storage-default/water-timeseries-v2/near-real-time/output/` |

To preview what it would upload, set `dry_run=True` in the `.env` file it is given.

**From GCS to the cluster.** `google_cloud_utils/download.yaml` is a one-off Kubernetes Job that
runs `google_cloud_utils/download_from_cloud.py` to copy a bucket folder onto the PVC (as
written, `gs://pdg-storage-default/water_timeseries_v2/data/` into
`/data/water_time_series_v2/input`; edit the args before using it for something else):

```bash
kubectl -n argo apply -f google_cloud_utils/download.yaml
```

**From GCS to your machine.** With `gcloud auth application-default login` done, the same script
runs locally:

```bash
python google_cloud_utils/download_from_cloud.py pdg-storage-default pdg-project-406720 water-timeseries-v2/data/output/ ./gcs-output/
```

or use `gcloud storage cp -r gs://pdg-storage-default/<path> <local dir>`.

**From your machine to GCS.** `upload_utils/upload_to_cloud_bucket.py` uploads a file or
directory to `gs://<bucket>/<base-path>/<YYYYMMDD>/` (defaults: bucket `pdg-storage-default`,
base path `water_timeseries_v2/test_data/test_output`):

```bash
python upload_utils/upload_to_cloud_bucket.py ./output --base-path water_timeseries_v2/test_data/test_output
```
