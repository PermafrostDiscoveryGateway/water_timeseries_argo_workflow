# 10. Running the Test Pipeline with Snakemake on Kubernetes

The snakemake version of the test pipeline
([7. Running the testing pipeline](07-running-the-testing-pipeline.md#running-the-test-pipeline-locally-with-snakemake))
can also run on a Kubernetes cluster, using the
[Kubernetes executor plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/kubernetes.html).
It runs the same `snakemake/Snakefile` against one explicit month, but each step runs in its own
pod, with the same image, `/data/water_timeseries` layout and resource requests as the Argo test
pipeline.

Everything for this lives in its own `snakemake` namespace, separate from `argo`. It has its own
volume and its own copies of the secrets, so a snakemake run can never touch the Argo pipeline's
data.

## How it works

```
controller Job (snakemake --profile .../kubernetes)          namespace: snakemake
   │  submits one Kubernetes Job per rule, waits for it
   ├─▶ download (TEST)      ─┐
   ├─▶ download (EURASIA3)   │  every pod: same image,
   ├─▶ merge_and_backfill …  │  snakemake-share PVC at /data,
   ├─▶ process …             │  credentials from the controller
   └─▶ create_new_historical_file / create_historical_zarr_archive
```

- **The controller** is a Kubernetes Job that runs the main `snakemake` process inside the
  cluster. It has to be in the cluster, not on your laptop, because snakemake decides what's
  finished by checking the done-markers the job pods write to the shared volume.
- **Each rule** (`download`, `merge_and_backfill`, `force_merge`, `process`, ...) becomes its
  own Kubernetes Job. The job pod runs the rule's Python script the same way the local run does.
- **The shared volume** (`snakemake-share`, mounted at `/data`) holds the inputs, outputs,
  logs and done-markers. The scripts read and write `/data` directly, as in Argo.
- **The done-markers** are the only files snakemake itself tracks. The Kubernetes executor
  always assumes the pods share no filesystem, so snakemake keeps the markers in a "storage"
  directory on the volume (`/data/water_timeseries/snakemake_storage`, via the `fs` storage
  plugin). It copies them into and out of each job pod with rsync.
- **The image** (`snakemake/Dockerfile`) is the same environment as the Argo image, plus
  snakemake and the workflow files. The controller and every job use it.
- **Credentials**: the controller reads the `earth-engine-creds` and `personal-gcp-creds`
  secrets and passes them to each job pod as environment variables. The Snakefile
  (`credential_env()`) writes them to files and points the scripts at them.

The files involved:

| File | What it is |
|------|------------|
| `snakemake/Snakefile` | The pipeline (same as for local runs) |
| `snakemake/config-kubernetes.yaml` | Overrides of `config.yaml` for the cluster (paths, no uv build) |
| `snakemake/profiles/kubernetes/config.yaml` | Executor settings, per-rule resources, concurrency limits |
| `snakemake/Dockerfile` | The image |
| `.github/workflows/build_snakemake_image.yml` | Builds and pushes the image |
| `snakemake/kubernetes/*.yaml` | Namespace, PVC, service accounts/RBAC, LimitRange, inspector pod, polygon job, controller job |
| `snakemake/kubernetes/seed-from-argo.sh` | Copies the inputs from the Argo volume onto the new one |

## Differences from the Argo test pipeline

These are the same as for the local snakemake run:

- Each download step runs once, with no retries.
- The day-15+ `download-region-last-attempts` step is skipped. Each region always gets the
  forced best-effort merge.
- It runs against one explicit month (`TARGET_DATE`) instead of "today".

---

## One-time setup

All commands are run from the repo root on your machine.

### 1. Connect to the cluster

Make sure `kubectl` is pointed at the right cluster and that you're logged in:

```bash
kubectl config use-context software-dev
```

```bash
kubectl get ns
```

If you get `Forbidden` / `system:unauthenticated`, your Rancher token has expired. Download a
fresh kubeconfig from Rancher (cluster → **Download KubeConfig**), or generate a new API token,
and try again.

### 2. Create the namespace

```bash
kubectl apply -f snakemake/kubernetes/namespace.yaml
```

!!! note "Rancher-managed clusters"
    If you're not allowed to create namespaces with `kubectl`, or want the namespace to belong to
    your Rancher project so its quotas and permissions apply, create it from the Rancher UI
    instead: **Cluster → Projects/Namespaces → Create Namespace** in your project, named
    `snakemake`.

Check that you can manage the things the pipeline needs in it:

```bash
for r in pods jobs.batch secrets persistentvolumeclaims serviceaccounts roles.rbac.authorization.k8s.io rolebindings.rbac.authorization.k8s.io; do
  echo "$r: $(kubectl auth can-i create $r -n snakemake)"
done
```

All of them should be `yes`.

If the namespace belongs to a Rancher project with resource quotas, check what's allowed:

```bash
kubectl describe resourcequota -n snakemake
```

```bash
kubectl describe limitrange -n snakemake
```

The biggest steps request ~14GB of memory and 2 CPUs each, and up to 3 can run at once.

### 3. Create the volume (PVC)

PVCs belong to one namespace, so `snakemake` needs its own. It uses the same `nfs-taiga` storage
class as `argo-workflows-share`:

```bash
kubectl apply -f snakemake/kubernetes/pvc.yaml
```

Check that it's bound. This can take a minute; `STATUS` should be `Bound`:

```bash
kubectl get pvc snakemake-share -n snakemake
```

If it stays `Pending`, check the events (for example, a missing storage class on this cluster):

```bash
kubectl describe pvc snakemake-share -n snakemake
```

```bash
kubectl get storageclass
```

The new volume is empty. It's seeded with the pipeline's inputs in [step 7](#7-seed-the-volume).

### 4. Set up the secrets

The pipeline needs the same three secrets as the Argo pipeline
([4. Setting up secrets](04-setting-up-secrets.md)), in the `snakemake` namespace:

- `ghcr-secret` — pulling the image from ghcr
- `earth-engine-creds` — Earth Engine credentials (key `credentials`)
- `personal-gcp-creds` — gcloud application default credentials (key `key.json`)

Secrets can't be shared across namespaces. Either copy them from `argo` (if they exist there on
this cluster) or create them from your local credentials.

**Option A — copy them from the `argo` namespace** (needs `jq`):

```bash
for s in ghcr-secret earth-engine-creds personal-gcp-creds; do
  kubectl get secret "$s" -n argo -o json \
    | jq 'del(.metadata.namespace, .metadata.uid, .metadata.resourceVersion, .metadata.creationTimestamp, .metadata.ownerReferences, .metadata.managedFields, .metadata.annotations)' \
    | kubectl apply -n snakemake -f -
done
```

**Option B — create them from your local credentials**, the same way as in
[4. Setting up secrets](04-setting-up-secrets.md), with `-n snakemake`:

```bash
kubectl create secret generic personal-gcp-creds -n snakemake \
    --from-file=key.json=$HOME/.config/gcloud/application_default_credentials.json
```

```bash
kubectl create secret generic earth-engine-creds -n snakemake \
    --from-file=credentials=$HOME/.config/earthengine/credentials
```

```bash
export GH_TOKEN=$(gh auth status --show-token | grep "Token:" | awk '{print $3}')
```

```bash
kubectl create secret docker-registry ghcr-secret -n snakemake \
    --docker-server=ghcr.io \
    --docker-username=<your-github-username> \
    --docker-password="${GH_TOKEN}" \
    --docker-email=<your-email>
```

Check all three exist, with the right keys:

```bash
kubectl get secret personal-gcp-creds earth-engine-creds ghcr-secret -n snakemake
```

```bash
kubectl get secret personal-gcp-creds -n snakemake -o jsonpath='{.data}' | jq 'keys'
```

```bash
kubectl get secret earth-engine-creds -n snakemake -o jsonpath='{.data}' | jq 'keys'
```

Expected: `["key.json"]` and `["credentials"]`.

!!! note
    `personal-gcp-creds` holds your personal gcloud login. If you run `gcloud auth
    application-default login` again later, recreate the secret (delete it, then create it again)
    so the pods get the new credentials.

### 5. Service accounts and permissions

```bash
kubectl apply -f snakemake/kubernetes/rbac.yaml
```

This creates:

- `snakemake-controller` — used by the controller Job. It may create, watch and delete Jobs and
  pods, read pod logs, and create/delete Secrets, only in the `snakemake` namespace. The executor
  passes environment variables to job pods through a per-run Secret.
- `snakemake-job` — used by every job pod. It has no API access, and exists so the pods get the
  `ghcr-secret` image pull secret. The executor can't set an image pull secret itself.

Check it:

```bash
kubectl auth can-i create jobs.batch -n snakemake --as=system:serviceaccount:snakemake:snakemake-controller
```

```bash
kubectl auth can-i create jobs.batch -n snakemake --as=system:serviceaccount:snakemake:snakemake-job
```

The first should be `yes` and the second `no`. The `--as` check itself needs impersonation
rights; if you don't have them, skip it.

### 6. Optional: default resource limits

The Kubernetes executor only sets resource **requests** on job pods, not limits. On a shared
cluster, it's a good idea to give them the same limits as the Argo test pipeline (20Gi memory,
4 CPU):

```bash
kubectl apply -f snakemake/kubernetes/limitrange.yaml
```

This also makes pods schedulable if the namespace's quota requires every container to have
limits.

### 7. Seed the volume

The pipeline expects these inputs on the volume, at the same paths as on the Argo volume:

| Path on the volume | What it is |
|--------------------|------------|
| `/data/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet` | Global lake vector file (~3GB) |
| `/data/water_timeseries/region_lake_polygons/` | Per-region lake polygons (~2.5GB), see [7. Running the testing pipeline](07-running-the-testing-pipeline.md#generate-the-region-lake-polygons) |
| `/data/water_timeseries/test/dynamic_world_data/*.nc` | The test run's baseline netCDF files: `lakes_dw_V2d_compressed.nc` (used by `process_NRT.py`) and the latest `dynamic_world_historical_*.nc` (used by `create_new_historical_file.py`) |

Start the inspector pod in the `snakemake` namespace:

```bash
kubectl apply -f snakemake/kubernetes/inspector.yaml
```

```bash
kubectl -n snakemake wait --for=condition=Ready pod/snakemake-inspector --timeout=120s
```

**Option A — copy from the Argo volume** (recommended, if `argo-workflows-share` is on this
cluster). The inputs are taken from these places on the Argo volume:

| On the Argo volume | Copied to |
|--------------------|-----------|
| `input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet` | same path |
| `region_lake_polygons/` | same path |
| `test/dynamic_world_data/dynamic_world_historical_2026-06.nc` | same path (the test baseline) |
| `dynamic_world_data/lakes_dw_V2d_compressed.nc` | `test/dynamic_world_data/` (it isn't in the Argo test directory) |

A volume can't be mounted in two namespaces, so the copy goes from an inspector pod in `argo`
straight to the one in `snakemake`, over the cluster network. Streaming it through your machine
with `kubectl exec ... | kubectl exec ...` works in principle, but for ~14GB it's slow and the
Rancher connection tends to time out partway.

Make sure the Python inspector is running in `argo`
([9. Data access](09-data-access.md)). The script needs `python3` and GNU `tar` in it:

```bash
kubectl -n argo get pod pvc-inspector-python
```

```bash
kubectl -n argo apply -f storage_setup/python-inspector.yaml
```

Then run the copy. It takes a few minutes, prints progress, and checks both ends' exit codes:

```bash
snakemake/kubernetes/seed-from-argo.sh
```

The Argo volume is only read from. To use a different baseline, e.g. a newer
`dynamic_world_historical_*.nc` from the Argo test directory, set `BASELINE=<file name>` in front
of the command. `tar` keeps the files' modification times, which matters because
`create_new_historical_file.py` picks the newest `dynamic_world_historical_*.nc` as its baseline.

Check that the sizes match the Argo copies:

```bash
kubectl -n snakemake exec snakemake-inspector -- sh -c 'cd /data/water_timeseries; ls -l input test/dynamic_world_data; ls region_lake_polygons'
```

```bash
kubectl -n argo exec pvc-inspector-python -- sh -c 'cd /data/water_timeseries; ls -l input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet test/dynamic_world_data/*.nc dynamic_world_data/lakes_dw_V2d_compressed.nc; ls region_lake_polygons'
```

**Option B — upload from your machine / from GCS.** This goes through your machine, so it's
slower and can time out on large files; re-run a command if it does. First create the
directories:

```bash
kubectl -n snakemake exec snakemake-inspector -- mkdir -p /data/water_timeseries/input /data/water_timeseries/test/dynamic_world_data
```

The vector file can be streamed straight from the bucket:

```bash
gcloud storage cat gs://pdg-storage-default/water_timeseries_v2/data/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet \
  | kubectl -n snakemake exec -i snakemake-inspector -- sh -c 'cat > /data/water_timeseries/input/Nitze_etal_Lakes_filtered_full_set_V2d.parquet'
```

Then either generate the region polygons on the cluster (needs the vector file first, and
the image from [step 8](#8-build-the-image)):

```bash
kubectl -n snakemake create -f snakemake/kubernetes/generate-region-lake-polygons.yaml
```

...or copy local copies up, for example the ones under `data/` from a local run.
`COPYFILE_DISABLE=1` stops macOS `tar` from adding `._*` metadata files:

```bash
COPYFILE_DISABLE=1 tar cf - -C data region_lake_polygons | kubectl -n snakemake exec -i snakemake-inspector -- tar xf - -C /data/water_timeseries
```

```bash
COPYFILE_DISABLE=1 tar cf - -C data/dynamic_world_test . | kubectl -n snakemake exec -i snakemake-inspector -- tar xf - -C /data/water_timeseries/test/dynamic_world_data
```

Check what's there:

```bash
kubectl -n snakemake exec snakemake-inspector -- sh -c 'du -sh /data/water_timeseries/*; ls -la /data/water_timeseries/test/dynamic_world_data'
```

### 8. Build the image

The image is built by the `Build and Push Snakemake Image` GitHub Action
(`.github/workflows/build_snakemake_image.yml`). It runs on pushes to `main` and to this branch
that touch the pipeline code, or manually from the **Actions** tab. It's pushed as:

```
ghcr.io/permafrostdiscoverygateway/water_timeseries_argo_workflow:snakemake-<branch>
ghcr.io/permafrostdiscoverygateway/water_timeseries_argo_workflow:snakemake-<short sha>
```

The image tag is set in **three places**; keep them the same:

- `snakemake/profiles/kubernetes/config.yaml` (`container-image`, used by the job pods)
- `snakemake/kubernetes/controller-job.yaml` (`image`)
- `snakemake/kubernetes/generate-region-lake-polygons.yaml` (`image`)

Because the workflow files are baked into the image, **any change to the Snakefile, the configs
or the scripts needs a new image** before it shows up on the cluster, the same as for Argo.

To build locally instead (from the repo root; `snakemake/Dockerfile.dockerignore` keeps `data/`
and the local venvs out of the build):

```bash
docker build -f snakemake/Dockerfile -t ghcr.io/permafrostdiscoverygateway/water_timeseries_argo_workflow:snakemake-59-adopt-the-test-pipeline-using-snakemake-kubernetes .
```

The cluster nodes are most likely `linux/amd64`. On an Apple Silicon Mac, add
`--platform linux/amd64` to the build, then `docker push` the tag.

---

## Running the pipeline

### Start a run

Set the month to run in `snakemake/kubernetes/controller-job.yaml` (`TARGET_DATE`, `YYYY-MM`),
then submit the controller. Use `create`, not `apply`, because the Job uses `generateName`, so
every run gets a new name:

```bash
kubectl -n snakemake create -f snakemake/kubernetes/controller-job.yaml
```

### Follow it

The controller's log is snakemake's own output: which jobs were submitted, finished or failed.

```bash
kubectl -n snakemake get jobs
```

```bash
kubectl -n snakemake logs -f job/<snakemake-test-pipeline-xxxxx>
```

The individual steps show up as `snakejob-...` Jobs and pods:

```bash
kubectl -n snakemake get pods | grep snakejob-
```

Each step's script output is also written to the volume, which is the easiest place to read it
because finished job pods are deleted:

```
/data/water_timeseries/snakemake_work/<target_date>/
    logs/<stage>_<region>.log        script output, one per step
    scratch/<stage>/<region>/.env    the .env each step ran with
    nrt_pipeline_log.jsonl           one line per step with its exit code
/data/water_timeseries/snakemake_storage/
    markers/<target_date>/           done-markers (what snakemake checks)
    snakemake-workflow-sources.*     workflow files snakemake ships to each job pod
```

```bash
kubectl -n snakemake exec snakemake-inspector -- ls /data/water_timeseries/snakemake_work/2025-08/logs
```

```bash
kubectl -n snakemake exec snakemake-inspector -- tail -50 /data/water_timeseries/snakemake_work/2025-08/logs/process_TEST.log
```

To keep failed pods around for `kubectl logs` / `kubectl describe`, set
`kubernetes-omit-job-cleanup: true` in the profile (needs a new image).

### Re-running

Finished steps are recorded by their done-markers on the volume. A new controller run only does
what isn't finished yet. To redo a month from scratch, delete its markers first, and its logs if
you want a clean slate. The pipeline's own data (downloads, merge files, output) is separate;
see [7. Seed the volume](#7-seed-the-volume) for what's input and what's generated.

```bash
kubectl -n snakemake exec snakemake-inspector -- rm -rf /data/water_timeseries/snakemake_storage/markers/2025-08 /data/water_timeseries/snakemake_work/2025-08
```

Other snakemake options go in `SNAKEMAKE_EXTRA_ARGS` in `controller-job.yaml`. For example,
`--dry-run` shows what a run would do without starting anything, and `--forcerun process`
re-runs the process steps. On its own that resumes from each region's
`incremental_results_<target_date>.parquet` and only processes lakes that are missing from it.
Add `--config recompute_process=true` to delete that month's process outputs first and
recompute every lake.

Snakemake's own state (`.snakemake/`, including its lock) lives inside the controller pod and
goes away with it, so a killed run never leaves a lock behind. This also means nothing stops
two controllers from running the same month at once, so only start one per month.

### Stopping a run

Delete the controller Job, then any step Jobs it left behind:

```bash
kubectl -n snakemake delete job <snakemake-test-pipeline-xxxxx>
```

```bash
kubectl -n snakemake get jobs -o name | grep snakejob- | xargs kubectl -n snakemake delete
```

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Pods stuck in `ImagePullBackOff` | `ghcr-secret` missing or expired in `snakemake`, or the image tag doesn't exist yet (check the GitHub Action) |
| Pods stuck `Pending` | Not enough free CPU/memory on the nodes, or the namespace quota. Check `kubectl describe pod`. Lower concurrency with `resources:` in the profile |
| Pods rejected with "must specify limits" | The namespace quota requires limits; apply `limitrange.yaml` ([step 6](#6-optional-default-resource-limits)) |
| Controller: `Forbidden ... cannot create resource "jobs"` | `rbac.yaml` not applied, or the controller isn't running as `snakemake-controller` |
| Controller fails about `EE_CREDENTIALS_JSON` / `GCP_CREDENTIALS_JSON` not being set, or the pod is stuck in `CreateContainerConfigError` | `earth-engine-creds` / `personal-gcp-creds` missing, or they don't have the `credentials` / `key.json` keys |
| Earth Engine auth errors in the step logs | Expired credentials: recreate the secrets ([step 4](#4-set-up-the-secrets)) |
| `Vector lake file not found` / no lakes for a region | Volume not seeded ([step 7](#7-seed-the-volume)) |
| `MissingOutputException` for a `.done` marker | Volume slow to show the file; raise `latency-wait` in the profile |
| `If no shared filesystem is assumed ... a default storage provider ... has to be set` | The image predates the storage settings in the profile; rebuild it |
| Job pod fails in `pip install ... snakemake-storage-plugin-fs` or `--deploy-sources` before the step starts | Job pods need internet access (PyPI) to install the storage plugin, and read access to `/data/water_timeseries/snakemake_storage` |
| `rsync: not found` | The image predates `rsync` being added to `snakemake/Dockerfile`; rebuild it |

## Cleaning up

Remove the inspector pod when you're done with it:

```bash
kubectl -n snakemake delete -f snakemake/kubernetes/inspector.yaml
```

The rest of the namespace (volume, secrets, service accounts) can stay for the next run.
