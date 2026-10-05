# 10. Running on Delta (NCSA / ACCESS)

The test pipeline can run on [Delta](https://docs.ncsa.illinois.edu/systems/delta/en/latest/index.html)
instead of the Argo cluster, using the snakemake version described in
[7. Running the testing pipeline](07-running-the-testing-pipeline.md#running-the-test-pipeline-locally-with-snakemake).
Snakemake runs on a Delta login node and submits every pipeline step as a Slurm job.

The Delta-specific settings live in two files:

- [`snakemake/config.delta.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/snakemake/config.delta.yaml)
  — paths, credentials and the venv location. Anything not set there comes from `snakemake/config.yaml`.
- [`snakemake/profiles/delta/config.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/snakemake/profiles/delta/config.yaml)
  — the Slurm executor, the account (`biyc-delta-cpu`), partition, and the limits on concurrent
  Earth Engine downloads and process steps.

## Where things live

Following the [Delta data management guidance](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html):

| Path | What goes there |
| --- | --- |
| `/projects/biyc/water_timeseries` | This repo checkout and the venv (`venv/`). Shared and persistent. |
| `/work/hdd/biyc/water_timeseries` | Everything jobs read and write: `dynamic_world_data/`, `output/`, `combined_zarr_datasets/`, `snakemake_work/` (logs, markers, per-stage `.env` files). |
| `/tmp/$USER/water_timeseries` (compute node) | Temp NetCDF and Dask spill space. Node-local SSD, wiped after each job. |
| `/taiga/.../water_timeseries` | The Argo cluster's shared volume. Only the read-only lake vectors are read from it. |
| `~/.config/water_timeseries/` | The Google Cloud and Earth Engine credentials (owner-only). |

Both `/projects/biyc` and `/work/hdd/biyc` are shared by everyone on the biyc allocation. Keep
them group-writable (`chmod -R g+rwX`, setgid on directories) so anyone on the allocation can run
the pipeline. Neither is backed up or snapshotted; only your home directory (`/u`) has daily
snapshots.

## One-time setup

Run everything below on a Delta login node (`ssh login.delta.ncsa.illinois.edu`) unless noted.

### 1. Check out the repo under /projects

The venv path in `config.delta.yaml` is `/projects/biyc/water_timeseries/venv`, so the checkout
is expected next to it:

```bash
cd /projects/biyc && git clone https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow.git water_timeseries
```

```bash
chmod -R g+rwX /projects/biyc/water_timeseries && find /projects/biyc/water_timeseries -type d -exec chmod g+s {} +
```

Running from a checkout somewhere else (e.g. your home directory) also works, because the
Snakefile resolves paths relative to itself; the venv is still built at the path in the config.

### 2. Install uv, snakemake and the Slurm plugin

These go in `~/.local/bin`, so each person running the pipeline does this once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv tool install snakemake --with snakemake-executor-plugin-slurm
```

The profile needs snakemake 8 or later. `uv` is also what the `build_env` rule uses to build the
pipeline's own Python environment.

### 3. Copy the credentials

On a machine where `kubectl` can reach the Argo cluster, copy the cluster secrets to Delta as
described in
[Copying the cluster's credentials to Delta](07-running-the-testing-pipeline.md#copying-the-clusters-credentials-to-delta).
Then, on Delta, check that both files exist and are `-rw-------`:

```bash
ls -l ~/.config/water_timeseries/
```

Each person who runs the pipeline needs their own copy, since the paths are under `~`.

### 4. Seed the input data

`dynamic_world_data` must contain the historical `lakes_dw_*.nc` file(s) before the first run.
Link them from the Argo test data:

```bash
mkdir -p /work/hdd/biyc/water_timeseries/dynamic_world_data
```

```bash
ln -s /taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26/water_timeseries/test/dynamic_world_data/*.nc /work/hdd/biyc/water_timeseries/dynamic_world_data/
```

The per-region lake polygons are read from the Argo volume
(`region_lake_polygons_dir` in `config.delta.yaml`), so the
[region lake polygon job](07-running-the-testing-pipeline.md#generate-the-region-lake-polygons)
must have been run on the cluster at least once.

## Running the pipeline

All commands are run from the repo root (`/projects/biyc/water_timeseries`).

### Dry run

Check what will run without submitting anything:

```bash
snakemake -s snakemake/Snakefile --configfile snakemake/config.delta.yaml --profile snakemake/profiles/delta --config target_date=2025-08 -n
```

### Real run

Snakemake has to keep running on the login node for the whole pipeline (it submits jobs and waits
for them), so start it inside `tmux`. Delta has several login nodes; note which one you're on so
you can get back to the same session:

```bash
hostname
```

```bash
tmux new -s wts
```

```bash
snakemake -s snakemake/Snakefile --configfile snakemake/config.delta.yaml --profile snakemake/profiles/delta --config target_date=2025-08
```

Detach with `Ctrl-b d`. To reattach later, ssh to the same login node (e.g.
`ssh dt-login03.delta.ncsa.illinois.edu`) and run `tmux attach -t wts`.

The first run builds the venv on the login node (`build_env`), which takes a few minutes before
any Slurm jobs appear. Its log is `/work/hdd/biyc/water_timeseries/snakemake_work/build_env.log`.

### Following jobs

```bash
squeue -u $USER
```

Per-step logs, `.env` files and done-markers are in
`/work/hdd/biyc/water_timeseries/snakemake_work/<target_date>/`. Delete that directory (or pass
`--forcerun <rule>`) to re-run a month. If a run is interrupted, re-running the same command
picks up where it left off; if snakemake complains the directory is locked, add `--unlock` once,
then run again without it.

## Allocation, quotas and limits

### Compute time (service units)

Delta charges in service units (SUs). On CPU nodes, one SU is one core for one hour **or** 2 GB of
memory for one hour, whichever is larger, for the resources the job *reserves* (not what it
actually uses), for the time it actually runs. See
[Job Accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html).

Check the allocation's remaining balance (shared by everyone on the project):

```bash
accounts
```

The `Project` column lists the charge accounts (`biyc-delta-cpu` for this pipeline) and
`Balance (Hours)` is the remaining SUs.

See what the project has been charged, per user, over the last 30 days:

```bash
/sw/user/scripts/jobcharge -a biyc-delta-cpu -d 30
```

Add `--detail` for a per-job breakdown.

For this pipeline, memory dominates the charge: `merge_and_backfill`, `force_merge`, `process`,
`create_historical_zarr_archive` and `create_new_historical_file` each reserve 20 GB, so they
cost 10 SUs per hour they run; `download` reserves 8 GB (4 SUs/hour). Lowering `mem_mb` in the
Snakefile, where the steps allow it, is the main way to reduce the cost.

If jobs sit in the queue with reason `QOSGrpBillingMinutes`, the allocation doesn't have enough
balance left for the jobs requested; the PI needs to request a supplement.

### Partition limits

The profile submits to the `cpu` partition, which allows jobs of up to 48 hours. Each rule's
`runtime` (minutes) sets its Slurm time limit; steps in this pipeline ask for 2–4 hours.
See [Partitions](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html)
for the other partitions (`cpu-preempt` is charged at half rate but jobs can be preempted).

### Storage

```bash
quota
```

shows usage and limits for your home directory and for each allocation directory (`/projects/biyc`,
`/work/hdd/biyc`, `/work/nvme/biyc`). Errors about `/dev/shm/usertmp` are harmless. The
allocation directories are shared by the whole project, and have both a size limit and a file
count limit; a Python venv with the geospatial stack holds tens of thousands of files, so the
file count on `/projects` is the limit most likely to be reached. Avoid extra venvs or checkouts
there.
