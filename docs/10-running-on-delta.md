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
| `/work/hdd/biyc/water_timeseries` | Only the results: `output/`, `combined_zarr_datasets/`, and the PMTiles (`precomputed_nrt/`, `nrt_tiles/`; `main/...` for the production data). |
| `/tmp/$USER/water_timeseries` (compute node) | Temp NetCDF and Dask spill space. Node-local SSD, wiped after each job. |
| `/taiga/.../water_timeseries` | The Argo cluster's shared volume, used exactly as the Argo test pipeline uses it: `base_dir`, `input/` (lake vectors), `region_lake_polygons/`, and `test/dynamic_world_data/` (read and written). Delta's own `test/snakemake_work/` (logs, markers, per-stage `.env` files) is here too. |
| `~/.config/water_timeseries/` | The Google Cloud and Earth Engine credentials (owner-only). |

`/taiga/...` is the full path in [`snakemake/config.delta.yaml`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/snakemake/config.delta.yaml).
Because Delta and the Argo test pipeline share the Taiga directories, don't run both at the same
time. Delta also needs write access to `water_timeseries/test/` on Taiga, which has to be granted
from the cluster side (see [step 4](#4-give-the-allocations-group-write-access-to-the-taiga-test-directory)).

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

The profile needs snakemake 8 or later. `uv` is also what the `build_environment` rule uses to build the
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

### 4. Give the allocation's group write access to the Taiga test directory

Delta reads and writes `water_timeseries/test/` on Taiga (`dynamic_world_data/` and
`snakemake_work/`), but the Argo pods create everything there as `root` with mode `755`, so a
Delta user can't write to it. Without this step, the first job fails with:

```
WorkflowError:
Failed to create output directory .../water_timeseries/test/snakemake_work/2025-08/markers/TEST.
PermissionError: [Errno 13] Permission denied: '.../water_timeseries/test/snakemake_work'
```

Only root can change ownership there, so it's done from a pod on the cluster, where the volume is
mounted at `/data`. Give `test/` to the allocation's Unix group (`delta_biyc` for the biyc
allocation) and make it group-writable, with setgid on directories so new ones keep the group.

On Delta, get the group's numeric ID (the number after the second `:`). The pod doesn't know
Delta's group names, so it needs the number:

```bash
getent group delta_biyc
```

On a machine with cluster access, start the inspector pod (see
[9. Data access](09-data-access.md#browsing-the-volume-with-a-pvc-inspector-pod)):

```bash
kubectl -n argo apply -f storage_setup/python-inspector.yaml
```

Then, replacing `<GID>` with the number from above:

```bash
kubectl -n argo exec pvc-inspector-python -- sh -c 'cd /data/water_timeseries/test && chgrp -R <GID> . && chmod -R g+rwX . && find . -type d -exec chmod g+s {} +'
```

Check from Delta that you can write there:

```bash
mkdir -p /taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26/water_timeseries/test/snakemake_work && ls -la /taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26/water_timeseries/test/
```

The directories should show the `delta_biyc` group (or its number) with `rws` group permissions.

!!! warning "Re-apply after Argo test runs"
    Files and directories an Argo test run creates afterwards are again owned by `root` and not
    group-writable. If a Delta run later fails with `Permission denied` on something under
    `test/`, run the `chgrp`/`chmod` command above again.

The rest of the Taiga volume (`input/`, `region_lake_polygons/`, the production
`dynamic_world_data/`) is only read by Delta, so it doesn't need any changes. It is readable by
everyone.

### 5. Check the Dynamic World data

`dynamic_world_data` is the Argo test pipeline's own directory on Taiga,
`/taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26/water_timeseries/test/dynamic_world_data`, so Delta and the Argo test pipeline use the same files. It must
contain the historical NetCDF file (`lakes_dw_*.nc` or `dynamic_world_historical_*.nc`) before the
first run:

```bash
ls -la /taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26/water_timeseries/test/dynamic_world_data/
```

To use a production historical file for testing, copy it in from the inspector pod. A copy from
Delta fails with `Permission denied` unless step 4 has been done, and files created by the pod
need step 4 re-applied afterwards anyway. `-p` keeps the timestamps; the scripts use the newest
`.nc` file as the baseline.

```bash
kubectl -n argo exec pvc-inspector-python -- cp -p /data/water_timeseries/dynamic_world_data/dynamic_world_historical_2026-07.nc /data/water_timeseries/test/dynamic_world_data/
```

The pipeline also writes `downloads/`, `merge/` and the new historical file into this directory,
so a Delta run and an Argo test run shouldn't run at the same time.

The per-region lake polygons are read from the Argo volume
(`region_lake_polygons_dir` in `config.delta.yaml`), so the
[region lake polygon job](07-running-the-testing-pipeline.md#generate-the-region-lake-polygons)
must have been run on the cluster at least once.

## Running the pipeline

### With the helper script

[`snakemake/run_delta.sh`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/snakemake/run_delta.sh)
does all of the steps below in one go. It changes to the repo root, installs uv and Snakemake if
they're missing, checks that the credentials exist, and starts the run inside a `tmux` session
named `water_timeseries-<date>`:

```bash
/projects/biyc/water_timeseries/snakemake/run_delta.sh 2025-08 -n
```

```bash
/projects/biyc/water_timeseries/snakemake/run_delta.sh 2025-08
```

The first argument is the target month, and anything after it is passed on to Snakemake. A dry
run (`-n`) runs directly instead of in `tmux`. There's no Python environment to activate:
Snakemake comes from `uv tool install`, and the pipeline steps use the venv that `build_environment`
builds.

### By hand

All commands are run from the repo root (`/projects/biyc/water_timeseries`).

#### Dry run

Check what will run without submitting anything:

```bash
snakemake -s snakemake/Snakefile --configfile snakemake/config.delta.yaml --profile snakemake/profiles/delta --config target_date=2025-08 -n
```

#### Real run

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

The first run builds the venv on the login node (`build_environment`), which takes a few minutes before
any Slurm jobs appear. Its log is `/taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26/water_timeseries/test/snakemake_work/build_environment.log`.

#### Following jobs

```bash
squeue -u $USER
```

Per-step logs, `.env` files and done-markers are in
`/taiga/ncsa/radiant/bbfa/software-dev/argo-argo-workflows-share-pvc-082f0001-1fbe-4f7d-91d7-410c880ebd26/water_timeseries/test/snakemake_work/<target_date>/`. Delete that directory (or pass
`--forcerun <rule>`) to re-run a month. If a run is interrupted, re-running the same command
picks up where it left off; if snakemake complains the directory is locked, add `--unlock` once,
then run again without it.

## PMTiles

The pipeline's last step, `create_nrt_pmtiles`, builds the month's `nrt_<month>_drainage.pmtiles`
from the combined zarr and copies it to `/work/hdd/biyc/water_timeseries/nrt_tiles/`, with the
running drain-breaks table in `.../precomputed_nrt/`. It runs as a Slurm job (2 cores, 14 GB, up
to 4 hours) and works in the compute node's `/tmp`. See
[PMTiles](07-running-the-testing-pipeline.md#pmtiles) for what it does and its settings.

### tippecanoe on Delta

The PMTiles are built with tippecanoe, which `build_environment` compiles from source into the venv's
`bin/` on the login node (see [tippecanoe](07-running-the-testing-pipeline.md#tippecanoe)). The
first run after this was added rebuilds the venv once to do that; steps that already finished
aren't re-run.

The build needs a C++ compiler, `make`, `git` and the sqlite3 and zlib development headers. If it
fails, the end of `build_environment.log` (in `snakemake_work/`) shows why, e.g. `sqlite3.h: No such file
or directory`. In that case, build tippecanoe by hand somewhere you can (e.g. in a conda env or
from a module with the headers) and put it on your `PATH`, then set
`environment.tippecanoe_version: ""` in `snakemake/config.delta.yaml` so `build_environment` skips it.

### Rebuilding only the PMTiles

[`snakemake/Snakefile.pmtiles`](https://github.com/PermafrostDiscoveryGateway/water-timeseries-argo-workflow/blob/main/snakemake/Snakefile.pmtiles)
rebuilds the archives from an existing combined zarr without the rest of the pipeline, e.g. after
the PMTiles code in water-timeseries-v2 changes (details in
[Building only the PMTiles](07-running-the-testing-pipeline.md#building-only-the-pmtiles)). On
Delta, each month is a Slurm job; run it from the repo root, in `tmux` for long runs.

The latest month of this pipeline's own output (`test`):

```bash
snakemake -s snakemake/Snakefile.pmtiles --configfile snakemake/config.delta.yaml --profile snakemake/profiles/delta --forcerun pmtiles_month
```

The production Argo pipeline's data (`main`), for specific months:

```bash
snakemake -s snakemake/Snakefile.pmtiles --configfile snakemake/config.delta.yaml --profile snakemake/profiles/delta --forcerun pmtiles_month --config pmtiles_dataset=main pmtiles_months=2026-07,2026-08
```

`main` reads `water_timeseries/combined_zarr_datasets` on Taiga (read only) and writes its own
drain-breaks table and archives to `/work/hdd/biyc/water_timeseries/main/`, so it never touches
the Argo pipeline's published tiles. Add `build_environment` to `--forcerun` to pick up new commits on the
water-timeseries-v2 branch first. Logs and markers are in `snakemake_work/pmtiles_<dataset>/`
next to the pipeline's.

## Moving to a new allocation or project space

Everything above is tied to the biyc allocation. When moving to a new one (new project code
`<new>`, e.g. after a renewal under a different allocation, or a different Delta project):

1. **Charge account**: `slurm_account` in `snakemake/profiles/delta/config.yaml`
   (`<new>-delta-cpu`; `accounts` lists the names).
2. **Paths** in `snakemake/config.delta.yaml`: `output_dir`, `combined_zarr_datasets`,
   `nrt_precomputed_dir`, `nrt_pmtiles_output_dir` and the `pmtiles.datasets.main` outputs
   (`/work/hdd/<new>/...`) and `environment.path` (`/projects/<new>/water_timeseries/venv`).
   Update the paths in this page and in `snakemake/run_delta.sh`'s usage comment to match.
3. **Checkout and venv**: clone the repo under `/projects/<new>` (step 1). The venv is rebuilt
   automatically on the first run.
4. **Group permissions on Delta**: make the new `/projects/<new>/water_timeseries` and
   `/work/hdd/<new>/water_timeseries` group-writable with setgid on directories (step 1), so
   everyone on the allocation can run the pipeline.
5. **Group permissions on Taiga**: repeat step 4 with the new allocation's group
   (`getent group delta_<new>`), so the new group can write to `water_timeseries/test/`.
6. **Credentials** are per user, under `~`, and don't change (step 3). Anyone new running the
   pipeline needs their own copy.
7. **Results**: copy anything you want to keep from the old `/work/hdd/biyc/water_timeseries`
   (`output/`, `combined_zarr_datasets/`); the Taiga data stays where it is.

If the Argo cluster's volume moves instead (a new PVC), the Taiga path changes: update every
`/taiga/...` path in `snakemake/config.delta.yaml`, then repeat steps 4 and 5.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `PermissionError: [Errno 13] Permission denied` under `.../water_timeseries/test/` | The Taiga test directory (or something an Argo run created in it since) isn't writable by your group. Re-apply [step 4](#4-give-the-allocations-group-write-access-to-the-taiga-test-directory). |
| `cp: cannot create regular file ... Permission denied` copying into `test/dynamic_world_data` | Same; or copy from the inspector pod instead ([step 5](#5-check-the-dynamic-world-data)). |
| `ValueError: max() iterable argument is empty` in a download log | No historical `.nc` file in `dynamic_world_data` ([step 5](#5-check-the-dynamic-world-data)). |
| `Missing credentials file` from `run_delta.sh` | Copy the credentials ([step 3](#3-copy-the-credentials)). |
| Snakemake reports `SLURM status is: 'TIMEOUT'` | A step hit its time limit. For a download, raise that region's `download_runtime_hours` ([Time limits](#time-limits)) and re-run; finished steps and downloaded tiles are kept. |
| `build_environment` fails while building tippecanoe | Missing build tools or headers on the login node; see [tippecanoe on Delta](#tippecanoe-on-delta). |
| `create_nrt_pmtiles` fails with `tippecanoe is not installed or not on PATH` | The venv was built without tippecanoe (`environment.tippecanoe_version: ""`) and none is on `PATH`. |
| `create_nrt_pmtiles` fails with `no drainage_confidence column` | The month was processed with an older water-timeseries library; reprocess it and rebuild the combined zarr. |
| Jobs pending with reason `QOSGrpBillingMinutes` | The allocation is out of SUs (see below). |
| Snakemake says the directory is locked | A previous run was interrupted: run the same command once with `--unlock`, then again without it. |

For any failed step, the Slurm log path is printed in the Snakemake output
(`.snakemake/slurm_logs/rule_<name>/<region>/<jobid>.log` in the repo), and the pipeline's own log
is `<taiga>/water_timeseries/test/snakemake_work/<target_date>/logs/<stage>_<region>.log`.

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

### Time limits

The profile submits to the `cpu` partition, which allows jobs of up to **48 hours**. Slurm kills a
job that reaches its time limit (state `TIMEOUT`). Each rule's `runtime` (minutes) in the
Snakefile sets its limit: 2–4 hours for merge, process and the combine steps.

Downloads can take much longer for large regions, so their limit is set per region, in hours, under
`download_runtime_hours` in `snakemake/config.yaml`:

| Region | Limit |
| --- | --- |
| `EURASIA1`, `EURASIA2`, `CANADA1`–`CANADA4` | 96 hours |
| `EURASIA3` | 24 hours |
| `ALASKA` | 12 hours |
| anything else (`default`, e.g. `TEST`) | 4 hours |

The limit covers the full download and its missing-ID fallback together. A limit over 48 hours
is split into back-to-back jobs of up to 48 hours each: when one is killed at its limit, Snakemake
resubmits the download (`retries` on the `download` rule), and the new job skips the tiles already
on disk and carries on. So 96 hours means up to two 48-hour jobs. The second job waits in the queue
like any other, so the total time can be a bit longer than the limit.

Delta charges only for the time a job actually runs, not its limit, so a generous limit costs
nothing extra if the download finishes early; a longer limit can only mean a slightly longer wait
in the queue. You can lower a running job's limit but not raise it, so change
`download_runtime_hours` before starting a run. To see a running job's elapsed time (`%M`) and
limit (`%l`):

```bash
squeue -u $USER -o "%.10i %.10M %.10l %.45k"
```

Two things to keep in mind for multi-day runs:

- Snakemake itself has to keep running on the login node for the whole time. Login nodes are
  occasionally rebooted for maintenance; if that kills the `tmux` session, start the same command
  again. Finished steps are skipped, and downloads resume from the tiles on disk. Running jobs are
  left alone, so check `squeue` first to avoid starting a second copy of a step.
- A tile that was being written when a job was killed can be left truncated, and the resumed
  download skips any tile file that isn't empty. If a merge later fails reading a file in
  `downloads/`, delete that file and re-run.

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
