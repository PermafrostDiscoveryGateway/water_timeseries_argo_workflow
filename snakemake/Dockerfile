# Image for running the snakemake test pipeline on Kubernetes: the same
# environment as the Argo image (../Dockerfile) plus snakemake and its
# Kubernetes executor, and the workflow files. The controller Job and every
# job pod it starts run this one image.
#
# Build from the repo root; IMAGE_REF must be the tag you push, because the
# controller starts the job pods from it:
#   docker build -f snakemake/Dockerfile --build-arg IMAGE_REF=<image> -t <image> .
FROM ghcr.io/permafrostdiscoverygateway/water-timeseries-v2:ncsa-water-timeseries

# Same extra dependencies as ../Dockerfile (keep that file and
# envs/requirements-extra.txt in sync).
COPY snakemake/envs/requirements-extra.txt /tmp/requirements-extra.txt
RUN uv pip install -r /tmp/requirements-extra.txt

# rsync for the fs storage plugin (profiles/kubernetes: default-storage-provider).
RUN apt-get update && apt-get install -y --no-install-recommends rsync \
    && rm -rf /var/lib/apt/lists/*

# Separate layer, after the extras, so it can't change what they resolve to.
# pip: every job pod runs `pip install` for the storage plugin before it
# starts, and the uv venv has none of its own.
# The Kubernetes plugin is pinned below 0.5.1: from 0.5.1 on, a failed job pod's
# log is read with previous=True, which Kubernetes rejects (400) for these
# restartPolicy=Never pods, and the exception kills the controller. Lift the pin
# once that's fixed upstream (snakemake/snakemake-executor-plugin-kubernetes,
# check_active_jobs).
RUN uv pip install "snakemake>=9.6,<10" "snakemake-executor-plugin-kubernetes<0.5.1" \
    snakemake-storage-plugin-fs pip

RUN python -c "import geemap; assert hasattr(geemap, 'ee_initialize'), 'ee_initialize missing!'; print(f'geemap {geemap.__version__}')" \
    && snakemake --version

COPY google_cloud_utils/ /app/google_cloud_utils/
COPY near_real_time /app/near_real_time
COPY utils /app/utils
COPY upload_utils /app/upload_utils
COPY historical_run /app/historical_run
COPY snakemake/Snakefile snakemake/config.yaml snakemake/config-kubernetes.yaml /app/snakemake/
COPY snakemake/profiles /app/snakemake/profiles

# Same as the Argo pods set.
ENV PYTHONPATH="/app/.venv/lib/python3.12/site-packages:/app"

# The image's own immutable tag (snakemake-<short sha>), passed in by the build
# (.github/workflows/build_snakemake_image.yml). The controller hands it to
# `snakemake --container-image`, so the job pods run exactly the image the
# controller is running instead of whatever a node cached under a mutable tag.
# Last, so a changing value doesn't invalidate the layers above.
ARG IMAGE_REF=""
ENV SNAKEMAKE_CONTAINER_IMAGE=${IMAGE_REF}

WORKDIR /app

# The executor runs every job as `/bin/sh -c "<snakemake command>"`.
ENTRYPOINT []
CMD ["snakemake", "--help"]
