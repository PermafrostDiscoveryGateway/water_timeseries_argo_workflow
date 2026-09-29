# Image for running the snakemake test pipeline on Kubernetes: the same
# environment as the Argo image (../Dockerfile) plus snakemake and its
# Kubernetes executor, and the workflow files. The controller Job and every
# job pod it starts run this one image.
#
# Build from the repo root:
#   docker build -f snakemake/Dockerfile -t <image> .
FROM ghcr.io/permafrostdiscoverygateway/water-timeseries-v2:ncsa-water-timeseries

# Same extra dependencies as ../Dockerfile (keep that file and
# envs/requirements-extra.txt in sync).
COPY snakemake/envs/requirements-extra.txt /tmp/requirements-extra.txt
RUN uv pip install -r /tmp/requirements-extra.txt

# Separate layer, after the extras, so it can't change what they resolve to.
RUN uv pip install "snakemake>=9.6,<10" snakemake-executor-plugin-kubernetes

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

WORKDIR /app

# The executor runs every job as `/bin/sh -c "<snakemake command>"`.
ENTRYPOINT []
CMD ["snakemake", "--help"]
