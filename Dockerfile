FROM ghcr.io/permafrostdiscoverygateway/water-timeseries-v2:main

# Install additional Python dependencies using uv
RUN uv pip install --system \
    python-dotenv \
    google-cloud-storage \
    google-api-core \
    toml \
    dask[dataframe] \
    pyarrow \
    pandas

# Install sudo (needed for the final installation attempt in Argo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY google_cloud_utils/ /app/google_cloud_utils/
COPY near_real_time /app/near_real_time

# Add /app to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set working directory
WORKDIR /app

# No verification step - just run
ENTRYPOINT ["python"]
CMD ["-c", "print('Usage: docker run <script.py> [args...]')"]