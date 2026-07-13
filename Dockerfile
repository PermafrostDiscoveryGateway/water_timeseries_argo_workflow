FROM python:3.12-slim

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install water-timeseries directly from GitHub (main branch)
RUN uv pip install --system git+https://github.com/permafrostdiscoverygateway/water-timeseries-v2.git@main

# Install additional Python dependencies using uv
RUN uv pip install --system \
    python-dotenv \
    google-cloud-storage \
    google-api-core \
    toml \
    dask[dataframe] \
    pyarrow \
    geemap==0.37.2

# Copy your application code
COPY google_cloud_utils/ /app/google_cloud_utils/
COPY near_real_time /app/near_real_time
COPY utils /app/utils
COPY upload_utils /app/upload_utils
COPY historical_run /app/historical_run

# Add /app to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Verify geemap version after install
RUN python -c "import geemap; print(f'geemap version after install: {geemap.__version__}'); print(f'Has ee_initialize: {hasattr(geemap, \"ee_initialize\")}')" || echo "geemap verification failed"

RUN python -c "import geemap; print(f'✅ geemap version: {geemap.__version__}'); assert hasattr(geemap, 'ee_initialize'), 'ee_initialize missing!'; print('✅ ee_initialize exists!')"

# Verify water-timeseries has absolute values
RUN python -c "from water_timeseries.breakpoint import NRTBreakpoint; assert 'water_observed_absolute' in NRTBreakpoint.output_columns, '❌ water-timeseries is missing absolute values!'; print('✅ water-timeseries has absolute values!'); print(f'   Output columns: {len(NRTBreakpoint.output_columns)} columns')"

# Set working directory
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["-c", "print('Usage: docker run <script.py> [args...]')"]