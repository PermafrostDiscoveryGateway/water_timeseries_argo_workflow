FROM ghcr.io/permafrostdiscoverygateway/water-timeseries-v2:151-error-on-running-historical-data

# Install additional Python dependencies using uv
# Pin geemap to prevent it from upgrading
RUN uv pip install \
    python-dotenv \
    google-cloud-storage \
    google-api-core \
    toml \
    dask[dataframe] \
    pyarrow \
    geemap==0.37.2

# Alternatively, prevent upgrading any packages that are already installed:
# RUN uv pip install --no-deps \  # This would skip dependencies but might break things
#     python-dotenv \
#     google-cloud-storage \
#     ...

# Copy your application code
COPY google_cloud_utils/ /app/google_cloud_utils/
COPY near_real_time /app/near_real_time
COPY utils /app/utils
COPY historical_run /app/historical_run

# Add /app to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Verify geemap version after install
RUN python -c "import geemap; print(f'geemap version after install: {geemap.__version__}'); print(f'Has ee_initialize: {hasattr(geemap, \"ee_initialize\")}')" || echo "geemap verification failed"

RUN python -c "import geemap; print(f'✅ geemap version: {geemap.__version__}'); assert hasattr(geemap, 'ee_initialize'), 'ee_initialize missing!'; print('✅ ee_initialize exists!')"

# Set working directory
WORKDIR /app

# No verification step - just run
ENTRYPOINT ["python"]
CMD ["-c", "print('Usage: docker run <script.py> [args...]')"]