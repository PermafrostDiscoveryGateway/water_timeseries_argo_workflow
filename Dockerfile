FROM ghcr.io/permafrostdiscoverygateway/water-timeseries-v2:main

WORKDIR /app

# Install into the project venv (/app/.venv), which is first on PATH — not --system.
RUN uv pip install \
    python-dotenv \
    google-cloud-storage \
    google-api-core \
    toml \
    dask[dataframe] \
    pyarrow \
    pandas

RUN python -c "from dotenv import load_dotenv; import dask.dataframe, pyarrow"

# Copy your application code
COPY google_cloud_utils/ /app/google_cloud_utils/
COPY near_real_time /app/near_real_time

ENV PYTHONPATH="/app"

ENTRYPOINT ["python"]
CMD ["-c", "print('Usage: docker run <script.py> [args...]')"]
