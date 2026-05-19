FROM ghcr.io/permafrostdiscoverygateway/water-timeseries-v2:main

# Install additional Python dependencies using uv (matching the base image)
# Move this BEFORE the verification
COPY requirements.txt /tmp/requirements.txt

# Use uv to install packages (ensures they go to the right environment)
RUN uv pip install --system \
    python-dotenv \
    google-cloud-storage \
    google-api-core \
    toml

# Copy your application code
COPY google_cloud_utils/ /app/google_cloud_utils/
COPY near_real_time /app/near_real_time

# Add /app to Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set working directory
WORKDIR /app

# Verify installations (now AFTER the pip install)
RUN python -c "import dotenv; print('✅ dotenv installed successfully')" && \
    python -c "import google.cloud.storage; print('✅ google-cloud-storage installed successfully')" && \
    python -c "import toml; print('✅ toml installed successfully')"

# Entrypoint
ENTRYPOINT ["python"]
CMD ["-c", "print('Usage: docker run <script.py> [args...]')"]