FROM ghcr.io/permafrostdiscoverygateway/water-timeseries-v2:main

# Install additional Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt

# Copy google_cloud_utils folder and everything in it to /app/google_cloud_utils
COPY google_cloud_utils/ /app/google_cloud_utils/

# Copy utils folder and everything in it to /app/utils
COPY download/ /app/download/

# Add /app to Python path so modules can be imported
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set working directory
WORKDIR /app

# Entrypoint - allows running any Python script with parameters
ENTRYPOINT ["python"]

# Default command (shows help if no script specified)
CMD ["-c", "print('Usage: docker run <image> <script.py> [args...]')"]