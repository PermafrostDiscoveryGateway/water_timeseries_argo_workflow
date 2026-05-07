FROM ghcr.io/permafrostdiscoverygateway/water-timeseries-v2:main

# Copy your application code only
COPY google_cloud_utils/ /app/google_cloud_utils/
COPY download/ /app/download/

ENV PYTHONPATH="/app:${PYTHONPATH}"
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["-c", "print('Usage: docker run <script.py> [args...]')"]