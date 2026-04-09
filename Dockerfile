# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies (optional, for gcloud CLI if needed)
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both scripts to container
COPY upload_to_gcs.py download_from_gcs.py ./

# Make scripts executable
RUN chmod +x upload_to_gcs.py download_from_gcs.py