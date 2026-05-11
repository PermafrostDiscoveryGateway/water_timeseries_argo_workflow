#!/bin/bash

# Set paths
VENV_PATH="/home/ext_tcnichol_illinois_edu/water-timeseries-v2/.venv/bin/activate"
SCRIPT_PATH="/home/ext_tcnichol_illinois_edu/water_timeseries_argo_workflow/download/download_dynamic_world_split_parquet.py"
ENV_PATH="/home/ext_tcnichol_illinois_edu/water_timeseries_argo_workflow/download/.env2"

# Check if virtual environment exists
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi

# Check if .env file exists
if [ ! -f "$ENV_PATH" ]; then
    echo "Warning: .env file not found at $ENV_PATH"
    echo "Continuing without custom .env (will use default)"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH"

# Run python script with env path
echo "Running download script with env: $ENV_PATH"
python "$SCRIPT_PATH" "$ENV_PATH"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
else
    echo "Download failed with exit code $?"
    exit 1
fi