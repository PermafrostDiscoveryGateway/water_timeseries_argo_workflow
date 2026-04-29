import argparse
import os
import toml
from loguru import logger
from water_timeseries.downloader import EarthEngineDownloader
import water_timeseries

import xarray as xr
import netCDF4 as nc
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Tuple, Dict, Any
from pathlib import Path
import zarr
import nest_asyncio
import sys
import asyncio
from check_zarr_dataset import get_zarr_dates
# Set the event loop policy for better compatibility
if sys.platform == 'darwin':  # macOS
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
else:  # Linux (Kubernetes)
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Only use nest_asyncio if you're in a nested environment
try:
    loop = asyncio.get_running_loop()
    nest_asyncio.apply()  # Only if already running in a loop
except RuntimeError:
    pass

def load_config(config_path="/app/config/config.toml"):
    """Load configuration from TOML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = toml.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Near Real Time Run")
    parser.add_argument("--config", help="Path to config file", default="/app/config/config.toml")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = toml.load(f)
        ee_project = config.get("ee", {}).get("project", "pdg-project-406720")
        final_ee_project = ee_project
        vector_dataset = config.get("ee", {}).get("vector_dataset", "")
        final_vector_dataset = vector_dataset
        dynamic_world_dir = config.get("dynamic_world", {}).get("base_path", "")
        file_format = config.get("dynamic_world", {}).get("format", "")
        output_dir = config.get("output", {}).get("output_path", "")
    print('we got values from config')

    existing_zarr_datasets = [os.path.join(dynamic_world_dir, d) for d in os.listdir(dynamic_world_dir)]

    # Get the most recently modified dataset
    most_recent_zarr_dataset = max(existing_zarr_datasets, key=os.path.getmtime)

    def is_valid_zarr(zarr_path):
        """Check if a Zarr dataset is valid and readable"""
        try:
            # Try to open with pure zarr first (less overhead than xarray)
            root = zarr.open_group(zarr_path, mode='r')
            # Try to read a small piece of data
            for key in root.array_keys():
                arr = root[key]
                # Try to read just the first element
                if arr.size > 0:
                    _ = arr[0] if arr.ndim == 1 else arr[0, ...]
            return True
        except Exception as e:
            print(f"Invalid Zarr dataset: {e}")
            return False

    def diagnose_zarr(zarr_path):
        """Quick diagnostic for your specific dataset"""
        zarr_path = Path(zarr_path)

        print(f"Checking: {zarr_path}")
        print(f"Exists: {zarr_path.exists()}")

        if zarr_path.exists():
            # Check size and contents
            total_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
            print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

            # List contents
            print("\nContents:")
            for item in zarr_path.iterdir():
                if item.is_dir():
                    num_files = len(list(item.rglob('*')))
                    print(f"  📁 {item.name}/ ({num_files} files)")
                else:
                    size = item.stat().st_size
                    print(f"  📄 {item.name} ({size} bytes)")

            # Check for .zgroup (essential)
            zgroup_file = zarr_path / '.zgroup'
            print(f"\n.zgroup exists: {zgroup_file.exists()}")
            if zgroup_file.exists():
                with open(zgroup_file, 'r') as f:
                    print(f"Content: {f.read()[:200]}")

    diagnose_zarr(most_recent_zarr_dataset)

    most_recent_zarr_dates = get_zarr_dates(most_recent_zarr_dataset)
    valid = is_valid_zarr(most_recent_zarr_dataset)
    print('we got the dates')

    print(existing_zarr_datasets)

if __name__ == '__main__':
    main()