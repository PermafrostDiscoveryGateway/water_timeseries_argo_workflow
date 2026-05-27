import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
from loguru import logger
import os
import gc
import numpy as np
import glob
import xarray as xr
import geopandas as gpd
from dotenv import load_dotenv
from water_timeseries.downloader import EarthEngineDownloader
from water_timeseries.utils.io import load_vector_dataset


def get_new_lake_ids_streaming(old_file_path, new_dir_path, chunk_size=100000):
    """
    Memory-efficient version that streams through new files and yields
    new lake_ids in batches without storing all of them at once.
    """

    # Load old lake IDs (this is necessary but only ~hundreds of MB)
    print(f"Loading old file: {old_file_path}")
    with xr.open_dataset(old_file_path, decode_times=False) as old_ds:
        old_lake_ids = set(old_ds.id_geohash.values)
    print(f"Old file has {len(old_lake_ids):,} unique lake_ids")

    # Process new files
    chunk_files = sorted(glob.glob(os.path.join(new_dir_path, "*.nc")))
    print(f"Found {len(chunk_files)} chunk files")

    new_batch = []
    total_new = 0

    for i, chunk_file in enumerate(chunk_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(chunk_files)} chunks, "
                  f"new lakes found so far: {total_new:,}")

        try:
            with xr.open_dataset(chunk_file, decode_times=False) as ds:
                for lake_id in ds.id_geohash.values:
                    if lake_id not in old_lake_ids:
                        new_batch.append(lake_id)
                        total_new += 1

                        if len(new_batch) >= chunk_size:
                            yield new_batch
                            new_batch = []

        except Exception as e:
            print(f"Warning: Error reading {os.path.basename(chunk_file)}: {e}")

    if new_batch:
        yield new_batch

    print(f"Total new lake_ids found: {total_new:,}")


def check_existing_files(missing_historical_data_dir, chunk_num, years, months):
    """Check which year-month combinations already have files for a specific chunk."""
    files_to_download = []

    for year in years:
        for month in months:
            current_filename = f'historical_{year}_{month}_chunk_{chunk_num}.nc'
            current_filepath = os.path.join(missing_historical_data_dir, current_filename)

            if os.path.isfile(current_filepath):
                file_size = os.path.getsize(current_filepath)
                if file_size > 0:
                    print(f"  ✓ Skipping {current_filename} - already exists ({file_size / 1024 / 1024:.1f} MB)")
                else:
                    print(f"  ⚠ Warning: {current_filename} exists but is empty, will re-download")
                    files_to_download.append((year, month, current_filepath))
            else:
                print(f"  ✗ Need to download: {current_filename}")
                files_to_download.append((year, month, current_filepath))

    return files_to_download


if __name__ == "__main__":

    load_dotenv()

    vector_file = os.environ['vector_lake_file']
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID

    dynamic_world_dir = os.environ['dynamic_world_dir']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    earliest_dynamic_world_file = min(all_dynamic_world_files, key=os.path.getctime)

    missing_historical_data_dir = os.environ['missing_historical_data_dir']
    logger.debug(f"Missing historical data directory: {missing_historical_data_dir}")
    if not os.path.isdir(missing_historical_data_dir):
        os.makedirs(missing_historical_data_dir, exist_ok=True)

    # ===== CRITICAL: Load vector file ONCE with geometries =====
    print("Loading full vector file with geometries...")
    full_gdf = load_vector_dataset(vector_file, logger=logger)
    print(f"Loaded {len(full_gdf):,} total geometries")

    CHUNK_SIZE = 5000  # Adjust based on available RAM
    earlier_years = list(range(2016, 2025))
    months = [6, 7, 8, 9]

    print(f"Processing in chunks of {CHUNK_SIZE:,} lake_ids")
    print(f"Years to process: {earlier_years}")
    print(f"Months to process: {months}")

    # TODO get new lake ids here

    # Filter to new lakes
    print("Filtering to target lakes...")
    target_gdf = full_gdf[full_gdf['id_geohash'].isin(new_lake_ids)].copy()

    # Optional: Clear full_gdf from memory to save space
    del full_gdf
    import gc

    gc.collect()

    print(f"Ready to download for {len(target_gdf)} lakes")