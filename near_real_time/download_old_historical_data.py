import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
from loguru import logger
import os
import numpy as np
import glob
import xarray as xr
from dotenv import load_dotenv
from water_timeseries.downloader import EarthEngineDownloader


def get_new_lake_ids(old_file_path, new_file_or_dir_path):
    """
    Find lake_ids that are in new data but not in old data.

    Parameters:
    -----------
    old_file_path : str
        Path to the existing/old NetCDF file
    new_file_or_dir_path : str
        Either a single NetCDF file or a directory containing chunk NetCDF files

    Returns:
    --------
    set
        Set of lake_ids that are in new data but not in old data
    """

    # Load old file
    print(f"Loading old file: {old_file_path}")
    old_ds = xr.open_dataset(old_file_path, decode_times=False)
    old_lake_ids = set(old_ds.id_geohash.values)
    print(f"Old file has {len(old_lake_ids):,} unique lake_ids")
    old_ds.close()

    # Load new data (either single file or directory of chunks)
    new_lake_ids = set()

    if os.path.isfile(new_file_or_dir_path):
        # Single file
        print(f"Loading new file: {new_file_or_dir_path}")
        new_ds = xr.open_dataset(new_file_or_dir_path, decode_times=False)
        new_lake_ids = set(new_ds.id_geohash.values)
        new_ds.close()

    elif os.path.isdir(new_file_or_dir_path):
        # Directory of chunk files
        chunk_files = sorted(glob.glob(os.path.join(new_file_or_dir_path, "*.nc")))
        print(f"Found {len(chunk_files)} chunk files in directory: {new_file_or_dir_path}")

        for i, chunk_file in enumerate(chunk_files):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(chunk_files)} chunks, "
                      f"unique lakes so far: {len(new_lake_ids):,}")

            try:
                with xr.open_dataset(chunk_file, decode_times=False) as ds:
                    for lake_id in ds.id_geohash.values:
                        new_lake_ids.add(lake_id)
            except Exception as e:
                print(f"Warning: Error reading {os.path.basename(chunk_file)}: {e}")

    else:
        raise ValueError(f"Path does not exist or is not a file/directory: {new_file_or_dir_path}")

    print(f"New data has {len(new_lake_ids):,} unique lake_ids")

    # Find lakes that are only in new data
    only_new_lake_ids = new_lake_ids - old_lake_ids
    print(f"\nFound {len(only_new_lake_ids):,} lake_ids that are in new data but not in old data")

    return only_new_lake_ids

if __name__ == "__main__":

    load_dotenv()

    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    earliest_dynamic_world_file = min(all_dynamic_world_files, key=os.path.getctime)

    logger.debug(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")
    logger.debug(f"First dynamic world file: {earliest_dynamic_world_file}")

    new_lake_ids = get_new_lake_ids(earliest_dynamic_world_file, most_recent_dynamic_world_file)
    logger.debug(f"New lake_ids: {len(new_lake_ids)} exist in new file")

    earlier_years = list(range(2016, 2025))
    months = [6,7,8,9]

    print(earlier_years)
    print(months)

