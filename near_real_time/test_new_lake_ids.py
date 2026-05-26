from loguru import logger
from dotenv import load_dotenv
from water_timeseries.downloader import EarthEngineDownloader
from datetime import datetime


def get_date_range_manual():
    start_date = datetime(2015, 6, 1)
    end_date = datetime(2024, 9, 1)

    dates = []
    current_date = start_date

    while current_date <= end_date:
        # Check if month is in [6, 7, 8, 9] (June, July, August, September)
        if current_date.month in [6, 7, 8, 9]:
            dates.append(current_date)

        # Move to next month (first day of next month)
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return dates


import pandas as pd
import xarray as xr
import glob
import os
from pathlib import Path


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


def save_lake_ids_to_file(lake_ids, output_path):
    """Save lake_ids to a text file, one per line."""
    with open(output_path, 'w') as f:
        for lake_id in sorted(lake_ids):
            f.write(f"{lake_id}\n")
    print(f"Saved {len(lake_ids)} lake_ids to {output_path}")


# Example usage
if __name__ == "__main__":
    load_dotenv()
    # Option 1: Specify paths directly
    old_file = "/Users/helium/Desktop/dynanic_world/lakes_dw_V2d.nc"
    new_data = "/Users/helium/Desktop/dynanic_world/lakes_dw_V2d_2025_09_01.nc"  # or a single file

    output_file = "new_lake_ids.txt"
    if os.path.isfile(output_file):
        print(f"File already exists: {output_file}")
    else:
        # Get the new lake IDs
        new_lake_ids = get_new_lake_ids(old_file, new_data)

        # Print first 10 as sample
        if new_lake_ids:
            print("\nSample of new lake_ids (first 10):")
            for lake_id in sorted(new_lake_ids)[:10]:
                print(f"  {lake_id}")
        # Optionally save to file
        save_lake_ids_to_file(new_lake_ids, output_file)

    split_vector_dataset_file = os.environ["vector_lake_file"]
    # get new lake_ids
    new_lake_ids = []
    with open(output_file, 'r') as f:
        lake_ids = f.readlines()
        logger.debug(f"Found {len(lake_ids):,} lake_ids")
        for line in lake_ids:
            current_lake_id = line.rstrip('\n')
            new_lake_ids.append(current_lake_id)
    previous_dates = get_date_range_manual()

    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID

    for date in previous_dates:
        current_year = date.year
        current_month = date.month
        print(f"Current year: {current_year} and month: {current_month}")
        print(f"What would we process? Coming next....")
        dl = EarthEngineDownloader(ee_auth=True, logger=logger)
        ds = dl.download_dw_monthly(
            vector_dataset=split_vector_dataset_file,
            id_list=new_lake_ids,
            name_attribute="id_geohash",
            years=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            months=[current_month],
            max_total_requests=500,
            n_parallel=1,
            no_download=True
        )

    # Option 2: If you want to use with your existing environment variables
    # from dotenv import load_dotenv
    # import os
    # load_dotenv()
    #
    # dynamic_world_dir = os.environ['dynamic_world_dir']
    # split_new_dynamic_world_data_dir = os.environ['split_new_dynamic_world_data_dir']
    #
    # # Find latest existing file
    # import glob
    # existing_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    # latest_existing = max(existing_files, key=os.path.getctime)
    #
    # new_lake_ids = get_new_lake_ids(latest_existing, split_new_dynamic_world_data_dir)

