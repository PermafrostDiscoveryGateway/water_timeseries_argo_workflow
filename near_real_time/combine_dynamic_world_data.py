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

def combine_new_dynamic_world_data_with_latest(env_path=None):
    # Load environment with fallback logic
    if env_path is None:
        load_dotenv()
        logger.info("Loading environment from default .env file")
    else:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")

    # Get environment variables (now guaranteed to exist after validation)
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    logger.debug(f"dynamic_world_dir: {dynamic_world_dir}")
    split_new_dynamic_world_data_dir = os.environ['split_new_dynamic_world_data_dir']
    logger.debug(f"split_new_dynamic_world_data_dir: {len(split_new_dynamic_world_data_dir)}")
    folders = []
    files = []
    for i in range(0, len(split_new_dynamic_world_data_dir)):
        if os.path.isdir(split_new_dynamic_world_data_dir[i]):
            folders.append(split_new_dynamic_world_data_dir[i])
        else:
            files.append(split_new_dynamic_world_data_dir[i])

    logger.debug(f"Split new dynamic world data folders: {len(folders)} and files {len(files)}")
    for folder in folders:
        logger.debug(f"Folder: {folder}")
    split_contents = os.listdir(split_new_dynamic_world_data_dir)
    # logger.debug(f"Split contents: {split_contents}")

    # get latest dynamic world file
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    logger.debug(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")


    current_split_dir = split_new_dynamic_world_data_dir
    # combine this with the other files
    downloaded_files = glob.glob(os.path.join(current_split_dir, "*.nc"))

    # Merge all downloaded chunks
    logger.info("Merging downloaded chunks...")
    merged_new_ds = None

    for chunk_file in downloaded_files:
        logger.debug(f"Loading chunk: {chunk_file}")
        try:
            ds_chunk = xr.open_dataset(chunk_file)

            if merged_new_ds is None:
                merged_new_ds = ds_chunk
            else:
                # Concatenate along id_geohash dimension
                merged_new_ds = xr.concat([merged_new_ds, ds_chunk], dim="id_geohash")
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_file}: {e}")
            continue

    if merged_new_ds is None:
        logger.error("No data to merge from chunks")
        return None

    # Remove duplicates in id_geohash if any
    _, unique_indices = np.unique(merged_new_ds.id_geohash.values, return_index=True)
    merged_new_ds = merged_new_ds.isel(id_geohash=sorted(unique_indices))
    dates = merged_new_ds.date.values
    logger.debug(f"New dates {dates}")
    latest_date = max(dates)
    logger.debug(f"the latest date is {latest_date}")
    latest_date_string = str(latest_date).split('T')[0].replace('-', '_')
    logger.info(f"The latest date is {latest_date_string}")
    for date in dates:
        print(date, type(date))
    logger.info(f"Merged new data: {len(merged_new_ds.id_geohash)} lakes x {len(merged_new_ds.date)} dates")

    # save the file
    logger.info("Merging existing data with new data...")
    existing_ds = xr.open_dataset(most_recent_dynamic_world_file)

    # Verify no date overlap (should be true based on missing dates logic)
    existing_dates = set(pd.to_datetime(existing_ds.date.values))
    new_dates = set(pd.to_datetime(merged_new_ds.date.values))
    overlapping_dates = existing_dates & new_dates

    if overlapping_dates:
        logger.warning(f"Found {len(overlapping_dates)} overlapping dates: {sorted(overlapping_dates)}")
        logger.warning("Removing overlapping dates from new dataset...")
        # Remove overlapping dates from new dataset
        mask = ~merged_new_ds.date.isin(list(overlapping_dates))
        merged_new_ds = merged_new_ds.sel(date=mask)

    # Concatenate along date dimension
    final_ds = xr.concat([existing_ds, merged_new_ds], dim="date", join="outer")

    # Sort by date
    final_ds = final_ds.sortby("date")

    # Verify the merge was successful
    final_dates = pd.to_datetime(final_ds.date.values)
    logger.info(f"Original dates: {len(existing_dates)}")
    logger.info(f"New dates added: {len(new_dates) - len(overlapping_dates)}")
    logger.info(f"Total dates after merge: {len(final_dates)}")
    logger.info(f"Date range after merge: {final_dates.min()} to {final_dates.max()}")

    # Create the final filename with the latest missing date
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    final_ds.to_netcdf(new_dynamic_world_data_file)

    logger.info(f"Successfully merged datasets!")
    logger.info(f"Original file: {most_recent_dynamic_world_file}")
    logger.info(f"New data chunks: {len(downloaded_files)} files in {current_split_dir}")
    logger.info(f"Merged file saved as: {new_dynamic_world_data_file}")

    # Print summary statistics about lake coverage
    existing_lakes = set(existing_ds.id_geohash.values)
    new_lakes = set(merged_new_ds.id_geohash.values)
    common_lakes = existing_lakes & new_lakes
    only_in_existing = existing_lakes - new_lakes
    only_in_new = new_lakes - existing_lakes

    logger.info(f"Lake coverage summary:")
    logger.info(f"  - Lakes in existing dataset: {len(existing_lakes)}")
    logger.info(f"  - Lakes in new dataset: {len(new_lakes)}")
    logger.info(f"  - Common lakes: {len(common_lakes)}")
    logger.info(f"  - Lakes only in existing: {len(only_in_existing)}")
    logger.info(f"  - Lakes only in new: {len(only_in_new)}")

    # Close datasets to free memory
    existing_ds.close()
    merged_new_ds.close()
    final_ds.close()

if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()