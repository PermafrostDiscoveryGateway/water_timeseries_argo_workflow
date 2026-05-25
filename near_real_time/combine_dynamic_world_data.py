import netCDF4 as nc
import pandas as pd
from netCDF4 import num2date
from datetime import datetime
from loguru import logger
import os
import numpy as np
import glob
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv

# Configure Dask for better performance
from dask.distributed import Client, LocalCluster


def combine_new_dynamic_world_data_with_latest(env_path=None):
    # Load environment with fallback logic
    if env_path is None:
        load_dotenv()
        logger.info("Loading environment from default .env file")
    else:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")

    # Get environment variables
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    logger.debug(f"dynamic_world_dir: {dynamic_world_dir}")
    split_new_dynamic_world_data_dir = os.environ['split_new_dynamic_world_data_dir']
    logger.debug(f"split_new_dynamic_world_data_dir: {split_new_dynamic_world_data_dir}")

    # Setup Dask client for parallel processing
    logger.info("Setting up Dask client for parallel processing...")
    # Adjust memory limit based on your system (e.g., 32GB)
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB')
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")

    # Find the latest valid dynamic world file
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    valid_files = []
    for f in all_dynamic_world_files:
        if os.path.getsize(f) > 1024 * 1024:  # Larger than 1 MB
            valid_files.append(f)
        else:
            logger.warning(f"Skipping empty/corrupted file: {f}")

    if not valid_files:
        logger.error("No valid existing dynamic world files found!")
        client.close()
        return None

    most_recent_dynamic_world_file = max(valid_files, key=os.path.getctime)
    logger.info(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")

    # Get downloaded chunk files
    downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
    logger.info(f"Found {len(downloaded_files)} chunk files")

    # Use xarray with Dask to read and combine chunks
    logger.info("Reading and combining new data chunks with Dask...")

    # Open all chunks as Dask-backed datasets
    chunks = []
    for chunk_file in downloaded_files:
        try:
            # Open with chunks for parallel processing
            ds_chunk = xr.open_dataset(chunk_file, chunks={'id_geohash': 100000})
            chunks.append(ds_chunk)
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_file}: {e}")
            continue

    if not chunks:
        logger.error("No data to merge from chunks")
        client.close()
        return None

    # Concatenate all chunks along id_geohash
    logger.info("Concatenating chunks with Dask...")
    merged_new_ds = xr.concat(chunks, dim="id_geohash")

    # Remove duplicates if any - need to compute the values first
    logger.info("Removing duplicate lake IDs...")
    # Get the values (this triggers computation)
    id_geohash_values = merged_new_ds.id_geohash.values
    # If it's a Dask array, compute it; if it's already numpy, use directly
    if hasattr(id_geohash_values, 'compute'):
        id_geohash_values = id_geohash_values.compute()

    _, unique_indices = np.unique(id_geohash_values, return_index=True)
    merged_new_ds = merged_new_ds.isel(id_geohash=sorted(unique_indices))

    # Get dates (these are likely small, so compute is fine)
    dates = merged_new_ds.date.values
    if hasattr(dates, 'compute'):
        dates = dates.compute()

    latest_date = max(dates)
    try:
        latest_date_string = str(latest_date).split('T')[0].replace('-', '_')
    except:
        latest_date_string = str(latest_date).replace('-', '_')

    logger.info(f"The latest date is {latest_date_string}")
    logger.info(f"Merged new data: {len(merged_new_ds.id_geohash):,} lakes x {len(dates)} dates")

    # Load existing dataset with Dask
    logger.info("Loading existing dataset with Dask...")
    existing_ds = xr.open_dataset(most_recent_dynamic_world_file, chunks={'id_geohash': 100000, 'date': -1})

    # Get existing dates and lakes (compute only the necessary metadata)
    existing_date_values = existing_ds.date.values
    if hasattr(existing_date_values, 'compute'):
        existing_date_values = existing_date_values.compute()
    existing_dates = existing_date_values

    existing_lake_values = existing_ds.id_geohash.values
    if hasattr(existing_lake_values, 'compute'):
        existing_lake_values = existing_lake_values.compute()
    existing_lake_ids = existing_lake_values
    existing_lake_ids_set = set(str(lake_id) for lake_id in existing_lake_ids)

    # Get new lake IDs
    new_lake_values = merged_new_ds.id_geohash.values
    if hasattr(new_lake_values, 'compute'):
        new_lake_values = new_lake_values.compute()
    new_lake_ids = new_lake_values
    new_lake_ids_set = set(str(lake_id) for lake_id in new_lake_ids)

    # Find new lakes
    lakes_only_in_new = new_lake_ids_set - existing_lake_ids_set
    lakes_only_in_existing = existing_lake_ids_set - new_lake_ids_set
    common_lakes = existing_lake_ids_set & new_lake_ids_set

    logger.info("=" * 60)
    logger.info("LAKE COMPARISON SUMMARY:")
    logger.info(f"  Existing lakes: {len(existing_lake_ids_set):,}")
    logger.info(f"  New lakes: {len(new_lake_ids_set):,}")
    logger.info(f"  Common lakes: {len(common_lakes):,}")
    logger.info(f"  New lakes to add: {len(lakes_only_in_new):,}")

    # Check for date overlap
    new_dates_set = set(pd.to_datetime(dates))
    existing_dates_set = set(pd.to_datetime(existing_dates))
    overlapping_dates = existing_dates_set & new_dates_set

    if overlapping_dates:
        logger.warning(f"Removing {len(overlapping_dates)} overlapping dates")
        mask = ~merged_new_ds.date.isin(list(overlapping_dates))
        merged_new_ds = merged_new_ds.sel(date=mask)
        dates = merged_new_ds.date.values
        if hasattr(dates, 'compute'):
            dates = dates.compute()

    if len(dates) == 0:
        logger.warning("No new dates to add")
        client.close()
        return None

    logger.info(f"Adding {len(dates)} new dates: {dates}")

    # FAST MERGE using Dask
    logger.info("=" * 60)
    logger.info("MERGING DATASETS WITH DASK...")

    # SIMPLIFIED APPROACH: Use xarray's built-in concat which works with Dask
    logger.info("Concatenating along date dimension with Dask...")

    # Combine existing and new data directly
    final_ds = xr.concat([existing_ds, merged_new_ds], dim="date", join="outer")

    # Sort by date
    final_ds = final_ds.sortby("date")

    # Note: If there are new lakes, they will automatically be handled by join='outer'

    logger.info(f"Final dataset size: {len(final_ds.id_geohash):,} lakes x {len(final_ds.date)} dates")

    # Save with Dask
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    logger.info(f"Saving merged dataset to: {new_dynamic_world_data_file}")

    # Use encoding for compression
    encoding = {var_name: {'zlib': True, 'complevel': 5}
                for var_name in final_ds.data_vars if var_name not in ['date', 'id_geohash']}

    # Save with Dask's parallel write capability
    logger.info("Writing file (this may take a while)...")
    with ProgressBar():
        final_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding, compute=True)

    logger.info("=" * 60)
    logger.info(f"✓ SUCCESSFULLY MERGED DATASETS WITH DASK!")
    logger.info(f"  Original: {most_recent_dynamic_world_file}")
    logger.info(f"  New file: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {len(final_ds.id_geohash):,}")
    logger.info(f"  Total dates: {len(final_ds.date)}")
    logger.info(f"  New lakes added: {len(lakes_only_in_new):,}")
    logger.info(f"  New dates added: {len(dates)}")
    logger.info("=" * 60)

    # Clean up
    client.close()
    existing_ds.close()
    merged_new_ds.close()
    final_ds.close()


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()