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
from dask.diagnostics import ProgressBar
from dotenv import load_dotenv
from dask.distributed import Client, LocalCluster


def combine_new_dynamic_world_data_with_latest(env_path=None):
    # Load environment
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
    split_new_dynamic_world_data_dir = os.environ['split_new_dynamic_world_data_dir']

    # Setup Dask client
    logger.info("Setting up Dask client for parallel processing...")
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB')
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")

    # Find latest valid file
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    valid_files = [f for f in all_dynamic_world_files if os.path.getsize(f) > 1024 * 1024]

    if not valid_files:
        logger.error("No valid existing dynamic world files found!")
        client.close()
        return None

    most_recent_dynamic_world_file = max(valid_files, key=os.path.getctime)
    logger.info(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")

    # Get downloaded chunk files
    downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
    logger.info(f"Found {len(downloaded_files)} chunk files")

    # Use open_mfdataset instead of manual concatenation (much more efficient)
    logger.info("Reading and combining new data chunks with Dask...")
    merged_new_ds = xr.open_mfdataset(
        downloaded_files,
        concat_dim="id_geohash",
        combine="nested",
        chunks={'id_geohash': 100000, 'date': -1},
        parallel=True
    )

    # Get dates (small data, safe to compute)
    dates = merged_new_ds.date.values
    if hasattr(dates, 'compute'):
        dates = dates.compute()

    latest_date = max(dates)
    try:
        latest_date_string = str(latest_date).split('T')[0].replace('-', '_')
    except:
        latest_date_string = str(latest_date).replace('-', '_')

    logger.info(f"Latest date: {latest_date_string}")
    logger.info(f"New data shape: {merged_new_ds.dims}")

    # Load existing dataset with Dask (lazy)
    logger.info("Loading existing dataset with Dask...")
    existing_ds = xr.open_dataset(
        most_recent_dynamic_world_file,
        chunks={'id_geohash': 100000, 'date': -1}
    )

    # Get small metadata without computing large arrays
    existing_dates = existing_ds.date.values
    if hasattr(existing_dates, 'compute'):
        existing_dates = existing_dates.compute()

    # For lake IDs, we need them for comparison, but we can compute them efficiently
    # Use pandas for faster set operations on computed data
    logger.info("Comparing lake IDs...")
    existing_lake_ids = existing_ds.id_geohash.values
    if hasattr(existing_lake_ids, 'compute'):
        existing_lake_ids = existing_lake_ids.compute()
    existing_lake_ids_set = set(str(x) for x in existing_lake_ids)

    new_lake_ids = merged_new_ds.id_geohash.values
    if hasattr(new_lake_ids, 'compute'):
        new_lake_ids = new_lake_ids.compute()
    new_lake_ids_set = set(str(x) for x in new_lake_ids)

    lakes_only_in_new = new_lake_ids_set - existing_lake_ids_set
    common_lakes = existing_lake_ids_set & new_lake_ids_set

    logger.info("=" * 60)
    logger.info("LAKE COMPARISON SUMMARY:")
    logger.info(f"  Existing lakes: {len(existing_lake_ids_set):,}")
    logger.info(f"  New lakes: {len(new_lake_ids_set):,}")
    logger.info(f"  Common lakes: {len(common_lakes):,}")
    logger.info(f"  New lakes to add: {len(lakes_only_in_new):,}")

    # Check date overlap
    new_dates_set = set(pd.to_datetime(dates))
    existing_dates_set = set(pd.to_datetime(existing_dates))
    overlapping_dates = existing_dates_set & new_dates_set

    if overlapping_dates:
        logger.warning(f"Removing {len(overlapping_dates)} overlapping dates")
        # Use boolean indexing with Dask (lazy)
        mask = ~merged_new_ds.date.isin(list(overlapping_dates))
        merged_new_ds = merged_new_ds.where(mask, drop=True)
        dates = merged_new_ds.date.values
        if hasattr(dates, 'compute'):
            dates = dates.compute()

    if len(dates) == 0:
        logger.warning("No new dates to add")
        client.close()
        return None

    logger.info(f"Adding {len(dates)} new dates: {dates}")

    # LAZY MERGE - no computation happens here
    logger.info("=" * 60)
    logger.info("MERGING DATASETS (lazy operation)...")

    # Concatenate along date dimension (both datasets are still lazy)
    final_ds = xr.concat([existing_ds, merged_new_ds], dim="date", join="outer")
    final_ds = final_ds.sortby("date")

    logger.info(f"Final dataset shape (lazy): {final_ds.dims}")

    # Save the result (this triggers the actual computation)
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    logger.info(f"Saving to: {new_dynamic_world_data_file}")

    # Use encoding for compression
    encoding = {var_name: {'zlib': True, 'complevel': 5}
                for var_name in final_ds.data_vars if var_name not in ['date', 'id_geohash']}

    # This is where the actual computation happens
    logger.info("Writing file (this will take time - check Dask dashboard)...")
    with ProgressBar():
        final_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding, compute=True)

    # Get final stats (after computation)
    final_lake_count = len(final_ds.id_geohash)
    final_date_count = len(final_ds.date)

    logger.info("=" * 60)
    logger.info(f"✓ MERGE COMPLETE!")
    logger.info(f"  Output: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {final_lake_count:,}")
    logger.info(f"  Total dates: {final_date_count}")
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