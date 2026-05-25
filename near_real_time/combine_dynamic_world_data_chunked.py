import pandas as pd
from datetime import datetime
from loguru import logger
import os
import numpy as np
import glob
import xarray as xr
from dotenv import load_dotenv


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

    # Find latest valid file
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    valid_files = []
    for f in all_dynamic_world_files:
        if os.path.getsize(f) > 1024 * 1024:
            valid_files.append(f)
        else:
            logger.warning(f"Skipping empty/corrupted file: {os.path.basename(f)}")

    if not valid_files:
        logger.error("No valid existing dynamic world files found!")
        return None

    most_recent_dynamic_world_file = max(valid_files, key=os.path.getctime)
    logger.info(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")

    # Get downloaded chunk files
    downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
    logger.info(f"Found {len(downloaded_files)} chunk files")

    # Validate chunk files
    valid_chunks = []
    for chunk_file in downloaded_files:
        try:
            with xr.open_dataset(chunk_file) as test:
                test.close()
            valid_chunks.append(chunk_file)
        except Exception as e:
            logger.warning(f"Skipping corrupted chunk: {os.path.basename(chunk_file)}")

    logger.info(f"Valid chunks: {len(valid_chunks)} out of {len(downloaded_files)}")

    if not valid_chunks:
        logger.error("No valid chunk files found!")
        return None

    # Read and combine new data chunks
    logger.info("Reading and combining new data chunks...")
    merged_new_ds = None

    for chunk_file in valid_chunks:
        try:
            ds_chunk = xr.open_dataset(chunk_file)
            if merged_new_ds is None:
                merged_new_ds = ds_chunk
            else:
                merged_new_ds = xr.concat([merged_new_ds, ds_chunk], dim="id_geohash")
        except Exception as e:
            logger.error(f"Failed to load chunk {os.path.basename(chunk_file)}: {e}")
            continue

    if merged_new_ds is None:
        logger.error("No data to merge from chunks")
        return None

    # Remove duplicates
    logger.info("Removing duplicate lake IDs...")
    id_geohash_values = merged_new_ds.id_geohash.values
    _, unique_indices = np.unique(id_geohash_values, return_index=True)
    merged_new_ds = merged_new_ds.isel(id_geohash=sorted(unique_indices))

    # Get dates
    dates = merged_new_ds.date.values
    latest_date = max(dates)
    try:
        latest_date_string = str(latest_date).split('T')[0].replace('-', '_')
    except:
        latest_date_string = str(latest_date).replace('-', '_')

    logger.info(f"Latest date: {latest_date_string}")
    logger.info(f"New data: {len(merged_new_ds.id_geohash):,} lakes x {len(dates)} dates")

    # Load existing dataset with chunking
    logger.info("Loading existing dataset with chunking...")
    existing_ds = xr.open_dataset(
        most_recent_dynamic_world_file,
        chunks={'id_geohash': 50000, 'date': -1}  # Process 50k lakes at a time
    )

    # Get existing dates
    existing_dates = existing_ds.date.values
    logger.info(f"Existing dates: {len(existing_dates)} dates")

    # Remove overlapping dates
    new_dates_set = set(pd.to_datetime(dates))
    existing_dates_set = set(pd.to_datetime(existing_dates))
    overlapping_dates = existing_dates_set & new_dates_set

    if overlapping_dates:
        logger.warning(f"Removing {len(overlapping_dates)} overlapping dates")
        mask = ~merged_new_ds.date.isin(list(overlapping_dates))
        merged_new_ds = merged_new_ds.sel(date=mask)
        dates = merged_new_ds.date.values

    if len(dates) == 0:
        logger.warning("No new dates to add")
        existing_ds.close()
        merged_new_ds.close()
        return None

    logger.info(f"Adding {len(dates)} new dates")

    # Merge datasets (lazy operation - no memory load yet)
    logger.info("Merging datasets (lazy operation)...")
    final_ds = xr.concat([existing_ds, merged_new_ds], dim="date", join="outer")
    final_ds = final_ds.sortby("date")

    # Also chunk the final dataset for writing
    final_ds = final_ds.chunk({'id_geohash': 50000, 'date': -1})

    # Create output filename
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    logger.info(f"Saving to: {new_dynamic_world_data_file}")
    logger.info("This will process in chunks and may take 30-60 minutes...")

    # Save with compression - this is where actual computation happens
    encoding = {var_name: {'zlib': True, 'complevel': 5}
                for var_name in final_ds.data_vars if var_name not in ['date', 'id_geohash']}

    # Write to disk (processes chunk by chunk)
    final_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding)

    # Get final statistics (requires loading metadata only)
    final_lake_count = len(final_ds.id_geohash)
    final_date_count = len(final_ds.date)

    logger.info("=" * 60)
    logger.info(f"✓ MERGE COMPLETE!")
    logger.info(f"  Output: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {final_lake_count:,}")
    logger.info(f"  Total dates: {final_date_count}")
    logger.info(f"  New lakes: {final_lake_count - len(existing_ds.id_geohash):,}")
    logger.info(f"  New dates: {final_date_count - len(existing_dates):,}")
    logger.info("=" * 60)

    # Clean up
    existing_ds.close()
    merged_new_ds.close()
    final_ds.close()


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()