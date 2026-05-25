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

    # Get environment variables
    project = os.environ['project']
    EE_PROJECT_ID = project
    os.environ["EE_PROJECT"] = EE_PROJECT_ID
    dynamic_world_dir = os.environ['dynamic_world_dir']
    logger.debug(f"dynamic_world_dir: {dynamic_world_dir}")
    split_new_dynamic_world_data_dir = os.environ['split_new_dynamic_world_data_dir']
    logger.debug(f"split_new_dynamic_world_data_dir: {split_new_dynamic_world_data_dir}")

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
        return None

    most_recent_dynamic_world_file = max(valid_files, key=os.path.getctime)
    logger.info(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")

    # Get downloaded chunk files
    downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
    logger.info(f"Found {len(downloaded_files)} chunk files")

    # Use xarray to read and combine chunks
    logger.info("Reading and combining new data chunks with xarray...")
    merged_new_ds = None

    for chunk_file in downloaded_files:
        logger.debug(f"Loading chunk: {chunk_file}")
        try:
            ds_chunk = xr.open_dataset(chunk_file)
            if merged_new_ds is None:
                merged_new_ds = ds_chunk
            else:
                merged_new_ds = xr.concat([merged_new_ds, ds_chunk], dim="id_geohash")
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_file}: {e}")
            continue

    if merged_new_ds is None:
        logger.error("No data to merge from chunks")
        return None

    # Remove duplicates
    _, unique_indices = np.unique(merged_new_ds.id_geohash.values, return_index=True)
    merged_new_ds = merged_new_ds.isel(id_geohash=sorted(unique_indices))

    # Get dates
    dates = merged_new_ds.date.values
    latest_date = max(dates)
    try:
        latest_date_string = str(latest_date).split('T')[0].replace('-', '_')
    except:
        latest_date_string = str(latest_date).replace('-', '_')

    logger.info(f"The latest date is {latest_date_string}")
    logger.info(f"Merged new data: {len(merged_new_ds.id_geohash):,} lakes x {len(dates)} dates")

    # Load existing dataset
    logger.info("Loading existing dataset...")
    existing_ds = xr.open_dataset(most_recent_dynamic_world_file)

    # Get existing dates and lakes
    existing_dates = existing_ds.date.values
    existing_lake_ids = existing_ds.id_geohash.values
    existing_lake_ids_set = set(str(lake_id) for lake_id in existing_lake_ids)

    # Get new lake IDs
    new_lake_ids = merged_new_ds.id_geohash.values
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
    new_dates_set = set(pd.to_datetime(merged_new_ds.date.values))
    existing_dates_set = set(pd.to_datetime(existing_ds.date.values))
    overlapping_dates = existing_dates_set & new_dates_set

    if overlapping_dates:
        logger.warning(f"Removing {len(overlapping_dates)} overlapping dates")
        mask = ~merged_new_ds.date.isin(list(overlapping_dates))
        merged_new_ds = merged_new_ds.sel(date=mask)
        dates = merged_new_ds.date.values

    if len(dates) == 0:
        logger.warning("No new dates to add")
        return None

    logger.info(f"Adding {len(dates)} new dates: {dates}")

    # FAST MERGE using xarray's built-in concatenation
    logger.info("=" * 60)
    logger.info("MERGING DATASETS (using xarray)...")

    # Reindex the new dataset to match existing lake IDs
    logger.info("Reindexing new data to match existing lake structure...")

    # Create a mapping for common lakes
    # Convert to pandas Series for faster mapping
    existing_ids_series = pd.Series(range(len(existing_lake_ids)), index=[str(x) for x in existing_lake_ids])

    # Find indices of common lakes in existing dataset
    common_lakes_list = list(common_lakes)
    common_existing_indices = [existing_ids_series[lake_id] for lake_id in common_lakes_list]

    # For new data, we need to align with existing lake order
    # Create a new dataset aligned with existing lake order
    aligned_new_data = []

    for var_name in existing_ds.data_vars:
        if var_name not in ['date', 'id_geohash']:
            logger.info(f"  Aligning {var_name}...")
            # Create array of NaN for all existing lakes
            aligned_array = np.full((len(existing_lake_ids), len(dates)), np.nan, dtype=np.float64)

            # Fill in data for common lakes
            for i, lake_id in enumerate(common_lakes_list):
                # Find this lake in new data
                new_lake_idx = np.where([str(x) == lake_id for x in new_lake_ids])[0]
                if len(new_lake_idx) > 0:
                    # Get data for this lake from new dataset
                    lake_data = merged_new_ds[var_name].isel(id_geohash=new_lake_idx[0]).values
                    if len(lake_data.shape) == 1:  # 1D array (time)
                        aligned_array[common_existing_indices[i], :] = lake_data
                    else:
                        aligned_array[common_existing_indices[i], :] = lake_data.flatten()

            aligned_new_data.append(aligned_array)

    # Create new dataset for the aligned data
    aligned_new_ds = xr.Dataset(
        data_vars={
            var_name: (['id_geohash', 'date'], aligned_new_data[i])
            for i, var_name in enumerate([v for v in existing_ds.data_vars if v not in ['date', 'id_geohash']])
        },
        coords={
            'id_geohash': existing_lake_ids,
            'date': dates
        }
    )

    # Now concatenate along date dimension
    logger.info("Concatenating along date dimension...")
    final_ds = xr.concat([existing_ds, aligned_new_ds], dim="date")

    # Sort by date
    final_ds = final_ds.sortby("date")

    logger.info(f"Final dataset: {len(final_ds.id_geohash):,} lakes x {len(final_ds.date)} dates")

    # If there are new lakes, we need to add them
    if lakes_only_in_new:
        logger.info(f"Adding {len(lakes_only_in_new):,} new lakes to the dataset...")

        # Create dataset for new lakes
        new_lakes_data = {}
        new_lake_ids_list = list(lakes_only_in_new)

        # Find indices of new lakes in the new dataset
        new_lake_indices = []
        for lake_id in new_lake_ids_list:
            idx = np.where([str(x) == lake_id for x in new_lake_ids])[0]
            if len(idx) > 0:
                new_lake_indices.append(idx[0])

        for var_name in existing_ds.data_vars:
            if var_name not in ['date', 'id_geohash']:
                # Extract data for new lakes from merged_new_ds
                new_lake_data = merged_new_ds[var_name].isel(id_geohash=new_lake_indices).values
                if len(new_lake_data.shape) == 1:
                    new_lake_data = new_lake_data.reshape(-1, 1)
                new_lakes_data[var_name] = (['id_geohash', 'date'], new_lake_data)

        # Create dataset for new lakes
        new_lakes_ds = xr.Dataset(
            data_vars=new_lakes_data,
            coords={
                'id_geohash': new_lake_ids_list,
                'date': dates
            }
        )

        # Concatenate along id_geohash dimension
        final_ds = xr.concat([final_ds, new_lakes_ds], dim="id_geohash")
        logger.info(f"Final dataset with new lakes: {len(final_ds.id_geohash):,} lakes x {len(final_ds.date)} dates")

    # Save the final dataset
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    logger.info(f"Saving merged dataset to: {new_dynamic_world_data_file}")

    # Use compression for efficient storage
    encoding = {var_name: {'zlib': True, 'complevel': 5}
                for var_name in final_ds.data_vars if var_name not in ['date', 'id_geohash']}

    final_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding)

    logger.info("=" * 60)
    logger.info(f"✓ SUCCESSFULLY MERGED DATASETS!")
    logger.info(f"  Original: {most_recent_dynamic_world_file}")
    logger.info(f"  New file: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {len(final_ds.id_geohash):,}")
    logger.info(f"  Total dates: {len(final_ds.date)}")
    logger.info(f"  New lakes added: {len(lakes_only_in_new):,}")
    logger.info(f"  New dates added: {len(dates)}")
    logger.info("=" * 60)

    # Clean up
    existing_ds.close()
    merged_new_ds.close()
    final_ds.close()


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()