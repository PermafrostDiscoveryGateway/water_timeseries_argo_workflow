import pandas as pd
from datetime import datetime
from loguru import logger
import os
import numpy as np
import glob
import xarray as xr
from dotenv import load_dotenv
import gc


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
    logger.info(f"Most recent existing file: {os.path.basename(most_recent_dynamic_world_file)}")

    # Load existing file
    logger.info("Loading existing dataset...")
    existing_ds = xr.open_dataset(most_recent_dynamic_world_file, decode_times=False)
    existing_dates = set(existing_ds.date.values)
    logger.info(f"Existing dates: {len(existing_dates)}")

    # Get unique lake IDs from existing file
    existing_lake_ids = existing_ds.id_geohash.values
    lake_id_to_index = {lake_id: idx for idx, lake_id in enumerate(existing_lake_ids)}
    logger.info(f"Existing lakes: {len(lake_id_to_index):,}")

    # Process chunk files
    chunk_files = sorted(glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc")))
    logger.info(f"Found {len(chunk_files)} chunk files")

    days_offset = 3653
    ref_date = datetime(2015, 7, 1)

    # Dictionary to store new data: date -> array of values for each lake
    new_data_by_date = {}

    logger.info("Processing chunks and collecting new data...")

    for i, chunk_file in enumerate(chunk_files):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(chunk_files)} chunks")

        try:
            ds_chunk = xr.open_dataset(chunk_file, decode_times=False)

            # Adjust dates
            chunk_dates = ds_chunk.date.values + days_offset

            # Get lake IDs in this chunk
            chunk_lake_ids = ds_chunk.id_geohash.values

            # For each lake in chunk, find its index in existing file
            lake_indices = []
            valid_mask = []
            for lake_id in chunk_lake_ids:
                if lake_id in lake_id_to_index:
                    lake_indices.append(lake_id_to_index[lake_id])
                    valid_mask.append(True)
                else:
                    # This lake doesn't exist in existing file - skip it
                    valid_mask.append(False)
                    logger.debug(f"  Lake {lake_id} not found in existing data, skipping")

            if not any(valid_mask):
                ds_chunk.close()
                continue

            # For each date in this chunk
            for date_idx, orig_date_val in enumerate(ds_chunk.date.values):
                date_val = orig_date_val + days_offset

                # Skip if date already exists
                if date_val in existing_dates:
                    continue

                # Initialize array for this date if not exists
                if date_val not in new_data_by_date:
                    new_data_by_date[date_val] = {
                        'lake_indices': [],
                        'data': {var: [] for var in existing_ds.data_vars}
                    }

                # Get data for this date
                date_data = ds_chunk.isel(date=date_idx)

                # Add to collection
                for var_name in existing_ds.data_vars:
                    if var_name in date_data.data_vars:
                        var_data = date_data[var_name].values
                        # Filter to valid lakes
                        for idx, valid in enumerate(valid_mask):
                            if valid:
                                new_data_by_date[date_val]['data'][var_name].append(var_data[idx])

                # Add lake indices (once per date)
                if not new_data_by_date[date_val]['lake_indices']:
                    for idx, valid in enumerate(valid_mask):
                        if valid:
                            new_data_by_date[date_val]['lake_indices'].append(lake_indices[idx])

            ds_chunk.close()
            gc.collect()

        except Exception as e:
            logger.warning(f"Error processing {os.path.basename(chunk_file)}: {e}")
            continue

    if not new_data_by_date:
        logger.warning("No new dates found!")
        existing_ds.close()
        return None

    logger.info(f"Found {len(new_data_by_date)} new dates")

    # Create output filename
    latest_date = max(new_data_by_date.keys())
    latest_date_obj = ref_date + pd.Timedelta(days=int(latest_date))
    latest_date_string = latest_date_obj.strftime('%Y_%m_%d')
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Write output file using netCDF4 directly (more efficient for appending)
    logger.info("Writing output file...")

    from netCDF4 import Dataset
    import tempfile

    # First, copy existing file to temp location
    temp_file = tempfile.mktemp(suffix='.nc')
    existing_ds.to_netcdf(temp_file)
    existing_ds.close()

    # Append new dates
    with Dataset(temp_file, 'a') as ncfile:
        current_date_count = len(ncfile.dimensions['date'])
        total_new_dates = len(new_data_by_date)

        # Extend date dimension
        ncfile.dimensions['date'] = current_date_count + total_new_dates

        # Prepare date variable
        date_var = ncfile.variables['date']
        date_var.resize(current_date_count + total_new_dates)

        # Prepare data variables
        for var_name in existing_ds.data_vars:
            if var_name in ncfile.variables:
                ncvar = ncfile.variables[var_name]
                # Resize along date dimension
                ncvar.resize(ncvar.shape[0], current_date_count + total_new_dates)

        # Add each new date
        for new_idx, (date_val, date_data) in enumerate(sorted(new_data_by_date.items())):
            logger.info(f"  Adding date {date_val} ({ref_date + pd.Timedelta(days=int(date_val))})...")

            # Add date value
            date_var[current_date_count + new_idx] = date_val

            # Add data for each variable
            for var_name in existing_ds.data_vars:
                if var_name in ncfile.variables:
                    ncvar = ncfile.variables[var_name]

                    # Create array for this date (initialize with NaN)
                    date_column = np.full(ncvar.shape[0], np.nan, dtype=np.float64)

                    # Fill in values where we have them
                    for lake_pos, lake_idx in enumerate(date_data['lake_indices']):
                        date_column[lake_idx] = date_data['data'][var_name][lake_pos]

                    # Write to file
                    ncvar[:, current_date_count + new_idx] = date_column

    # Move temp file to final location
    import shutil
    shutil.move(temp_file, new_dynamic_world_data_file)

    # Verify
    result_ds = xr.open_dataset(new_dynamic_world_data_file, decode_times=False)
    logger.info("=" * 60)
    logger.info(f"✓ MERGE COMPLETE!")
    logger.info(f"  Output: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {len(result_ds.id_geohash):,}")
    logger.info(f"  Total dates: {len(result_ds.date)}")
    logger.info(f"  New dates added: {len(new_data_by_date)}")
    logger.info("=" * 60)
    result_ds.close()

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()