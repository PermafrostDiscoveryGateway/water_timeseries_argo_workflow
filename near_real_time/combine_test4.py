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
    existing_lake_ids = list(existing_ds.id_geohash.values)
    existing_lake_set = set(existing_lake_ids)
    logger.info(f"Existing: {len(existing_lake_ids):,} lakes, {len(existing_dates)} dates")

    # Process chunk files to collect all new lakes and dates
    chunk_files = sorted(glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc")))
    logger.info(f"Found {len(chunk_files)} chunk files")

    days_offset = 3653
    ref_date = datetime(2015, 7, 1)

    # Collect all unique lake IDs from chunks
    logger.info("Step 1: Collecting all lake IDs from chunks...")
    all_chunk_lake_ids = set()
    for i, chunk_file in enumerate(chunk_files):
        if (i + 1) % 50 == 0:
            logger.info(
                f"  Scanned {i + 1}/{len(chunk_files)} chunks, total unique lakes so far: {len(all_chunk_lake_ids):,}")
        try:
            with xr.open_dataset(chunk_file, decode_times=False) as ds:
                for lake_id in ds.id_geohash.values:
                    all_chunk_lake_ids.add(lake_id)
        except Exception as e:
            logger.warning(f"Error reading {os.path.basename(chunk_file)}: {e}")

    # Find new lakes
    new_lake_ids = all_chunk_lake_ids - existing_lake_set
    logger.info(f"Found {len(new_lake_ids):,} new lakes (not in existing file)")

    # Create combined lake list (existing + new)
    all_lake_ids = list(existing_lake_ids) + list(new_lake_ids)
    lake_id_to_index = {lake_id: idx for idx, lake_id in enumerate(all_lake_ids)}
    logger.info(f"Total lakes in output: {len(all_lake_ids):,}")

    # Collect new dates and data
    logger.info("Step 2: Collecting new dates and data from chunks...")
    new_dates_found = set()
    new_data_by_date = {}

    for i, chunk_file in enumerate(chunk_files):
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(chunk_files)} chunks, found {len(new_dates_found)} new dates so far")

        try:
            ds_chunk = xr.open_dataset(chunk_file, decode_times=False)

            # Get chunk lake IDs and map to indices
            chunk_lake_ids = ds_chunk.id_geohash.values
            chunk_indices = [lake_id_to_index[lake_id] for lake_id in chunk_lake_ids]

            # For each date in this chunk
            for date_idx, orig_date_val in enumerate(ds_chunk.date.values):
                date_val = orig_date_val + days_offset

                # Skip if date already exists in existing file
                if date_val in existing_dates:
                    continue

                new_dates_found.add(date_val)

                # Initialize storage for this date if needed
                if date_val not in new_data_by_date:
                    new_data_by_date[date_val] = {
                        'indices': [],
                        'data': {var: [] for var in existing_ds.data_vars}
                    }

                # Get data for this date
                date_data = ds_chunk.isel(date=date_idx)

                # Add data for each variable
                for var_name in existing_ds.data_vars:
                    if var_name in date_data.data_vars:
                        var_values = date_data[var_name].values
                        for idx, val in enumerate(var_values):
                            new_data_by_date[date_val]['data'][var_name].append(val)

                # Add lake indices (once per date)
                if not new_data_by_date[date_val]['indices']:
                    new_data_by_date[date_val]['indices'].extend(chunk_indices)

            ds_chunk.close()
            gc.collect()

        except Exception as e:
            logger.warning(f"Error processing {os.path.basename(chunk_file)}: {e}")
            continue

    if not new_dates_found:
        logger.warning("No new dates found!")
        existing_ds.close()
        return None

    logger.info(f"Found {len(new_dates_found)} new dates: {sorted(new_dates_found)}")

    # Create output filename
    latest_date = max(new_dates_found)
    latest_date_obj = ref_date + pd.Timedelta(days=int(latest_date))
    latest_date_string = latest_date_obj.strftime('%Y_%m_%d')
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Write output file
    logger.info("Step 3: Writing output file...")

    from netCDF4 import Dataset
    import tempfile

    # Create a temporary file for the new combined dataset
    temp_file = tempfile.mktemp(suffix='.nc')

    # First, write the existing data with the full lake set
    logger.info("  Creating new file structure...")

    # Create new file with dimensions
    with Dataset(temp_file, 'w', format='NETCDF4') as ncfile:
        # Create dimensions
        ncfile.createDimension('id_geohash', len(all_lake_ids))
        ncfile.createDimension('date', len(existing_dates) + len(new_dates_found))

        # Create id_geohash variable (string)
        id_var = ncfile.createVariable('id_geohash', 'S12', ('id_geohash',))
        logger.info("  Writing lake IDs...")
        for idx, lake_id in enumerate(all_lake_ids):
            if idx % 500000 == 0:
                logger.info(f"    Written {idx:,}/{len(all_lake_ids):,} lake IDs")
            id_var[idx] = lake_id.encode('utf-8')

        # Create date variable
        date_var = ncfile.createVariable('date', 'i8', ('date',), zlib=True, complevel=5)
        date_var.units = 'days since 2015-07-01'
        date_var.calendar = 'proleptic_gregorian'

        # Create data variables
        logger.info("  Creating data variables...")
        for var_name in existing_ds.data_vars:
            ncfile.createVariable(var_name, 'f8', ('id_geohash', 'date'),
                                  zlib=True, complevel=5, fill_value=np.nan)
            ncfile.variables[var_name].units = 'ha'

        # Fill existing dates
        logger.info("  Filling existing dates...")
        existing_dates_sorted = sorted(existing_dates)
        for date_idx, date_val in enumerate(existing_dates_sorted):
            if date_idx % 5 == 0:
                actual_date = ref_date + pd.Timedelta(days=int(date_val))
                logger.info(
                    f"    Writing existing date {date_idx + 1}/{len(existing_dates_sorted)}: {date_val} ({actual_date.date()})")

            date_var[date_idx] = date_val

            # Get data for this date from existing dataset
            existing_date_data = existing_ds.sel(date=date_val)
            for var_name in existing_ds.data_vars:
                var_data = existing_date_data[var_name].values
                # Write directly - no mapping needed as order is same
                ncfile.variables[var_name][:, date_idx] = var_data

            # Force flush every few dates
            if date_idx % 10 == 0:
                ncfile.sync()

        # Fill new dates
        logger.info("  Filling new dates...")
        current_date_idx = len(existing_dates)

        for new_idx, (date_val, date_data) in enumerate(sorted(new_data_by_date.items())):
            actual_date = ref_date + pd.Timedelta(days=int(date_val))
            logger.info(f"    Adding new date {new_idx + 1}/{len(new_data_by_date)}: {date_val} ({actual_date.date()})")

            date_var[current_date_idx + new_idx] = date_val

            # Fill data for this date
            for var_name in existing_ds.data_vars:
                # Get the data for this variable
                var_values = date_data['data'][var_name]
                lake_indices = date_data['indices']

                # Initialize with NaN
                full_column = np.full(len(all_lake_ids), np.nan, dtype=np.float64)

                # Fill in values where we have them
                for pos, lake_idx in enumerate(lake_indices):
                    full_column[lake_idx] = var_values[pos]

                # Write to file
                ncfile.variables[var_name][:, current_date_idx + new_idx] = full_column

            # Force flush after each new date
            ncfile.sync()

    existing_ds.close()

    # Move temp file to final location
    import shutil
    shutil.move(temp_file, new_dynamic_world_data_file)

    # Verify
    result_ds = xr.open_dataset(new_dynamic_world_data_file, decode_times=False)
    logger.info("=" * 60)
    logger.info(f"✓ MERGE COMPLETE!")
    logger.info(f"  Output: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {len(result_ds.id_geohash):,}")
    logger.info(f"    - Existing lakes: {len(existing_lake_ids):,}")
    logger.info(f"    - New lakes: {len(new_lake_ids):,}")
    logger.info(f"  Total dates: {len(result_ds.date)}")
    logger.info(f"    - Existing dates: {len(existing_dates)}")
    logger.info(f"    - New dates: {len(new_dates_found)}")
    logger.info("=" * 60)
    result_ds.close()

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()