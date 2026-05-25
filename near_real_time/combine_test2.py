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

    # Load existing file WITHOUT decoding times
    logger.info("Loading existing dataset...")
    existing_ds = xr.open_dataset(most_recent_dynamic_world_file, decode_times=False)
    existing_dates = set(existing_ds.date.values)
    logger.info(f"Existing dates: {len(existing_dates)}")

    # Process each chunk and combine by date
    chunk_files = sorted(glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc")))
    logger.info(f"Found {len(chunk_files)} chunk files")

    days_offset = 3653
    ref_date = datetime(2015, 7, 1)

    # Collect all new data by date
    all_new_data = {}

    for i, chunk_file in enumerate(chunk_files):
        if (i + 1) % 20 == 0:
            logger.info(f"Processing chunk {i + 1}/{len(chunk_files)}")

        try:
            # Load chunk
            ds = xr.open_dataset(chunk_file, decode_times=False)

            # Adjust dates
            ds['date'] = ds['date'] + days_offset

            # Get unique dates in this chunk
            for date_val in ds.date.values:
                if date_val not in existing_dates and date_val not in all_new_data:
                    # Extract this date's data
                    date_data = ds.sel(date=date_val)
                    # Keep only the data variables (not date and id_geohash)
                    date_data = date_data.drop_vars(['date'], errors='ignore')
                    all_new_data[date_val] = date_data
                    logger.debug(f"  Found new date: {date_val} ({ref_date + pd.Timedelta(days=int(date_val))})")

            ds.close()

        except Exception as e:
            logger.warning(f"Error processing {os.path.basename(chunk_file)}: {e}")
            continue

    if not all_new_data:
        logger.warning("No new dates found!")
        existing_ds.close()
        return None

    logger.info(f"Found {len(all_new_data)} new dates")

    # Create output filename
    latest_date = max(all_new_data.keys())
    latest_date_obj = ref_date + pd.Timedelta(days=int(latest_date))
    latest_date_string = latest_date_obj.strftime('%Y_%m_%d')
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Write output file in chunks (by date)
    logger.info("Writing output file...")

    # First, write the existing data
    encoding = {var_name: {'zlib': True, 'complevel': 5}
                for var_name in existing_ds.data_vars if var_name not in ['date', 'id_geohash']}

    existing_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding)
    existing_ds.close()

    # Now append each new date using netCDF4
    from netCDF4 import Dataset

    for date_val, date_data in sorted(all_new_data.items()):
        logger.info(f"  Adding date {date_val} ({ref_date + pd.Timedelta(days=int(date_val))})...")

        with Dataset(new_dynamic_world_data_file, 'a') as ncfile:
            current_date_dim = len(ncfile.dimensions['date'])
            new_date_idx = current_date_dim

            # Extend date dimension
            ncfile.dimensions['date'] = (current_date_dim + 1,)

            # Add date value
            if 'date' in ncfile.variables:
                date_var = ncfile.variables['date']
                date_var.resize(current_date_dim + 1)
                date_var[new_date_idx] = date_val

            # Add data for each variable
            for var_name in date_data.data_vars:
                if var_name != 'id_geohash':
                    var_data = date_data[var_name].values

                    if var_name in ncfile.variables:
                        ncvar = ncfile.variables[var_name]
                        # Resize along date dimension (assuming shape: (lakes, dates))
                        current_shape = ncvar.shape
                        ncvar.resize(current_shape[0], current_date_dim + 1)
                        ncvar[:, new_date_idx] = var_data

    # Verify
    result_ds = xr.open_dataset(new_dynamic_world_data_file, decode_times=False)
    logger.info("=" * 60)
    logger.info(f"✓ MERGE COMPLETE!")
    logger.info(f"  Output: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {len(result_ds.id_geohash):,}")
    logger.info(f"  Total dates: {len(result_ds.date)}")
    logger.info(f"  New dates added: {len(all_new_data)}")
    logger.info("=" * 60)
    result_ds.close()

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()