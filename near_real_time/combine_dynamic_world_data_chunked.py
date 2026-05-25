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
            # Test with decode_times=True to ensure dates can be decoded
            with xr.open_dataset(chunk_file, decode_times=True) as test:
                test.close()
            valid_chunks.append(chunk_file)
        except Exception as e:
            logger.warning(f"Skipping chunk with date decoding issue: {os.path.basename(chunk_file)}")
            logger.warning(f"Will try manual date handling for: {os.path.basename(chunk_file)}")
            # Still consider it valid, we'll handle dates manually
            valid_chunks.append(chunk_file)

    logger.info(f"Valid chunks: {len(valid_chunks)} out of {len(downloaded_files)}")

    if not valid_chunks:
        logger.error("No valid chunk files found!")
        return None

    # Read and combine new data chunks with proper date handling
    logger.info("Reading and combining new data chunks...")
    merged_new_ds = None

    for chunk_file in valid_chunks:
        logger.debug(f"Loading chunk: {os.path.basename(chunk_file)}")
        try:
            # Try to open normally first
            try:
                ds_chunk = xr.open_dataset(chunk_file, decode_times=True)
            except ValueError as e:
                if "unable to decode time units" in str(e):
                    # Manual date decoding for proleptic_gregorian calendar
                    logger.warning(f"Manual date decoding for: {os.path.basename(chunk_file)}")
                    ds_chunk = xr.open_dataset(chunk_file, decode_times=False)

                    # Manually decode the dates
                    if 'date' in ds_chunk.variables:
                        date_var = ds_chunk['date']
                        units = date_var.attrs.get('units', 'days since 2015-07-01')
                        calendar = date_var.attrs.get('calendar', 'proleptic_gregorian')

                        # Use pandas for conversion (handles proleptic_gregorian)
                        # Parse the reference date from units
                        import re
                        match = re.search(r'days since (\d{4}-\d{2}-\d{2})', units)
                        if match:
                            ref_date = pd.Timestamp(match.group(1))
                            # Convert integer days to datetime
                            date_values = ref_date + pd.to_timedelta(date_var.values, unit='D')
                            ds_chunk['date'] = xr.DataArray(
                                date_values,
                                dims=date_var.dims,
                                attrs={
                                    'long_name': 'date',
                                    'standard_name': 'time',
                                    'units': units,
                                    'calendar': calendar
                                }
                            )
                        else:
                            raise ValueError(f"Could not parse units: {units}")
                else:
                    raise e

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

    # Get dates (now as datetime objects)
    dates = pd.to_datetime(merged_new_ds.date.values)
    latest_date = max(dates)
    latest_date_string = latest_date.strftime('%Y_%m_%d')

    logger.info(f"Latest date: {latest_date_string}")
    logger.info(f"New data: {len(merged_new_ds.id_geohash):,} lakes x {len(dates)} dates")

    # Load existing dataset
    logger.info("Loading existing dataset...")
    try:
        existing_ds = xr.open_dataset(most_recent_dynamic_world_file, decode_times=True)
    except ValueError as e:
        if "unable to decode time units" in str(e):
            logger.warning("Manual date decoding for existing file")
            existing_ds = xr.open_dataset(most_recent_dynamic_world_file, decode_times=False)
            if 'date' in existing_ds.variables:
                date_var = existing_ds['date']
                units = date_var.attrs.get('units', 'days since 2015-07-01')
                import re
                match = re.search(r'days since (\d{4}-\d{2}-\d{2})', units)
                if match:
                    ref_date = pd.Timestamp(match.group(1))
                    date_values = ref_date + pd.to_timedelta(date_var.values, unit='D')
                    existing_ds['date'] = xr.DataArray(
                        date_values,
                        dims=date_var.dims,
                        attrs=date_var.attrs
                    )
        else:
            raise e

    # Get existing dates (as datetime)
    existing_dates = pd.to_datetime(existing_ds.date.values)
    logger.info(f"Existing dates: {len(existing_dates)} dates")
    logger.info(f"Date range: {existing_dates.min()} to {existing_dates.max()}")

    # Remove overlapping dates
    new_dates_set = set(dates)
    existing_dates_set = set(existing_dates)
    overlapping_dates = existing_dates_set & new_dates_set

    if overlapping_dates:
        logger.warning(f"Removing {len(overlapping_dates)} overlapping dates")
        # Convert to string comparison for safety
        overlapping_str = [d.strftime('%Y-%m-%d') for d in overlapping_dates]
        mask = ~merged_new_ds.date.dt.strftime('%Y-%m-%d').isin(overlapping_str)
        merged_new_ds = merged_new_ds.sel(date=mask)
        dates = pd.to_datetime(merged_new_ds.date.values)

    if len(dates) == 0:
        logger.warning("No new dates to add")
        existing_ds.close()
        merged_new_ds.close()
        return None

    logger.info(f"Adding {len(dates)} new dates")
    logger.info(f"New date range: {dates.min()} to {dates.max()}")

    # Merge datasets
    logger.info("Merging datasets...")
    final_ds = xr.concat([existing_ds, merged_new_ds], dim="date", join="outer")
    final_ds = final_ds.sortby("date")

    logger.info(f"Final dataset: {len(final_ds.id_geohash):,} lakes x {len(final_ds.date)} dates")

    # Save the file
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    logger.info(f"Saving to: {new_dynamic_world_data_file}")

    # Use compression
    encoding = {var_name: {'zlib': True, 'complevel': 5}
                for var_name in final_ds.data_vars if var_name not in ['date', 'id_geohash']}

    # For dates, ensure they're saved in a standard format
    final_ds['date'].encoding = {'units': 'days since 2000-01-01', 'calendar': 'proleptic_gregorian'}

    final_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding)

    logger.info("=" * 60)
    logger.info(f"✓ MERGE COMPLETE!")
    logger.info(f"  Output: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {len(final_ds.id_geohash):,}")
    logger.info(f"  Total dates: {len(final_ds.date)}")
    logger.info(f"  Date range: {final_ds.date.values.min()} to {final_ds.date.values.max()}")
    logger.info("=" * 60)

    # Clean up
    existing_ds.close()
    merged_new_ds.close()
    final_ds.close()


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()