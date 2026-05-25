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

    try:
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
        logger.info(f"File size: {os.path.getsize(most_recent_dynamic_world_file) / (1024 ** 3):.2f} GB")

        # Get downloaded chunk files
        downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
        logger.info(f"Found {len(downloaded_files)} chunk files")

        # Validate chunk files
        valid_chunks = []
        for chunk_file in downloaded_files:
            try:
                with xr.open_dataset(chunk_file, decode_times=False) as test:
                    test.close()
                valid_chunks.append(chunk_file)
            except Exception as e:
                logger.warning(f"Skipping corrupted chunk: {os.path.basename(chunk_file)} - {str(e)[:50]}")

        logger.info(f"Valid chunks: {len(valid_chunks)} out of {len(downloaded_files)}")

        if not valid_chunks:
            logger.error("No valid chunk files found!")
            return None

        # For the one-date-at-a-time approach, we don't need to combine all chunks first
        # Instead, we'll process each date as we find it

        # First, collect all unique dates from all chunks
        logger.info("Collecting unique dates from chunks...")
        all_dates = set()
        chunk_info = []  # Store (chunk_file, date_value) pairs

        for chunk_file in valid_chunks:
            try:
                with xr.open_dataset(chunk_file, decode_times=False) as ds:
                    # Convert dates
                    days_offset = 3653
                    dates_in_chunk = ds.date.values + days_offset
                    for date_val in dates_in_chunk:
                        all_dates.add(date_val)
                        chunk_info.append((chunk_file, date_val))
            except Exception as e:
                logger.warning(f"Error reading dates from {os.path.basename(chunk_file)}: {e}")

        logger.info(f"Found {len(all_dates)} unique dates: {sorted(all_dates)}")

        # Load existing dataset to get its dates
        logger.info("Loading existing dataset...")
        existing_ds = xr.open_dataset(most_recent_dynamic_world_file, decode_times=False)
        existing_dates = set(existing_ds.date.values)
        logger.info(f"Existing dates: {len(existing_dates)}")

        # Filter to only new dates (not in existing)
        new_dates = all_dates - existing_dates
        if not new_dates:
            logger.warning("No new dates to add!")
            existing_ds.close()
            return None

        logger.info(f"Adding {len(new_dates)} new dates: {sorted(new_dates)}")
        latest_date = max(new_dates)

        # Convert to actual date for filename
        ref_date = datetime(2015, 7, 1)
        latest_date_obj = ref_date + pd.Timedelta(days=int(latest_date))
        latest_date_string = latest_date_obj.strftime('%Y_%m_%d')

        # Create output filename
        new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
        new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

        # Copy existing dataset to new file
        logger.info("Copying existing data to new file...")
        encoding = {var_name: {'zlib': True, 'complevel': 5}
                    for var_name in existing_ds.data_vars if var_name not in ['date', 'id_geohash']}

        existing_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding)
        existing_ds.close()

        # Now process each new date one at a time
        logger.info("Adding new dates one by one...")
        from netCDF4 import Dataset

        for date_val in sorted(new_dates):
            actual_date = ref_date + pd.Timedelta(days=int(date_val))
            logger.info(f"  Processing date {date_val} ({actual_date.date()})...")

            # Collect data for this specific date from all chunks
            date_data_combined = None

            for chunk_file in valid_chunks:
                try:
                    ds_chunk = xr.open_dataset(chunk_file, decode_times=False)

                    # Convert dates
                    days_offset = 3653
                    ds_chunk['date'] = ds_chunk['date'] + days_offset

                    # Check if this chunk has our date
                    if date_val in ds_chunk.date.values:
                        # Extract just this date
                        date_slice = ds_chunk.sel(date=date_val)

                        if date_data_combined is None:
                            date_data_combined = date_slice
                        else:
                            # Combine along id_geohash dimension
                            date_data_combined = xr.concat([date_data_combined, date_slice], dim="id_geohash")

                    ds_chunk.close()

                except Exception as e:
                    logger.warning(f"Error processing chunk {os.path.basename(chunk_file)} for date {date_val}: {e}")

            if date_data_combined is not None:
                # Remove duplicate lake IDs for this date
                _, unique_idx = np.unique(date_data_combined.id_geohash.values, return_index=True)
                date_data_combined = date_data_combined.isel(id_geohash=sorted(unique_idx))

                # Append to netCDF file
                with Dataset(new_dynamic_world_data_file, 'a') as ncfile:
                    # Get current number of dates
                    current_date_dim = len(ncfile.dimensions['date'])
                    new_date_idx = current_date_dim

                    # Extend date dimension
                    ncfile.dimensions['date'] = (current_date_dim + 1,)

                    # Add the date value
                    if 'date' in ncfile.variables:
                        date_var = ncfile.variables['date']
                        date_var.resize(current_date_dim + 1, axis=0)
                        date_var[new_date_idx] = date_val

                    # Add all data variables for this date
                    for var_name in existing_ds.data_vars:
                        if var_name not in ['date', 'id_geohash'] and var_name in date_data_combined.data_vars:
                            var_data = date_data_combined[var_name].values

                            if var_name in ncfile.variables:
                                ncvar = ncfile.variables[var_name]
                                # Resize along date dimension (usually axis 1 for date)
                                current_shape = ncvar.shape
                                if len(current_shape) == 2:  # (lake, date)
                                    ncvar.resize(current_shape[0], current_date_dim + 1)
                                    ncvar[:, new_date_idx] = var_data
                                else:
                                    logger.warning(f"Unexpected shape for {var_name}: {current_shape}")

                # Clean up
                del date_data_combined
                gc.collect()
            else:
                logger.warning(f"No data found for date {date_val}")

        logger.info("=" * 60)
        logger.info(f"✓ MERGE COMPLETE!")
        logger.info(f"  Output: {new_dynamic_world_data_file}")
        logger.info(f"  Added {len(new_dates)} new dates")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()