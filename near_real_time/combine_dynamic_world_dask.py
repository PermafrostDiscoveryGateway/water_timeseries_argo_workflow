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
        logger.info(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")
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

        # Read and combine new data chunks
        logger.info("Reading and combining new data chunks...")

        batch_size = 20
        combined_data = None

        for batch_start in range(0, len(valid_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_chunks))
            logger.info(f"Processing chunks {batch_start + 1}-{batch_end}/{len(valid_chunks)}...")

            batch_data = []
            for chunk_file in valid_chunks[batch_start:batch_end]:
                try:
                    ds_chunk = xr.open_dataset(chunk_file, decode_times=False)

                    # Convert dates
                    if 'date' in ds_chunk.variables:
                        days_offset = 3653
                        ds_chunk['date'] = ds_chunk['date'] + days_offset
                        ds_chunk['date'].attrs['units'] = 'days since 2015-07-01'
                        ds_chunk['date'].attrs['calendar'] = 'proleptic_gregorian'

                    batch_data.append(ds_chunk)

                except Exception as e:
                    logger.error(f"Failed to load chunk {os.path.basename(chunk_file)}: {e}")
                    continue

            if batch_data:
                if len(batch_data) == 1:
                    batch_combined = batch_data[0]
                else:
                    batch_combined = xr.concat(batch_data, dim="id_geohash")

                if combined_data is None:
                    combined_data = batch_combined
                else:
                    # Use combine_first instead of concat to avoid dimension mismatch
                    combined_data = xr.combine_first(combined_data, batch_combined)

            del batch_data
            gc.collect()

        if combined_data is None:
            logger.error("No data to merge from chunks")
            return None

        # Remove duplicate lake IDs by keeping first occurrence
        logger.info("Removing duplicate lake IDs...")
        _, unique_indices = np.unique(combined_data.id_geohash.values, return_index=True)
        combined_data = combined_data.isel(id_geohash=sorted(unique_indices))

        # Get dates
        logger.info("Getting date information...")
        dates = combined_data.date.values
        latest_date = max(dates)

        # Convert to actual date for filename
        ref_date = datetime(2015, 7, 1)
        latest_date_obj = ref_date + pd.Timedelta(days=int(latest_date))
        latest_date_string = latest_date_obj.strftime('%Y_%m_%d')

        logger.info(f"Latest date (actual): {latest_date_string}")
        logger.info(f"New data: {len(combined_data.id_geohash):,} lakes x {len(dates)} dates")

        # Load existing dataset
        logger.info("Loading existing dataset...")
        existing_ds = xr.open_dataset(most_recent_dynamic_world_file, decode_times=False)

        # Get existing dates
        existing_dates = existing_ds.date.values
        logger.info(f"Existing dates: {len(existing_dates)} dates")

        # Remove overlapping dates from new data
        overlapping_dates = set(existing_dates) & set(dates)

        if overlapping_dates:
            logger.warning(f"Removing {len(overlapping_dates)} overlapping dates")
            date_mask = ~np.isin(combined_data.date.values, list(overlapping_dates))
            combined_data = combined_data.isel(date=date_mask)
            dates = combined_data.date.values

        if len(dates) == 0:
            logger.warning("No new dates to add")
            existing_ds.close()
            combined_data.close()
            return None

        logger.info(f"Adding {len(dates)} new dates: {dates}")

        # Create output filename
        new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
        new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

        # Process one date at a time to avoid memory issues
        logger.info("Merging and writing one date at a time...")

        # First, copy existing dataset to new file
        logger.info("Copying existing data to new file...")
        encoding = {var_name: {'zlib': True, 'complevel': 5}
                    for var_name in existing_ds.data_vars if var_name not in ['date', 'id_geohash']}

        # Write existing dataset
        existing_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding)

        # Now append each new date
        from netCDF4 import Dataset

        for date_val in dates:
            logger.info(f"Adding date {date_val} (converted: {ref_date + pd.Timedelta(days=int(date_val))})...")

            # Extract data for this date
            date_data = combined_data.sel(date=date_val)

            # Append to file
            with Dataset(new_dynamic_world_data_file, 'a') as ncfile:
                # Get current number of dates
                current_date_dim = len(ncfile.dimensions['date'])
                new_date_idx = current_date_dim

                # Extend date dimension
                ncfile.dimensions['date'] = (current_date_dim + 1,)

                # Add the date value
                if 'date' in ncfile.variables:
                    date_var = ncfile.variables['date']
                    # Resize the date variable
                    date_var.resize(current_date_dim + 1, axis=0)
                    date_var[new_date_idx] = date_val

                # Add all data variables for this date
                for var_name in existing_ds.data_vars:
                    if var_name not in ['date', 'id_geohash']:
                        # Get the data for this date
                        var_data = date_data[var_name].values

                        # Get or create variable
                        if var_name in ncfile.variables:
                            ncvar = ncfile.variables[var_name]
                            # Resize along date dimension
                            ncvar.resize(current_date_dim + 1, axis=1)
                            ncvar[:, new_date_idx] = var_data
                        else:
                            # Create new variable (shouldn't happen)
                            pass

            # Clear date_data from memory
            del date_data
            gc.collect()

        logger.info("=" * 60)
        logger.info(f"✓ MERGE COMPLETE!")
        logger.info(f"  Output: {new_dynamic_world_data_file}")
        logger.info(f"  Total lakes: {len(existing_ds.id_geohash):,}")
        logger.info(f"  Total dates: {len(existing_dates) + len(dates)}")
        logger.info("=" * 60)

        # Clean up
        existing_ds.close()
        combined_data.close()

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()