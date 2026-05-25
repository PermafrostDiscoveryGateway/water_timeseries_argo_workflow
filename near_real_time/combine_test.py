import os
import glob
import shutil
import tempfile
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv
import netCDF4 as nc
import numpy as np


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
    logger.info(f"File size: {os.path.getsize(most_recent_dynamic_world_file) / (1024 ** 3):.2f} GB")

    # Get existing dates
    logger.info("Reading existing dates...")
    with nc.Dataset(most_recent_dynamic_world_file, 'r') as src:
        existing_dates = set(src.variables['date'][:])
    logger.info(f"Existing dates: {len(existing_dates)}")

    # Get new chunk files
    downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
    logger.info(f"Found {len(downloaded_files)} chunk files")

    if not downloaded_files:
        logger.error("No chunk files found!")
        return None

    # Create working copy
    temp_dir = tempfile.mkdtemp(prefix="netcdf_merge_")
    working_file = os.path.join(temp_dir, "combined.nc")
    logger.info(f"Creating working copy: {working_file}")
    shutil.copy2(most_recent_dynamic_world_file, working_file)

    # Process each chunk
    days_offset = 3653
    ref_date = datetime(2015, 7, 1)
    all_new_dates = set()

    logger.info(f"Processing {len(downloaded_files)} chunk files...")

    for i, chunk_file in enumerate(downloaded_files):
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(downloaded_files)} chunks processed")

        try:
            # Read chunk
            with nc.Dataset(chunk_file, 'r') as chunk:
                chunk_dates = chunk.variables['date'][:] + days_offset

                # Find new dates
                new_dates_in_chunk = [d for d in chunk_dates if d not in existing_dates]

                if not new_dates_in_chunk:
                    logger.debug(f"No new dates in chunk {i + 1}")
                    continue

                logger.debug(f"Found {len(new_dates_in_chunk)} new dates in chunk {i + 1}")
                all_new_dates.update(new_dates_in_chunk)

                # Get lake IDs in this chunk
                chunk_lakes = chunk.variables['id_geohash'][:]

                # Open working file for appending
                with nc.Dataset(working_file, 'a') as working:
                    # Get current dimensions
                    current_lake_count = len(working.dimensions['id_geohash'])
                    current_date_count = len(working.dimensions['date'])

                    # Get existing lake IDs
                    existing_lakes = working.variables['id_geohash'][:]

                    # Find which lakes in chunk are not in working file
                    # (or we need to match by ID)
                    lake_indices = []
                    for lake_id in chunk_lakes:
                        # Find where this lake exists in the working file
                        idx = np.where(existing_lakes == lake_id)[0]
                        if len(idx) > 0:
                            lake_indices.append(idx[0])
                        else:
                            # This lake is new - but that shouldn't happen
                            logger.warning(f"New lake ID {lake_id} not found in existing file")
                            lake_indices.append(-1)

                    # For each new date, append data
                    for date_val in new_dates_in_chunk:
                        # Get date index in chunk
                        date_idx = np.where(chunk_dates == date_val)[0][0]

                        # Extend date dimension
                        new_date_idx = current_date_count
                        working.dimensions['date'] = current_date_count + 1

                        # Add date value
                        if 'date' in working.variables:
                            date_var = working.variables['date']
                            # Resize and add new date
                            date_var.resize(current_date_count + 1)
                            date_var[new_date_idx] = date_val

                        # For each variable, append the data for this date
                        for var_name in working.variables:
                            if var_name not in ['date', 'id_geohash']:
                                var_data = chunk.variables[var_name][date_idx, :]

                                # Map to correct lake order
                                ordered_data = np.zeros(len(existing_lakes))
                                for chunk_idx, lake_idx in enumerate(lake_indices):
                                    if lake_idx >= 0:
                                        ordered_data[lake_idx] = var_data[chunk_idx]

                                # Append to variable
                                working_var = working.variables[var_name]
                                # Resize along date dimension
                                current_shape = working_var.shape
                                working_var.resize(current_shape[0], current_date_count + 1)
                                working_var[:, new_date_idx] = ordered_data

                    current_date_count += 1

        except Exception as e:
            logger.error(f"Error processing chunk {i + 1}: {e}")
            continue

    if not all_new_dates:
        logger.warning("No new dates found in any chunk!")
        shutil.rmtree(temp_dir)
        return None

    # Get latest date for filename
    latest_date = max(all_new_dates)
    latest_date_obj = ref_date + timedelta(days=int(latest_date))
    latest_date_string = latest_date_obj.strftime('%Y_%m_%d')
    logger.info(f"Latest date added: {latest_date_string}")

    # Final output filename
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Move working file to final location
    logger.info(f"Saving final file: {new_dynamic_world_data_file}")
    shutil.move(working_file, new_dynamic_world_data_file)

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    # Verify
    with nc.Dataset(new_dynamic_world_data_file, 'r') as result:
        logger.info("=" * 60)
        logger.info(f"✓ MERGE COMPLETE!")
        logger.info(f"  Output: {new_dynamic_world_data_file}")
        logger.info(f"  Total lakes: {len(result.dimensions['id_geohash']):,}")
        logger.info(f"  Total dates: {len(result.dimensions['date'])}")
        logger.info(f"  New dates added: {len(all_new_dates)}")
        logger.info("=" * 60)

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()