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

        # First, figure out what new dates we have
        logger.info("Collecting new dates from chunks...")
        all_new_dates = set()
        days_offset = 3653
        ref_date = datetime(2015, 7, 1)

        for chunk_file in valid_chunks:
            try:
                with xr.open_dataset(chunk_file, decode_times=False) as ds:
                    dates_in_chunk = ds.date.values + days_offset
                    for date_val in dates_in_chunk:
                        all_new_dates.add(date_val)
            except Exception as e:
                logger.warning(f"Error reading dates from {os.path.basename(chunk_file)}: {e}")

        logger.info(f"Found {len(all_new_dates)} unique dates in new data: {sorted(all_new_dates)}")

        # Load existing dataset to get its dates (just metadata, not data)
        existing_ds = xr.open_dataset(most_recent_dynamic_world_file, decode_times=False)
        existing_dates = set(existing_ds.date.values)
        logger.info(f"Existing dates: {len(existing_dates)}")

        # Filter to only new dates
        new_dates = all_new_dates - existing_dates
        if not new_dates:
            logger.warning("No new dates to add!")
            existing_ds.close()
            return None

        logger.info(f"Adding {len(new_dates)} new dates: {sorted(new_dates)}")
        latest_date = max(new_dates)
        latest_date_obj = ref_date + pd.Timedelta(days=int(latest_date))
        latest_date_string = latest_date_obj.strftime('%Y_%m_%d')

        # Create output filename
        new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
        new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

        # Process in chunks of lakes
        lake_chunk_size = 100000
        total_lakes = len(existing_ds.id_geohash)
        num_lake_chunks = (total_lakes + lake_chunk_size - 1) // lake_chunk_size

        logger.info(f"Processing {total_lakes:,} lakes in {num_lake_chunks} chunks of {lake_chunk_size:,}")
        logger.info(f"Will add {len(new_dates)} new dates")

        from netCDF4 import Dataset

        # Create the output file and write it chunk by chunk
        first_chunk = True

        for lake_chunk_idx in range(num_lake_chunks):
            lake_start = lake_chunk_idx * lake_chunk_size
            lake_end = min(lake_start + lake_chunk_size, total_lakes)

            logger.info(
                f"Processing lake chunk {lake_chunk_idx + 1}/{num_lake_chunks} (lakes {lake_start:,}-{lake_end:,})...")

            # Load chunk of existing data
            existing_chunk = existing_ds.isel(id_geohash=slice(lake_start, lake_end))

            # Get the lake IDs in this chunk
            lake_ids_in_chunk = set(existing_chunk.id_geohash.values)

            # Collect new data for these lakes across all new dates
            new_data_by_date = {date_val: None for date_val in new_dates}

            for chunk_file in valid_chunks:
                try:
                    ds_chunk = xr.open_dataset(chunk_file, decode_times=False)

                    # Convert dates
                    ds_chunk['date'] = ds_chunk['date'] + days_offset
                    ds_chunk['date'].attrs['units'] = 'days since 2015-07-01'

                    # Filter to lakes we care about and new dates
                    chunk_lakes = set(ds_chunk.id_geohash.values)
                    overlap_lakes = lake_ids_in_chunk & chunk_lakes

                    if overlap_lakes:
                        for date_val in new_dates:
                            if date_val in ds_chunk.date.values:
                                # Extract this date and these lakes
                                date_slice = ds_chunk.sel(date=date_val)
                                lake_slice = date_slice.sel(id_geohash=list(overlap_lakes))

                                if new_data_by_date[date_val] is None:
                                    new_data_by_date[date_val] = lake_slice
                                else:
                                    new_data_by_date[date_val] = xr.concat(
                                        [new_data_by_date[date_val], lake_slice],
                                        dim="id_geohash"
                                    )

                    ds_chunk.close()

                except Exception as e:
                    logger.warning(f"Error in chunk {os.path.basename(chunk_file)}: {e}")

            # Combine existing chunk with new data for all dates
            all_dates_combined = [existing_chunk]

            for date_val in sorted(new_dates):
                if new_data_by_date[date_val] is not None:
                    # Ensure the date coordinate is set correctly
                    new_data_by_date[date_val]['date'] = date_val
                    all_dates_combined.append(new_data_by_date[date_val])

            # Combine along date dimension
            if len(all_dates_combined) > 1:
                chunk_final = xr.concat(all_dates_combined, dim="date")
                chunk_final = chunk_final.sortby("date")
            else:
                chunk_final = all_dates_combined[0]

            # Write this chunk to the output file
            encoding = {var_name: {'zlib': True, 'complevel': 5}
                        for var_name in chunk_final.data_vars if var_name not in ['date', 'id_geohash']}

            if first_chunk:
                # Write first chunk (creates the file)
                chunk_final.to_netcdf(new_dynamic_world_data_file, encoding=encoding, mode='w')
                first_chunk = False
            else:
                # Append this chunk to the existing file
                with Dataset(new_dynamic_world_data_file, 'a') as main_f:
                    # Get current number of lakes
                    current_lake_count = len(main_f.dimensions['id_geohash'])
                    new_lake_count = len(chunk_final.id_geohash)

                    # Get number of dates (should be same across chunks)
                    num_dates = len(main_f.dimensions['date'])

                    # Extend the id_geohash dimension
                    main_f.dimensions['id_geohash'] = current_lake_count + new_lake_count

                    # Append id_geohash values
                    if 'id_geohash' in main_f.variables:
                        id_var = main_f.variables['id_geohash']
                        id_var.resize(current_lake_count + new_lake_count)
                        id_var[current_lake_count:] = chunk_final.id_geohash.values

                    # Append data for each variable
                    for var_name in chunk_final.data_vars:
                        if var_name == 'date':
                            continue

                        var_data = chunk_final[var_name].values

                        if var_name in main_f.variables:
                            main_var = main_f.variables[var_name]

                            # Check the number of dimensions
                            if len(main_var.dimensions) == 2:  # (lakes, dates)
                                # Resize along lake dimension
                                main_var.resize(current_lake_count + new_lake_count, num_dates)
                                # Write data
                                main_var[current_lake_count:, :] = var_data
                            elif len(main_var.dimensions) == 1:  # Just lakes (id_geohash)
                                main_var.resize(current_lake_count + new_lake_count)
                                main_var[current_lake_count:] = var_data
                            else:
                                logger.warning(f"Unexpected dimensions for {var_name}: {main_var.dimensions}")

            # Clean up
            del existing_chunk, chunk_final, all_dates_combined, new_data_by_date
            gc.collect()

        # Close existing dataset
        existing_ds.close()

        # Verify the output
        logger.info("Verifying output...")
        result_ds = xr.open_dataset(new_dynamic_world_data_file, decode_times=False)
        final_lake_count = len(result_ds.id_geohash)
        final_date_count = len(result_ds.date)
        result_ds.close()

        logger.info("=" * 60)
        logger.info(f"✓ MERGE COMPLETE!")
        logger.info(f"  Output: {new_dynamic_world_data_file}")
        logger.info(f"  Total lakes: {final_lake_count:,}")
        logger.info(f"  Total dates: {final_date_count}")
        logger.info(f"  New dates added: {len(new_dates)}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

    return new_dynamic_world_data_file


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()