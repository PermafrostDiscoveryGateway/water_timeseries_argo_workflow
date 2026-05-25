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

    # get latest dynamic world file
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    logger.info(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")

    # First, inspect the existing file to understand its structure
    logger.info("Inspecting existing NetCDF file structure...")
    with nc.Dataset(most_recent_dynamic_world_file, 'r') as inspect_ds:
        logger.info(f"Dimensions: {list(inspect_ds.dimensions.keys())}")
        logger.info(f"Variables: {list(inspect_ds.variables.keys())}")

        # Find the time/date dimension name
        time_dim_name = None
        for dim_name in inspect_ds.dimensions.keys():
            if dim_name in ['date', 'time', 't', 'datetime']:
                time_dim_name = dim_name
                break

        # If not found, look for a variable that seems to be time
        if time_dim_name is None:
            for var_name in inspect_ds.variables.keys():
                if var_name in ['date', 'time', 't', 'datetime']:
                    time_dim_name = var_name
                    break

        logger.info(f"Time dimension/variable name: {time_dim_name}")

        # Also check what the lake ID dimension is called
        lake_dim_name = None
        for dim_name in inspect_ds.dimensions.keys():
            if dim_name in ['id_geohash', 'lake_id', 'geohash', 'id']:
                lake_dim_name = dim_name
                break

        logger.info(f"Lake dimension name: {lake_dim_name}")

        # Get sample time values - handle if it's a dimension without a variable
        if time_dim_name:
            # Check if there's a variable with the same name as the dimension
            if time_dim_name in inspect_ds.variables:
                time_var = inspect_ds.variables[time_dim_name]
                logger.info(f"Time variable type: {time_var.dtype}")
                logger.info(f"Sample time values (first 3): {time_var[:3]}")

                # Try to convert to datetime
                if hasattr(time_var, 'units'):
                    logger.info(f"Time units: {time_var.units}")
                    # Convert first few times for inspection
                    sample_times = num2date(time_var[:3], time_var.units)
                    logger.info(f"Sample dates: {sample_times}")
            else:
                # The time dimension doesn't have a separate variable, use the dimension
                logger.info(f"Time dimension '{time_dim_name}' has no variable, using dimension values")
                # For netCDF4, dimensions don't store values directly, we need to use xarray for this
                logger.info("Will use xarray to read time values")

    # Now use xarray to get the actual time values
    logger.info("Loading time values with xarray...")
    existing_ds_temp = xr.open_dataset(most_recent_dynamic_world_file)

    # Find time dimension in existing data
    time_dim_in_existing = None
    for dim in existing_ds_temp.dims:
        if dim in ['date', 'time', 't', 'datetime']:
            time_dim_in_existing = dim
            break

    # If not found in dims, look for coordinate
    if time_dim_in_existing is None:
        for coord in existing_ds_temp.coords:
            if coord in ['date', 'time', 't', 'datetime']:
                time_dim_in_existing = coord
                break

    logger.info(f"Time dimension in existing data: {time_dim_in_existing}")

    # Get existing dates
    if time_dim_in_existing:
        existing_dates = set(pd.to_datetime(existing_ds_temp[time_dim_in_existing].values))
        logger.info(f"Found {len(existing_dates)} existing dates")
        logger.info(f"First 3 existing dates: {sorted(existing_dates)[:3]}")
    else:
        logger.error("Could not find time dimension in existing file")
        return None

    # Get lake IDs
    lake_dim_in_existing = None
    for dim in existing_ds_temp.dims:
        if dim in ['id_geohash', 'lake_id', 'geohash', 'id']:
            lake_dim_in_existing = dim
            break

    if lake_dim_in_existing:
        existing_lake_ids_set = set(existing_ds_temp[lake_dim_in_existing].values)
        logger.info(f"Found {len(existing_lake_ids_set):,} existing lakes")
    else:
        logger.error("Could not find lake dimension in existing file")
        return None

    existing_ds_temp.close()

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
                # Concatenate along id_geohash dimension
                merged_new_ds = xr.concat([merged_new_ds, ds_chunk], dim="id_geohash")
        except Exception as e:
            logger.error(f"Failed to load chunk {chunk_file}: {e}")
            continue

    if merged_new_ds is None:
        logger.error("No data to merge from chunks")
        return None

    # Remove duplicates in id_geohash if any
    original_lake_count = len(merged_new_ds.id_geohash)
    _, unique_indices = np.unique(merged_new_ds.id_geohash.values, return_index=True)
    merged_new_ds = merged_new_ds.isel(id_geohash=sorted(unique_indices))
    unique_lake_count = len(merged_new_ds.id_geohash)

    if original_lake_count != unique_lake_count:
        logger.warning(f"Removed {original_lake_count - unique_lake_count} duplicate lake entries from new data")

    # Get dates from new data
    time_dim_in_new = None
    for dim in merged_new_ds.dims:
        if dim in ['date', 'time', 't', 'datetime']:
            time_dim_in_new = dim
            break

    if time_dim_in_new is None:
        # Try to find a coordinate that looks like time
        for coord in merged_new_ds.coords:
            if coord in ['date', 'time', 't', 'datetime']:
                time_dim_in_new = coord
                break

    logger.info(f"Time dimension in new data: {time_dim_in_new}")

    dates = merged_new_ds[time_dim_in_new].values
    latest_date = max(dates)

    # Convert latest_date to string safely
    try:
        # If it's a datetime64 object
        latest_date_string = str(latest_date).split('T')[0].replace('-', '_')
    except:
        # If it's a different format
        latest_date_string = str(latest_date).replace('-', '_')

    logger.info(f"The latest date is {latest_date_string}")
    logger.info(f"Merged new data: {len(merged_new_ds.id_geohash):,} lakes x {len(dates)} dates")

    # Check for new lakes
    logger.info("=" * 60)
    logger.info("CHECKING FOR NEW LAKES (should be none)...")
    new_lake_ids_set = set(merged_new_ds.id_geohash.values)

    # Convert to string for comparison
    existing_lakes_str = set()
    for lake_id in existing_lake_ids_set:
        if isinstance(lake_id, bytes):
            existing_lakes_str.add(lake_id.decode('utf-8'))
        else:
            existing_lakes_str.add(str(lake_id))

    new_lakes_str = set()
    for lake_id in new_lake_ids_set:
        if isinstance(lake_id, bytes):
            new_lakes_str.add(lake_id.decode('utf-8'))
        else:
            new_lakes_str.add(str(lake_id))

    # Find new lakes
    lakes_only_in_new = new_lakes_str - existing_lakes_str
    lakes_only_in_existing = existing_lakes_str - new_lakes_str
    common_lakes = existing_lakes_str & new_lakes_str

    logger.info(f"Lakes in existing dataset: {len(existing_lakes_str):,}")
    logger.info(f"Lakes in new dataset: {len(new_lakes_str):,}")
    logger.info(f"Common lakes: {len(common_lakes):,}")

    if lakes_only_in_new:
        logger.error(f"⚠️ FOUND {len(lakes_only_in_new)} NEW LAKES in new data!")
        logger.error(f"This should NOT happen according to expectations!")
        logger.info(f"First 10 new lake IDs: {list(lakes_only_in_new)[:10]}")

        # Save to file for inspection
        new_lakes_file = f"new_lakes_detected_{latest_date_string}.txt"
        with open(new_lakes_file, 'w') as f:
            for lake_id in sorted(lakes_only_in_new):
                f.write(f"{lake_id}\n")
        logger.info(f"Full list of new lakes saved to: {new_lakes_file}")
    else:
        logger.info("✓ VERIFIED: No new lakes detected! All lakes in new data exist in existing dataset.")

    if lakes_only_in_existing:
        logger.info(
            f"Note: {len(lakes_only_in_existing):,} lakes exist only in existing dataset (expected - not all lakes have data every month)")

    logger.info("=" * 60)

    # Check for date overlap
    new_dates = set(pd.to_datetime(merged_new_ds[time_dim_in_new].values))
    overlapping_dates = existing_dates & new_dates

    if overlapping_dates:
        logger.warning(f"Found {len(overlapping_dates)} overlapping dates: {sorted(overlapping_dates)}")
        logger.warning("Removing overlapping dates from new dataset...")
        mask = ~merged_new_ds[time_dim_in_new].isin(list(overlapping_dates))
        merged_new_ds = merged_new_ds.sel({time_dim_in_new: mask})
        dates = merged_new_ds[time_dim_in_new].values

    if len(dates) == 0:
        logger.warning("No new dates to add after removing overlaps")
        return None

    logger.info(f"Adding {len(dates)} new dates: {dates}")

    # Now use netCDF4 for memory-efficient merging
    logger.info("Merging datasets using netCDF4 (memory efficient)...")

    # Create output filename
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Open existing file and new data for streaming merge
    with nc.Dataset(most_recent_dynamic_world_file, 'r') as src:
        # Use the dimension names we already identified
        src_lake_dim = lake_dim_in_existing
        src_time_dim = time_dim_in_existing

        src_lakes = len(src.dimensions[src_lake_dim])
        src_dates = len(src.dimensions[src_time_dim])
        new_dates_count = len(dates)

        logger.info(f"Source lake dimension: {src_lake_dim} ({src_lakes:,} lakes)")
        logger.info(f"Source time dimension: {src_time_dim} ({src_dates} dates)")
        logger.info(f"New: {len(merged_new_ds.id_geohash):,} lakes x {new_dates_count} dates")

        # Create output file
        with nc.Dataset(new_dynamic_world_data_file, 'w', format='NETCDF4') as dst:
            # Create dimensions (using same names as source)
            dst.createDimension(src_lake_dim, src_lakes)
            dst.createDimension(src_time_dim, src_dates + new_dates_count)

            # Copy global attributes
            for attr_name, attr_value in src.ncattrs():
                setattr(dst, attr_name, attr_value)
            dst.date_last_updated = str(datetime.now())

            # Create mapping from lake ID to index - need to get from xarray since netCDF4 might not have it as variable
            logger.info("Building lake ID mapping...")
            existing_lake_ids = existing_lake_ids_set  # Use the set we already have

            # But we need ordered list, so reload with xarray
            temp_ds = xr.open_dataset(most_recent_dynamic_world_file)
            existing_lake_ids_ordered = temp_ds[src_lake_dim].values
            temp_ds.close()

            lake_to_idx = {}
            for idx, lake_id in enumerate(existing_lake_ids_ordered):
                if isinstance(lake_id, bytes):
                    lake_to_idx[lake_id.decode('utf-8')] = idx
                else:
                    lake_to_idx[str(lake_id)] = idx

            # Copy all variables from source using xarray to get them
            temp_ds = xr.open_dataset(most_recent_dynamic_world_file)

            for var_name in temp_ds.data_vars:
                logger.info(f"Processing variable: {var_name}")
                src_var_xr = temp_ds[var_name]
                dims = list(src_var_xr.dims)

                # Get dtype and fill value
                dtype = src_var_xr.dtype
                fill_value = src_var_xr.encoding.get('_FillValue', np.nan)

                # Create variable in destination
                dst_var = dst.createVariable(
                    var_name,
                    dtype,
                    dims,
                    zlib=True,
                    complevel=5,
                    fill_value=fill_value if not np.isnan(fill_value) else None
                )

                # Copy attributes
                for attr_name in src_var_xr.attrs:
                    setattr(dst_var, attr_name, src_var_xr.attrs[attr_name])

                # Copy data based on variable type
                if src_time_dim in dims and src_lake_dim in dims:
                    # 2D variable (time x space)
                    logger.info(f"Copying existing {var_name} data...")

                    # Copy existing data in chunks using xarray values
                    chunk_size = 100000
                    for start_idx in range(0, src_lakes, chunk_size):
                        end_idx = min(start_idx + chunk_size, src_lakes)
                        src_data = src_var_xr.isel({src_lake_dim: slice(start_idx, end_idx)}).values
                        # Need to transpose if dimensions order is (time, lake) or (lake, time)
                        if src_var_xr.dims[0] == src_time_dim:
                            # Shape is (time, lake)
                            dst_var[:, start_idx:end_idx] = src_data
                        else:
                            # Shape is (lake, time)
                            dst_var[start_idx:end_idx, :] = src_data
                        logger.debug(f"  Copied lakes {start_idx:,} to {end_idx:,}")

                    # Add new date data
                    logger.info(f"Adding new date data for {var_name}...")
                    for date_idx, date_val in enumerate(dates):
                        target_date_idx = src_dates + date_idx

                        # Get data for this date from merged_new_ds
                        date_data = merged_new_ds.sel({time_dim_in_new: date_val})

                        # Initialize array for this time slice
                        new_time_slice = np.full(src_lakes, fill_value, dtype=dtype)

                        # Fill in data for existing lakes
                        lakes_filled = 0
                        for lake_id_str in new_lakes_str:
                            if lake_id_str in lake_to_idx:
                                lake_idx = lake_to_idx[lake_id_str]
                                # Find this lake in new data
                                new_lake_idx = np.where(new_lake_ids_set == lake_id_str)[0]
                                if len(new_lake_idx) > 0:
                                    var_data = date_data.isel(id_geohash=new_lake_idx[0])[var_name].values
                                    new_time_slice[lake_idx] = var_data
                                    lakes_filled += 1

                        # Write the time slice - handle dimension order
                        if src_var_xr.dims[0] == src_time_dim:
                            dst_var[target_date_idx, :] = new_time_slice
                        else:
                            dst_var[:, target_date_idx] = new_time_slice

                        if (date_idx + 1) % 10 == 0:
                            logger.info(f"  Processed {date_idx + 1}/{new_dates_count} dates")

                elif src_time_dim in dims:
                    # 1D time variable
                    dst_var[:src_dates] = src_var_xr.values
                    for date_idx, date_val in enumerate(dates):
                        dst_var[src_dates + date_idx] = date_val

                else:
                    # Other variables (lake IDs, etc.)
                    dst_var[:] = src_var_xr.values

            temp_ds.close()

    logger.info("=" * 60)
    logger.info(f"Successfully merged datasets using netCDF4!")
    logger.info(f"Merged file saved as: {new_dynamic_world_data_file}")

    # Verification
    with nc.Dataset(new_dynamic_world_data_file, 'r') as verify_ds:
        final_lakes = len(verify_ds.dimensions[src_lake_dim])
        final_dates = len(verify_ds.dimensions[src_time_dim])
        logger.info(f"Final dataset: {final_lakes:,} lakes x {final_dates} dates")

    if lakes_only_in_new:
        logger.warning(f"SUMMARY: {len(lakes_only_in_new)} new lakes were detected but NOT added")
    else:
        logger.info("✓ SUMMARY: No new lakes detected - all good!")

    logger.info("Memory-efficient merge complete!")
    merged_new_ds.close()


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()