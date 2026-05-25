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

        # Get sample time values
        if time_dim_name:
            time_var = inspect_ds.variables[time_dim_name]
            logger.info(f"Time variable type: {time_var.dtype}")
            logger.info(f"Sample time values (first 3): {time_var[:3]}")

            # Try to convert to datetime
            if hasattr(time_var, 'units'):
                logger.info(f"Time units: {time_var.units}")
                # Convert first few times for inspection
                from netCDF4 import num2date
                sample_times = num2date(time_var[:3], time_var.units)
                logger.info(f"Sample dates: {sample_times}")

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

    # Get dates from new data - determine what the time dimension is called
    time_dim_in_new = None
    for dim in merged_new_ds.dims:
        if dim in ['date', 'time', 't']:
            time_dim_in_new = dim
            break

    if time_dim_in_new is None:
        # Try to find a coordinate that looks like time
        for coord in merged_new_ds.coords:
            if coord in ['date', 'time', 't']:
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

    # Now inspect existing file with xarray to get dates and lake IDs
    logger.info("Loading existing dataset metadata...")
    existing_ds_temp = xr.open_dataset(most_recent_dynamic_world_file)

    # Find time dimension in existing data
    time_dim_in_existing = None
    for dim in existing_ds_temp.dims:
        if dim in ['date', 'time', 't']:
            time_dim_in_existing = dim
            break

    if time_dim_in_existing is None:
        # Look for coordinate
        for coord in existing_ds_temp.coords:
            if coord in ['date', 'time', 't']:
                time_dim_in_existing = coord
                break

    logger.info(f"Time dimension in existing data: {time_dim_in_existing}")

    # Get existing dates
    existing_dates = set(pd.to_datetime(existing_ds_temp[time_dim_in_existing].values))

    # Get lake IDs
    lake_dim_in_existing = None
    for dim in existing_ds_temp.dims:
        if dim in ['id_geohash', 'lake_id', 'geohash', 'id']:
            lake_dim_in_existing = dim
            break

    existing_lake_ids_set = set(existing_ds_temp[lake_dim_in_existing].values)
    existing_ds_temp.close()

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
        # Get dimensions from source - use the actual dimension names
        src_lake_dim = None
        src_time_dim = None

        for dim_name in src.dimensions.keys():
            if dim_name in ['id_geohash', 'lake_id', 'geohash', 'id']:
                src_lake_dim = dim_name
            if dim_name in ['date', 'time', 't']:
                src_time_dim = dim_name

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

            # Create mapping from lake ID to index
            existing_lake_ids = src.variables[src_lake_dim][:]
            lake_to_idx = {}
            for idx, lake_id in enumerate(existing_lake_ids):
                if isinstance(lake_id, bytes):
                    lake_to_idx[lake_id.decode('utf-8')] = idx
                else:
                    lake_to_idx[str(lake_id)] = idx

            # Copy all variables
            for var_name in src.variables:
                src_var = src.variables[var_name]
                dims = list(src_var.dimensions)

                # Create variable in destination
                dst_var = dst.createVariable(
                    var_name,
                    src_var.dtype,
                    dims,
                    zlib=True,
                    complevel=5,
                    fill_value=src_var._FillValue if hasattr(src_var, '_FillValue') else None
                )

                # Copy attributes
                for attr_name in src_var.ncattrs():
                    if attr_name not in ['_FillValue', 'least_significant_digit']:
                        setattr(dst_var, attr_name, getattr(src_var, attr_name))

                # Copy data based on variable type
                if src_time_dim in dims and src_lake_dim in dims:
                    # 2D variable (time x space)
                    logger.info(f"Copying existing {var_name} data...")

                    # Copy existing data in chunks
                    chunk_size = 100000
                    for start_idx in range(0, src_lakes, chunk_size):
                        end_idx = min(start_idx + chunk_size, src_lakes)
                        src_data = src_var[:, start_idx:end_idx]
                        dst_var[:, start_idx:end_idx] = src_data
                        logger.debug(f"  Copied lakes {start_idx:,} to {end_idx:,}")

                    # Add new date data
                    logger.info(f"Adding new date data for {var_name}...")
                    for date_idx, date_val in enumerate(dates):
                        target_date_idx = src_dates + date_idx

                        # Get data for this date
                        date_data = merged_new_ds.sel({time_dim_in_new: date_val})

                        # Initialize array for this time slice
                        fill_value = src_var._FillValue if hasattr(src_var, '_FillValue') else np.nan
                        new_time_slice = np.full(src_lakes, fill_value, dtype=src_var.dtype)

                        # Fill in data for existing lakes
                        lakes_filled = 0
                        for lake_idx, lake_id in enumerate(existing_lake_ids):
                            lake_id_str = lake_id.decode('utf-8') if isinstance(lake_id, bytes) else str(lake_id)
                            if lake_id_str in new_lakes_str:
                                new_lake_idx = np.where(new_lake_ids == lake_id_str)[0]
                                if len(new_lake_idx) > 0:
                                    var_data = date_data.isel(id_geohash=new_lake_idx[0])[var_name].values
                                    new_time_slice[lake_idx] = var_data
                                    lakes_filled += 1

                        dst_var[target_date_idx, :] = new_time_slice

                        if (date_idx + 1) % 10 == 0:
                            logger.info(f"  Processed {date_idx + 1}/{new_dates_count} dates")

                elif src_time_dim in dims:
                    # 1D time variable
                    dst_var[:src_dates] = src_var[:]
                    for date_idx, date_val in enumerate(dates):
                        dst_var[src_dates + date_idx] = date_val

                else:
                    # Other variables (lake IDs, etc.)
                    dst_var[:] = src_var[:]

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