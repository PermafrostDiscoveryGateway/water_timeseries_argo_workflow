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

    # COMPREHENSIVE INSPECTION of existing file
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE INSPECTION OF EXISTING FILE")
    logger.info("=" * 80)

    # Inspect with netCDF4
    with nc.Dataset(most_recent_dynamic_world_file, 'r') as inspect_ds:
        logger.info("\n--- NetCDF4 Inspection ---")
        logger.info(f"Dimensions: {list(inspect_ds.dimensions.keys())}")
        for dim_name, dim_obj in inspect_ds.dimensions.items():
            logger.info(f"  Dimension '{dim_name}': size = {len(dim_obj)}")

        logger.info(f"\nVariables: {list(inspect_ds.variables.keys())}")
        for var_name in inspect_ds.variables:
            var = inspect_ds.variables[var_name]
            logger.info(f"  Variable '{var_name}': dimensions={var.dimensions}, dtype={var.dtype}")
            if hasattr(var, 'units'):
                logger.info(f"    units: {var.units}")
            if hasattr(var, '_FillValue'):
                logger.info(f"    fill_value: {var._FillValue}")

        logger.info(f"\nGlobal attributes: {list(inspect_ds.ncattrs())}")
        for attr in inspect_ds.ncattrs():
            logger.info(f"  {attr}: {getattr(inspect_ds, attr)}")

    # Inspect with xarray
    logger.info("\n--- Xarray Inspection ---")
    existing_ds_temp = xr.open_dataset(most_recent_dynamic_world_file)

    logger.info(f"Dataset dimensions: {dict(existing_ds_temp.dims)}")
    logger.info(f"Dataset coordinates: {list(existing_ds_temp.coords)}")
    logger.info(f"Dataset data variables: {list(existing_ds_temp.data_vars)}")
    logger.info(f"Dataset attributes: {existing_ds_temp.attrs}")

    # Print detailed info about each coordinate
    for coord_name in existing_ds_temp.coords:
        coord = existing_ds_temp[coord_name]
        logger.info(f"\nCoordinate '{coord_name}':")
        logger.info(f"  dimensions: {coord.dims}")
        logger.info(f"  shape: {coord.shape}")
        logger.info(f"  dtype: {coord.dtype}")
        logger.info(f"  first 3 values: {coord.values[:3] if len(coord) > 0 else 'empty'}")

    # Print detailed info about each data variable
    for var_name in existing_ds_temp.data_vars:
        var = existing_ds_temp[var_name]
        logger.info(f"\nData variable '{var_name}':")
        logger.info(f"  dimensions: {var.dims}")
        logger.info(f"  shape: {var.shape}")
        logger.info(f"  dtype: {var.dtype}")
        if var.size > 0:
            logger.info(f"  sample values (first 3x3): {var.values[:3, :3] if len(var.shape) > 1 else var.values[:3]}")
        logger.info(f"  attributes: {var.attrs}")

    logger.info("=" * 80)

    # Now find the time dimension - look in coordinates first
    time_dim_in_existing = None

    # Check coordinates (these are typically the dimension variables)
    for coord in existing_ds_temp.coords:
        if coord in ['date', 'time', 't', 'datetime', 'time_coordinate']:
            time_dim_in_existing = coord
            logger.info(f"Found time coordinate: {coord}")
            break

    # If not found in coordinates, check data variables
    if time_dim_in_existing is None:
        for var in existing_ds_temp.data_vars:
            if var in ['date', 'time', 't', 'datetime']:
                time_dim_in_existing = var
                logger.info(f"Found time variable: {var}")
                break

    # If still not found, try to identify by dimension
    if time_dim_in_existing is None:
        for dim_name, dim_size in existing_ds_temp.dims.items():
            if dim_name in ['date', 'time', 't', 'datetime']:
                time_dim_in_existing = dim_name
                logger.info(f"Found time dimension: {dim_name}")
                break

    # Last resort: find a dimension with size > 1 that might be time
    if time_dim_in_existing is None:
        for dim_name, dim_size in existing_ds_temp.dims.items():
            if dim_size > 1 and dim_size < 1000:  # Time dimension likely small
                logger.warning(f"Possible time dimension: {dim_name} (size={dim_size})")
                time_dim_in_existing = dim_name
                break

    logger.info(f"Identified time dimension/coordinate: {time_dim_in_existing}")

    # Get existing dates if we found a time dimension
    if time_dim_in_existing:
        try:
            time_values = existing_ds_temp[time_dim_in_existing].values
            logger.info(f"Time values type: {type(time_values)}")
            logger.info(f"Time values shape: {time_values.shape if hasattr(time_values, 'shape') else 'scalar'}")
            logger.info(f"Raw time values (first 3): {time_values[:3] if len(time_values) > 0 else time_values}")

            # Try to convert to datetime
            existing_dates = set(pd.to_datetime(time_values))
            logger.info(f"Successfully converted {len(existing_dates)} dates")
            logger.info(f"First 3 existing dates: {sorted(existing_dates)[:3]}")
            logger.info(f"Last 3 existing dates: {sorted(existing_dates)[-3:]}")
        except Exception as e:
            logger.error(f"Failed to convert time values to datetime: {e}")
            logger.info("Attempting alternative conversion...")
            # Try using netCDF4 num2date if available
            with nc.Dataset(most_recent_dynamic_world_file, 'r') as src:
                if time_dim_in_existing in src.variables:
                    time_var = src.variables[time_dim_in_existing]
                    if hasattr(time_var, 'units'):
                        time_values = src.variables[time_dim_in_existing][:]
                        existing_dates = set(num2date(time_values, time_var.units))
                        logger.info(f"Converted using num2date: {len(existing_dates)} dates")
                    else:
                        raise ValueError("No units attribute")
                else:
                    raise ValueError(f"{time_dim_in_existing} not in variables")
    else:
        logger.error("Could not find time dimension in existing file")
        logger.info("Available coordinates and variables:")
        logger.info(f"  Coordinates: {list(existing_ds_temp.coords)}")
        logger.info(f"  Data variables: {list(existing_ds_temp.data_vars)}")
        logger.info(f"  Dimensions: {list(existing_ds_temp.dims)}")
        existing_ds_temp.close()
        return None

    # Get lake IDs
    lake_dim_in_existing = None
    for dim in existing_ds_temp.dims:
        if dim in ['id_geohash', 'lake_id', 'geohash', 'id', 'lake_index']:
            lake_dim_in_existing = dim
            break

    if lake_dim_in_existing:
        existing_lake_ids_set = set(existing_ds_temp[lake_dim_in_existing].values)
        logger.info(f"Found {len(existing_lake_ids_set):,} existing lakes")
        logger.info(f"First 5 lake IDs: {list(existing_lake_ids_set)[:5]}")
    else:
        logger.error("Could not find lake dimension in existing file")
        logger.info(f"Available dimensions: {list(existing_ds_temp.dims)}")
        existing_ds_temp.close()
        return None

    existing_ds_temp.close()

    # Get downloaded chunk files
    downloaded_files = glob.glob(os.path.join(split_new_dynamic_world_data_dir, "*.nc"))
    logger.info(f"\nFound {len(downloaded_files)} chunk files")

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

    # TODO: Continue with netCDF4 merge if we get past this point
    logger.info("Proceeding with merge... (continue from here)")

    # Clean up
    merged_new_ds.close()


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()