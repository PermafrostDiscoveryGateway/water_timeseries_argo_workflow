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

    # Find the latest valid dynamic world file (skip empty/corrupted ones)
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_dir, "*.nc"))
    valid_files = []
    for f in all_dynamic_world_files:
        if os.path.getsize(f) > 1024 * 1024:  # Larger than 1 MB
            valid_files.append(f)
        else:
            logger.warning(f"Skipping empty/corrupted file: {f}")

    if not valid_files:
        logger.error("No valid existing dynamic world files found!")
        return None

    most_recent_dynamic_world_file = max(valid_files, key=os.path.getctime)
    logger.info(f"Most recent dynamic world file: {most_recent_dynamic_world_file}")

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
    time_dim_in_new = 'date'  # Based on your file structure
    dates = merged_new_ds[time_dim_in_new].values
    latest_date = max(dates)

    # Convert latest_date to string safely
    try:
        latest_date_string = str(latest_date).split('T')[0].replace('-', '_')
    except:
        latest_date_string = str(latest_date).replace('-', '_')

    logger.info(f"The latest date is {latest_date_string}")
    logger.info(f"Merged new data: {len(merged_new_ds.id_geohash):,} lakes x {len(dates)} dates")

    # Load existing dataset
    logger.info("Loading existing dataset...")
    existing_ds = xr.open_dataset(most_recent_dynamic_world_file)

    # Get existing dates
    existing_dates = set(pd.to_datetime(existing_ds.date.values))
    logger.info(f"Existing dates: {len(existing_dates)} dates (from {min(existing_dates)} to {max(existing_dates)})")

    # Get existing lake IDs
    existing_lake_ids = existing_ds.id_geohash.values
    existing_lake_ids_set = set(str(lake_id) for lake_id in existing_lake_ids)
    logger.info(f"Existing lakes: {len(existing_lake_ids_set):,}")

    # Get new lake IDs
    new_lake_ids = merged_new_ds.id_geohash.values
    new_lake_ids_set = set(str(lake_id) for lake_id in new_lake_ids)

    # Find new lakes
    lakes_only_in_new = new_lake_ids_set - existing_lake_ids_set
    lakes_only_in_existing = existing_lake_ids_set - new_lake_ids_set
    common_lakes = existing_lake_ids_set & new_lake_ids_set

    logger.info("=" * 60)
    logger.info("LAKE COMPARISON SUMMARY:")
    logger.info(f"  Lakes in existing dataset: {len(existing_lake_ids_set):,}")
    logger.info(f"  Lakes in new dataset: {len(new_lake_ids_set):,}")
    logger.info(f"  Common lakes: {len(common_lakes):,}")
    logger.info(f"  New lakes to add: {len(lakes_only_in_new):,}")
    logger.info(f"  Lakes only in existing (no new data): {len(lakes_only_in_existing):,}")

    if lakes_only_in_new:
        logger.info(f"✓ Will add {len(lakes_only_in_new):,} new lakes to the dataset")
        # Save list of new lakes for reference
        new_lakes_file = f"new_lakes_added_{latest_date_string}.txt"
        with open(new_lakes_file, 'w') as f:
            for lake_id in sorted(lakes_only_in_new):
                f.write(f"{lake_id}\n")
        logger.info(f"New lakes list saved to: {new_lakes_file}")
    else:
        logger.info("No new lakes detected")

    # Check for date overlap
    new_dates = set(pd.to_datetime(merged_new_ds.date.values))
    overlapping_dates = existing_dates & new_dates

    if overlapping_dates:
        logger.warning(f"Found {len(overlapping_dates)} overlapping dates: {sorted(overlapping_dates)}")
        logger.warning("Removing overlapping dates from new dataset...")
        mask = ~merged_new_ds.date.isin(list(overlapping_dates))
        merged_new_ds = merged_new_ds.sel(date=mask)
        dates = merged_new_ds.date.values

    if len(dates) == 0:
        logger.warning("No new dates to add after removing overlaps")
        return None

    logger.info(f"Adding {len(dates)} new dates: {dates}")

    # Now merge the datasets, handling new lakes
    logger.info("=" * 60)
    logger.info("MERGING DATASETS (including new lakes)...")

    # Combine the lake IDs
    all_lake_ids = list(existing_lake_ids_set.union(lakes_only_in_new))
    logger.info(f"Total lakes after merge: {len(all_lake_ids):,}")

    # Create a mapping from lake ID to index in the merged dataset
    lake_id_to_idx = {lake_id: idx for idx, lake_id in enumerate(all_lake_ids)}

    # Get the list of variables to process (exclude coordinates)
    data_vars = [var for var in existing_ds.data_vars if var not in ['date', 'id_geohash']]
    logger.info(f"Variables to process: {data_vars}")

    # Create the merged dataset structure
    merged_dates = sorted(list(existing_dates.union(new_dates)))
    logger.info(f"Total dates after merge: {len(merged_dates)}")

    # Create a new dataset with the combined dimensions
    merged_data_vars = {}

    for var_name in data_vars:
        logger.info(f"Processing variable: {var_name}")

        # Initialize array with NaN
        merged_array = np.full((len(all_lake_ids), len(merged_dates)), np.nan, dtype=np.float64)

        # Fill in existing data
        logger.info(f"  Filling existing data for {var_name}...")
        existing_data = existing_ds[var_name].values  # Shape: (existing_lakes, existing_dates)

        for i, lake_id in enumerate(existing_lake_ids):
            lake_id_str = str(lake_id)
            if lake_id_str in lake_id_to_idx:
                target_lake_idx = lake_id_to_idx[lake_id_str]
                for j, date in enumerate(existing_ds.date.values):
                    date_idx = merged_dates.index(date)
                    merged_array[target_lake_idx, date_idx] = existing_data[i, j]

        # Fill in new data
        logger.info(f"  Filling new data for {var_name}...")
        new_data = merged_new_ds[var_name].values  # Shape: (new_lakes, new_dates)

        for i, lake_id in enumerate(new_lake_ids):
            lake_id_str = str(lake_id)
            if lake_id_str in lake_id_to_idx:
                target_lake_idx = lake_id_to_idx[lake_id_str]
                for j, date in enumerate(merged_new_ds.date.values):
                    if date in merged_dates:
                        date_idx = merged_dates.index(date)
                        merged_array[target_lake_idx, date_idx] = new_data[i, j]

        merged_data_vars[var_name] = merged_array

    # Create the merged xarray dataset
    logger.info("Creating final xarray dataset...")

    # Prepare coordinate arrays
    id_geohash_array = np.array(all_lake_ids, dtype=object)
    date_array = np.array(merged_dates, dtype='datetime64[ns]')

    # Create dataset
    final_ds = xr.Dataset(
        data_vars={
            var_name: (['id_geohash', 'date'], merged_data_vars[var_name])
            for var_name in data_vars
        },
        coords={
            'id_geohash': id_geohash_array,
            'date': date_array
        }
    )

    # Add attributes to match original file
    final_ds.attrs = {
        'description': 'This datasets provides the monthly area of the dynamic world classes (water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow_and_ice) for selected lake polygons. The areas were calculated from the Dynamic World V1 dataset through Google Earth Engine. Lake polygons were calculated by Ingmar Nitze through the Permafrost Discovery Gateway Project. "id_geohash" is the lake_id, which needs be joined to the accompanying polygon vector dataset',
        'author': 'Ingmar Nitze (Alfred Wegener Institute), Kayla Hardie (Google), Chen Wang (NCSA, U Illinois), Todd Nicholson(NCSA, U Illinois)',
        'contact': 'ingmar.nitze@awi.de',
        'date_merged': str(datetime.now())
    }

    # Add units attributes
    for var_name in data_vars:
        final_ds[var_name].attrs['units'] = 'ha'
        final_ds[var_name].attrs['_FillValue'] = np.nan

    # Sort by date
    final_ds = final_ds.sortby('date')

    # Create output filename
    new_dynamic_world_filename = f'lakes_dw_V2d_{latest_date_string}.nc'
    new_dynamic_world_data_file = os.path.join(dynamic_world_dir, new_dynamic_world_filename)

    # Save the final dataset with compression
    logger.info(f"Saving merged dataset to: {new_dynamic_world_data_file}")
    encoding = {var_name: {'zlib': True, 'complevel': 5} for var_name in data_vars}
    final_ds.to_netcdf(new_dynamic_world_data_file, encoding=encoding)

    logger.info("=" * 60)
    logger.info(f"✓ SUCCESSFULLY MERGED DATASETS!")
    logger.info(f"  Original file: {most_recent_dynamic_world_file}")
    logger.info(f"  New file: {new_dynamic_world_data_file}")
    logger.info(f"  Total lakes: {len(final_ds.id_geohash):,}")
    logger.info(f"  Total dates: {len(final_ds.date)}")
    logger.info(f"  New lakes added: {len(lakes_only_in_new):,}")
    logger.info(f"  New dates added: {len(dates)}")
    logger.info("=" * 60)

    # Clean up
    existing_ds.close()
    merged_new_ds.close()
    final_ds.close()


if __name__ == "__main__":
    combine_new_dynamic_world_data_with_latest()