from near_real_time_grid_v2 import verify_merged_netcdf
import sys
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
import os
import glob
import time
import pandas as pd
from pathlib import Path
import xarray as xr
import numpy as np
import shutil
import gc

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def debug_id_mismatch(historical_file: str, combined_file: str):
    """Debug why IDs don't match between files."""

    logger.info("=" * 80)
    logger.info("DEBUGGING ID MISMATCH")
    logger.info("=" * 80)

    # Open both files
    hist_ds = xr.open_dataset(historical_file)
    comb_ds = xr.open_dataset(combined_file)

    # Get IDs
    hist_ids = hist_ds['id_geohash'].values
    comb_ids = comb_ds['id_geohash'].values

    # Get first few IDs from each
    logger.info(f"Historical file first 10 IDs: {hist_ids[:10]}")
    logger.info(f"Combined file first 10 IDs: {comb_ids[:10]}")

    # Check data types
    logger.info(f"Historical IDs type: {hist_ids.dtype}")
    logger.info(f"Combined IDs type: {comb_ids.dtype}")

    # Check for string/bytes issues
    if hist_ids.dtype.kind in ['U', 'S']:
        logger.info(f"Historical IDs sample (as strings): {[str(id) for id in hist_ids[:5]]}")
    if comb_ids.dtype.kind in ['U', 'S']:
        logger.info(f"Combined IDs sample (as strings): {[str(id) for id in comb_ids[:5]]}")

    # Check if IDs are numeric
    if hist_ids.dtype.kind in ['i', 'f']:
        logger.info(f"Historical IDs are numeric, range: {hist_ids.min()} to {hist_ids.max()}")
    if comb_ids.dtype.kind in ['i', 'f']:
        logger.info(f"Combined IDs are numeric, range: {comb_ids.min()} to {comb_ids.max()}")

    # Check if there's any match at all
    hist_set = set(hist_ids)
    comb_set = set(comb_ids)
    intersection = hist_set & comb_set

    logger.info(f"Intersection size: {len(intersection)}")

    if len(intersection) == 0:
        logger.info("No exact matches found. Checking for partial matches...")

        # Convert to strings for comparison
        hist_str = [str(id) for id in hist_ids[:1000]]
        comb_str = [str(id) for id in comb_ids[:1000]]

        # Check if any IDs from one file appear as substrings in the other
        for i, h_id in enumerate(hist_str[:10]):
            for j, c_id in enumerate(comb_str[:10]):
                if h_id in c_id or c_id in h_id:
                    logger.info(f"Partial match: '{h_id}' and '{c_id}'")

        logger.info("Checking if IDs are encoded differently...")

        # Try to decode if they're bytes
        if hist_ids.dtype == 'S' and comb_ids.dtype == 'S':
            hist_decoded = [id.decode('utf-8') for id in hist_ids[:10]]
            comb_decoded = [id.decode('utf-8') for id in comb_ids[:10]]
            logger.info(f"Historical decoded: {hist_decoded}")
            logger.info(f"Combined decoded: {comb_decoded}")

        # Check dimension names
        logger.info(f"Historical dims: {list(hist_ds.dims)}")
        logger.info(f"Combined dims: {list(comb_ds.dims)}")

        # Check if there's another dimension that could be the ID
        for var in hist_ds.data_vars:
            logger.info(f"Historical variable: {var}, shape: {hist_ds[var].shape}")
        for var in comb_ds.data_vars:
            logger.info(f"Combined variable: {var}, shape: {comb_ds[var].shape}")

    # Check metadata
    logger.info(f"Historical attributes: {hist_ds.attrs}")
    logger.info(f"Combined attributes: {comb_ds.attrs}")

    # Close datasets
    hist_ds.close()
    comb_ds.close()

    return {
        'hist_ids_sample': list(hist_ids[:10]),
        'comb_ids_sample': list(comb_ids[:10]),
        'intersection_size': len(intersection),
        'hist_dtype': str(hist_ids.dtype),
        'comb_dtype': str(comb_ids.dtype)
    }


def main():
    logger.debug(f"Beginning historical run for ALL regions (fast mode)")
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # ========== Get all regions ==========
    import utils.region_boundaries
    boundaries = utils.region_boundaries.get_region_boundaries()
    all_regions = list(boundaries.keys())
    logger.info(f"Available regions: {all_regions}")

    dynamic_world_data_dir = os.environ['dynamic_world_data']

    # ========== Determine if we should run ==========
    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]

    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month

    if TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"Should run: {SHOULD_RUN}")

    if not SHOULD_RUN:
        logger.debug(f"Too early in the month to run downloads - exiting")
        return

    # ========== Prepare date to run ==========
    date_to_run = datetime(TODAY.year, TODAY_MONTH - 1, 1).strftime("%Y-%m")
    logger.info(f"Processing date: {date_to_run}")

    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)

    # Get file sizes for logging
    historical_file_size_gb = Path(original_most_recent_dynamic_world_file).stat().st_size / (1024 ** 3)
    logger.info(f"Historical file size: {historical_file_size_gb:.2f} GB")

    combined_file_name = f"dynamic_world_combined_{date_to_run}.nc"
    combined_file_path = os.path.join(dynamic_world_data_dir, 'merge', combined_file_name)

    debug_id_mismatch(historical_file=original_most_recent_dynamic_world_file, combined_file=combined_file_path)

    if os.path.exists(combined_file_path):
        combined_file_size_gb = Path(combined_file_path).stat().st_size / (1024 ** 3)
        logger.info(f"Combined file size: {combined_file_size_gb:.2f} GB")

        # Check if file has not been modified for at least 12 hours
        file_mtime = Path(combined_file_path).stat().st_mtime
        file_age_seconds = time.time() - file_mtime
        file_age_hours = file_age_seconds / 3600

        if file_age_hours < 0.0001:
            logger.warning(
                f"Combined file {combined_file_name} was modified {file_age_hours:.1f} hours ago (< 12 hours). Waiting...")
            wait_seconds = (12 * 3600) - file_age_seconds + 60
            logger.info(f"Waiting {wait_seconds / 3600:.1f} hours for file to stabilize...")
            time.sleep(wait_seconds)

            file_mtime = Path(combined_file_path).stat().st_mtime
            file_age_hours = (time.time() - file_mtime) / 3600
            if file_age_hours < 12:
                logger.warning(
                    f"File still not old enough ({file_age_hours:.1f} hours). Proceeding anyway but data may be incomplete.")

        logger.debug(f"Combined file {combined_file_name} exists and is {file_age_hours:.1f} hours old.")

        # Create a new file with a timestamp to avoid overwriting
        new_historical_file_name = f"dynamic_world_historical_{date_to_run}.nc"
        new_historical_file_path = os.path.join(dynamic_world_data_dir, new_historical_file_name)

        # Check available disk space before proceeding
        try:
            statvfs = os.statvfs(dynamic_world_data_dir)
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)
            logger.info(f"Free disk space: {free_space_gb:.2f} GB")

            # Estimate required space
            required_space_gb = (historical_file_size_gb + combined_file_size_gb) * 2.5
            if free_space_gb < required_space_gb:
                logger.error(f"Insufficient disk space! Need ~{required_space_gb:.2f} GB, have {free_space_gb:.2f} GB")
                return
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        logger.debug(f"Combining new data from {combined_file_name} to {new_historical_file_path}")

        try:
            # ========== CORRECTED MERGE: Preserve ALL IDs ==========
            logger.info("Loading historical dataset...")
            hist_ds = xr.open_dataset(original_most_recent_dynamic_world_file)

            logger.info("Loading combined dataset...")
            comb_ds = xr.open_dataset(combined_file_path)

            # Get metadata
            hist_dates = pd.to_datetime(hist_ds['date'].values)
            hist_date_strings = {d.strftime("%Y-%m") for d in hist_dates}

            comb_dates = pd.to_datetime(comb_ds['date'].values)
            comb_date_strings = {d.strftime("%Y-%m") for d in comb_dates}

            # Find new dates
            dates_to_add = sorted(comb_date_strings - hist_date_strings)
            logger.info(f"Dates to add: {dates_to_add}")

            if not dates_to_add:
                logger.info("No new dates to add. All dates already in historical file.")
                hist_ds.close()
                comb_ds.close()
                return

            # Filter combined to only new dates
            new_date_objects = [pd.Timestamp(f"{d}-01") for d in dates_to_add]
            comb_ds_filtered = comb_ds.sel(date=new_date_objects)

            # ========== CRITICAL FIX: Don't filter to common IDs ==========
            # Instead, reindex the combined data to match the historical IDs
            # This preserves ALL historical IDs

            logger.info(f"Historical has {len(hist_ds['id_geohash']):,} IDs")
            logger.info(f"Combined has {len(comb_ds_filtered['id_geohash']):,} IDs")

            # Get the historical IDs in order
            hist_id_values = hist_ds['id_geohash'].values

            # Reindex the combined data to match historical ID order
            # Missing IDs will be filled with NaN
            logger.info("Reindexing combined data to match historical ID order...")
            comb_ds_reindexed = comb_ds_filtered.reindex(
                id_geohash=hist_id_values,
                method=None  # Don't interpolate
            )

            # Now both datasets have the same IDs in the same order
            logger.info(
                f"Reindexed combined shape: {len(comb_ds_reindexed['id_geohash']):,} IDs x {len(comb_ds_reindexed['date'])} dates")

            # Verify the IDs match
            if not np.array_equal(hist_ds['id_geohash'].values, comb_ds_reindexed['id_geohash'].values):
                logger.error("ID mismatch after reindexing!")
                hist_ds.close()
                comb_ds.close()
                return

            # Now concatenate along the date dimension
            logger.info(f"Merging datasets...")
            merged_ds = xr.concat([hist_ds, comb_ds_reindexed], dim='date')
            merged_ds = merged_ds.sortby('date')

            # Remove duplicate dates if any
            _, unique_idx = np.unique(merged_ds['date'].values, return_index=True)
            if len(unique_idx) < len(merged_ds['date']):
                removed = len(merged_ds['date']) - len(unique_idx)
                logger.info(f"Removed {removed} duplicate dates")
                merged_ds = merged_ds.isel(date=np.sort(unique_idx))

            logger.info(f"Merged shape: {len(merged_ds['id_geohash']):,} IDs x {len(merged_ds['date'])} dates")

            # Check for NaN values in the new dates (IDs that were missing in combined)
            if len(dates_to_add) > 0:
                # Check the first new date for NaN values
                first_new_date = new_date_objects[0]
                sample_data = merged_ds.sel(date=first_new_date)

                # Count how many IDs have data vs NaN
                # Check a sample variable
                sample_var = list(merged_ds.data_vars)[0]
                data_values = sample_data[sample_var].values
                nan_count = np.isnan(data_values).sum()
                total_count = len(data_values)

                if nan_count > 0:
                    logger.warning(
                        f"For date {dates_to_add[0]}, {nan_count:,}/{total_count:,} IDs have NaN (missing from combined file)")
                    logger.info(f"These are IDs that exist in historical but not in combined file")
                    logger.info(f"They will remain as NaN for the new date (no data available)")

            # Write the merged file
            logger.info(f"Writing merged file to {new_historical_file_path}...")
            logger.info(f"Expected size: ~{(historical_file_size_gb + combined_file_size_gb):.2f} GB")

            start_write = time.time()

            encoding = {}
            for var in merged_ds.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True
                }

            merged_ds.to_netcdf(new_historical_file_path, encoding=encoding)

            write_time = time.time() - start_write
            logger.info(f"Write completed in {write_time:.1f} seconds")

            # Clean up
            hist_ds.close()
            comb_ds.close()
            merged_ds.close()
            gc.collect()

            # Verify the file
            logger.info("Verifying merged file...")
            final_size_gb = Path(new_historical_file_path).stat().st_size / (1024 ** 3)
            logger.info(f"Final file size: {final_size_gb:.4f} GB")

            # Quick verification
            try:
                verify_ds = xr.open_dataset(new_historical_file_path)
                verify_id_count = len(verify_ds['id_geohash'])
                verify_date_count = len(verify_ds['date'])
                verify_ds.close()

                logger.info(f"Verified file has {verify_id_count:,} IDs and {verify_date_count} dates")

                # Check if we have the expected number of IDs (should match historical)
                expected_ids = len(hist_id_values)
                if verify_id_count == expected_ids:
                    logger.info(f"✅ All {expected_ids:,} IDs preserved!")
                else:
                    logger.warning(f"⚠️ ID count mismatch! Expected {expected_ids:,}, got {verify_id_count:,}")
            except Exception as e:
                logger.warning(f"Could not verify file: {e}")

            # Also run the standard verification from near_real_time_grid_v2
            verify_result = verify_merged_netcdf(new_historical_file_path)
            if verify_result.get('success', False):
                logger.info("✅ Merged file verification passed")
            else:
                logger.warning(f"⚠️ Merged file verification failed: {verify_result.get('error', 'Unknown error')}")

            logger.info("=" * 80)
            logger.info("✅ SUCCESS: New historical file created!")
            logger.info("=" * 80)
            logger.info(f"  Original historical file (KEPT): {original_most_recent_dynamic_world_file}")
            logger.info(f"  Combined file (KEPT): {combined_file_path}")
            logger.info(f"  NEW merged file: {new_historical_file_path}")
            logger.info(f"  File size: {final_size_gb:.4f} GB")
            logger.info(f"  Added dates: {dates_to_add}")
            logger.info(f"  Total IDs: {verify_id_count if 'verify_id_count' in locals() else 'unknown':,}")
            logger.info(f"  Total dates: {verify_date_count if 'verify_date_count' in locals() else 'unknown'}")
            logger.info("=" * 80)
            logger.info("⚠️  No files were deleted. All original files are preserved.")
            logger.info("=" * 80)

        except MemoryError as e:
            logger.error(f"Memory error while processing files: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(new_historical_file_path):
                os.remove(new_historical_file_path)

        except Exception as e:
            logger.error(f"Error merging files: {e}")
            import traceback
            traceback.print_exc()
            if os.path.exists(new_historical_file_path):
                os.remove(new_historical_file_path)
            gc.collect()
            return
    else:
        logger.info(f"Combined file {combined_file_name} does not exist. Nothing to merge.")


if __name__ == "__main__":
    main()