from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region, \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
    merge_near_real_time_region_v3_simple, \
    compare_netcdf_files, verify_merged_netcdf, verify_merged_data, merge_new_results, is_all_new_data_in_file
import sys
import utils.download_new_dynamic_world_data as download_new_dynamic_world_data
from loguru import logger
from datetime import date, datetime
from dotenv import load_dotenv
import os
import glob
import time
import pandas as pd
import utils.region_boundaries
from pathlib import Path
import xarray as xr
import numpy as np
import shutil
import gc
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# # =============================================================================
# # IMPLEMENTATION OF MERGE_NEW_RESULTS
# # =============================================================================
# def merge_new_results(
#         region: str = 'TEST',
#         date_to_merge: str = None,
#         merged_file_path: str = None,
#         env_path: str = None
# ) -> Dict[str, Any]:
#     """
#     Merge new downloaded results for a single date into a NetCDF file.
#
#     This function:
#     1. Finds all downloaded files for the specified date
#     2. Combines them into a single dataset
#     3. Saves the combined data to the specified file path
#
#     Args:
#         region: Region name (e.g., "TEST", "AFRICA")
#         date_to_merge: Date in "YYYY-MM" format
#         merged_file_path: Path where the merged file should be saved
#         env_path: Optional path to .env file
#
#     Returns:
#         dict: Result with status, file path, and statistics
#     """
#     logger.debug(f"Merging new results for {region} and {date_to_merge} into file {merged_file_path}")
#
#     # Load environment
#     if env_path:
#         load_dotenv(dotenv_path=env_path)
#     else:
#         load_dotenv()
#
#     # Validate inputs
#     if date_to_merge is None:
#         logger.error("date_to_merge is required")
#         return {'success': False, 'error': 'date_to_merge is required'}
#
#     if merged_file_path is None:
#         dynamic_world_data_dir = os.environ.get('dynamic_world_data')
#         if not dynamic_world_data_dir:
#             logger.error("dynamic_world_data not set in environment")
#             return {'success': False, 'error': 'dynamic_world_data not set'}
#         merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_merge}.nc")
#
#     # Ensure directory exists
#     Path(merged_file_path).parent.mkdir(parents=True, exist_ok=True)
#
#     # Find downloaded files for this date
#     dynamic_world_download_dir = Path(os.environ.get('dynamic_world_downloads', ''))
#     download_dir = dynamic_world_download_dir / region / f'download_{date_to_merge}'
#
#     if not download_dir.exists():
#         logger.warning(f"Download directory does not exist: {download_dir}")
#         return {'success': False, 'error': f'Download directory not found: {download_dir}'}
#
#     # Get all downloaded NetCDF files for this date
#     downloaded_files = sorted(glob.glob(str(download_dir / f'DW_{date_to_merge}_*.nc')))
#
#     if not downloaded_files:
#         logger.warning(f"No downloaded files found for {date_to_merge} in {download_dir}")
#         return {'success': False, 'error': f'No downloaded files found for {date_to_merge}'}
#
#     logger.info(f"Found {len(downloaded_files)} downloaded files for {date_to_merge}")
#
#     try:
#         # Combine all downloaded files into a single dataset
#         logger.info("Combining downloaded files...")
#         combined = None
#         failed_files = []
#
#         for nc_file in downloaded_files:
#             try:
#                 ds = xr.open_dataset(nc_file)
#                 if len(ds['id_geohash']) > 0:
#                     if combined is None:
#                         combined = ds
#                     else:
#                         # Concatenate along id_geohash dimension
#                         combined = xr.concat([combined, ds], dim='id_geohash')
#                         # Remove duplicate IDs
#                         _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
#                         if len(unique_idx) < len(combined['id_geohash']):
#                             combined = combined.isel(id_geohash=np.sort(unique_idx))
#                 else:
#                     logger.warning(f"File {nc_file} has no IDs, skipping")
#                     failed_files.append(nc_file)
#             except Exception as e:
#                 logger.error(f"Error opening {nc_file}: {e}")
#                 failed_files.append(nc_file)
#
#         if combined is None:
#             logger.error("No valid data to merge")
#             return {'success': False, 'error': 'No valid data to merge'}
#
#         logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")
#
#         # Ensure date dimension is correct
#         if len(combined['date']) > 0:
#             # Sort by date
#             combined = combined.sortby('date')
#
#         # Write to NetCDF with compression
#         logger.info(f"Writing merged data to {merged_file_path}")
#
#         encoding = {}
#         for var in combined.data_vars:
#             encoding[var] = {
#                 'zlib': True,
#                 'complevel': 4,
#                 'shuffle': True
#             }
#
#         # Write to file
#         combined.to_netcdf(merged_file_path, encoding=encoding)
#
#         # Get file size
#         file_size_gb = Path(merged_file_path).stat().st_size / (1024 ** 3)
#
#         # Clean up
#         combined.close()
#         gc.collect()
#
#         result = {
#             'success': True,
#             'file_path': merged_file_path,
#             'id_count': len(combined['id_geohash']),
#             'date_count': len(combined['date']),
#             'file_size_gb': round(file_size_gb, 4),
#             'files_merged': len(downloaded_files) - len(failed_files),
#             'files_failed': len(failed_files),
#             'failed_files': failed_files if failed_files else None,
#             'region': region,
#             'date': date_to_merge
#         }
#
#         logger.info(f"✅ Merge completed successfully!")
#         logger.info(f"  File: {merged_file_path}")
#         logger.info(f"  IDs: {result['id_count']:,}")
#         logger.info(f"  Dates: {result['date_count']}")
#         logger.info(f"  Size: {result['file_size_gb']:.4f} GB")
#
#         return result
#
#     except Exception as e:
#         logger.error(f"Error during merge: {e}")
#         import traceback
#         traceback.print_exc()
#         return {'success': False, 'error': str(e)}
#
#
# # =============================================================================
# # IMPLEMENTATION OF IS_ALL_NEW_DATA_IN_FILE
# # =============================================================================
# def is_all_new_data_in_file(
#         region: str = 'TEST',
#         date_to_check: str = None,
#         merged_file_path: str = None,
#         env_path: str = None
# ) -> Dict[str, Any]:
#     """
#     Check if all expected data for a date is present in the merged file.
#
#     This function verifies that:
#     1. The merged file exists and is valid
#     2. The file contains the expected date
#     3. All IDs from the region have data for that date
#
#     Args:
#         region: Region name
#         date_to_check: Date in "YYYY-MM" format
#         merged_file_path: Path to the merged NetCDF file
#         env_path: Optional path to .env file
#
#     Returns:
#         dict: Verification results with status and details
#     """
#     logger.debug(f"Checking if all new data for {region} and {date_to_check} is in {merged_file_path}")
#
#     # Load environment
#     if env_path:
#         load_dotenv(dotenv_path=env_path)
#     else:
#         load_dotenv()
#
#     # Validate inputs
#     if date_to_check is None:
#         logger.error("date_to_check is required")
#         return {'success': False, 'error': 'date_to_check is required'}
#
#     if merged_file_path is None:
#         dynamic_world_data_dir = os.environ.get('dynamic_world_data')
#         if not dynamic_world_data_dir:
#             logger.error("dynamic_world_data not set in environment")
#             return {'success': False, 'error': 'dynamic_world_data not set'}
#         merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_check}.nc")
#
#     # Check if file exists
#     if not Path(merged_file_path).exists():
#         logger.error(f"Merged file does not exist: {merged_file_path}")
#         return {
#             'success': False,
#             'error': 'File not found',
#             'file_exists': False,
#             'file_path': merged_file_path
#         }
#
#     try:
#         # Open the merged file
#         logger.info(f"Opening merged file: {merged_file_path}")
#         ds = xr.open_dataset(merged_file_path)
#
#         # Check if the date exists in the file
#         dates_in_file = pd.to_datetime(ds['date'].values)
#         date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
#
#         date_present = date_to_check in date_strings
#
#         if not date_present:
#             logger.warning(f"Date {date_to_check} not found in merged file")
#             ds.close()
#             return {
#                 'success': False,
#                 'date_present': False,
#                 'error': f'Date {date_to_check} not found in file',
#                 'dates_in_file': date_strings,
#                 'file_path': merged_file_path
#             }
#
#         # Get all IDs for this region from the vector file
#         from utils.region_boundaries import get_region_boundaries
#         region_boundaries = get_region_boundaries()
#
#         if region not in region_boundaries:
#             logger.warning(f"Region {region} not found in boundaries")
#             ds.close()
#             return {
#                 'success': False,
#                 'error': f'Region {region} not found in boundaries',
#                 'date_present': True
#             }
#
#         # Load vector file to get region IDs
#         vector_lake_file = os.environ.get('vector_lake_file')
#         if not vector_lake_file or not Path(vector_lake_file).exists():
#             logger.warning("Vector lake file not found, using IDs from merged file")
#             # Use IDs from the merged file
#             all_ids_in_file = set(ds['id_geohash'].values)
#             region_ids = list(all_ids_in_file)
#         else:
#             import geopandas as gpd
#             gdf = gpd.read_parquet(vector_lake_file)
#
#             # Get region bounds
#             bounds = region_boundaries[region]
#             x_min_start = bounds['X_MIN_START']
#             x_min_end = bounds['X_MIN_END']
#             y_min_start = bounds['Y_MIN_START']
#             y_min_end = bounds['Y_MIN_END']
#
#             # Filter by bounding box
#             gdf_subset = gdf[
#                 (gdf.geometry.x >= x_min_start) &
#                 (gdf.geometry.x <= x_min_end) &
#                 (gdf.geometry.y >= y_min_start) &
#                 (gdf.geometry.y <= y_min_end)
#                 ]
#
#             region_ids = gdf_subset['id_geohash'].values.tolist()
#
#         if not region_ids:
#             logger.warning(f"No IDs found for region {region}")
#             ds.close()
#             return {
#                 'success': False,
#                 'error': f'No IDs found for region {region}',
#                 'date_present': True,
#                 'region_id_count': 0
#             }
#
#         logger.info(f"Found {len(region_ids)} IDs for region {region}")
#
#         # Get all IDs in the merged file
#         file_ids = set(ds['id_geohash'].values)
#
#         # Check which region IDs are in the file
#         region_ids_in_file = [id_val for id_val in region_ids if id_val in file_ids]
#         region_ids_missing = [id_val for id_val in region_ids if id_val not in file_ids]
#
#         logger.info(f"IDs in file: {len(region_ids_in_file)}, IDs missing: {len(region_ids_missing)}")
#
#         # If no IDs from the region are in the file, fail
#         if not region_ids_in_file:
#             ds.close()
#             return {
#                 'success': False,
#                 'date_present': True,
#                 'region_ids_in_file': 0,
#                 'region_ids_missing': len(region_ids),
#                 'error': 'No region IDs found in merged file'
#             }
#
#         # Check if data exists for the date for these IDs
#         date_ts = pd.Timestamp(f"{date_to_check}-01")
#
#         # Select data for the date
#         date_data = ds.sel(date=date_ts)
#
#         # Check if we have data for all IDs
#         # Use a variable that should have data (e.g., 'water')
#         data_var = None
#         for var_candidate in ['water', 'water_observed', 'water_predicted']:
#             if var_candidate in date_data.data_vars:
#                 data_var = var_candidate
#                 break
#
#         if data_var is None:
#             # No data variable found, check date presence only
#             logger.warning("No data variable found, checking date presence only")
#             ds.close()
#             return {
#                 'success': True,
#                 'date_present': True,
#                 'region_ids_in_file': len(region_ids_in_file),
#                 'region_ids_missing': len(region_ids_missing),
#                 'all_ids_have_data': False,
#                 'warning': 'No data variable found to verify values',
#                 'file_path': merged_file_path
#             }
#
#         # Check data presence for each ID
#         ids_with_data = []
#         ids_without_data = []
#
#         for id_val in region_ids_in_file:
#             try:
#                 id_data = date_data.sel(id_geohash=id_val)
#                 if data_var in id_data:
#                     data_values = id_data[data_var].values
#                     # Check if there's at least one non-NaN value
#                     if np.any(~np.isnan(data_values)):
#                         ids_with_data.append(id_val)
#                     else:
#                         ids_without_data.append(id_val)
#                 else:
#                     ids_without_data.append(id_val)
#             except Exception as e:
#                 logger.debug(f"Error checking ID {id_val}: {e}")
#                 ids_without_data.append(id_val)
#
#         logger.info(f"IDs with data: {len(ids_with_data)}, IDs without data: {len(ids_without_data)}")
#
#         # Close the dataset
#         ds.close()
#
#         # Determine if all IDs have data
#         all_ids_have_data = len(ids_without_data) == 0 and len(ids_with_data) > 0
#
#         # Determine overall success
#         success = date_present and all_ids_have_data
#
#         result = {
#             'success': success,
#             'date_present': date_present,
#             'all_ids_have_data': all_ids_have_data,
#             'region_id_count': len(region_ids),
#             'region_ids_in_file': len(region_ids_in_file),
#             'region_ids_missing': len(region_ids_missing),
#             'ids_with_data': len(ids_with_data),
#             'ids_without_data': len(ids_without_data),
#             'date': date_to_check,
#             'region': region,
#             'file_path': merged_file_path,
#             'data_var_used': data_var
#         }
#
#         if success:
#             logger.info(f"✅ All data for {date_to_check} is present in {merged_file_path}")
#         else:
#             logger.warning(f"⚠️ Data for {date_to_check} is incomplete")
#             if not date_present:
#                 logger.warning(f"  - Date {date_to_check} not present in file")
#             if not all_ids_have_data:
#                 logger.warning(f"  - {len(ids_without_data)} IDs missing data for {date_to_check}")
#
#         return result
#
#     except Exception as e:
#         logger.error(f"Error checking data: {e}")
#         import traceback
#         traceback.print_exc()
#         return {
#             'success': False,
#             'error': str(e),
#             'file_path': merged_file_path
#         }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 ** 3)
    return 0


# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():
    logger.debug(f"Beginning historical run")
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    # ========== DEBUGGING: Check ALL environment variables ==========
    logger.info("=" * 80)
    logger.info("ENVIRONMENT VARIABLES (ALL)")
    logger.info("=" * 80)
    for key, value in sorted(os.environ.items()):
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    REGION = os.environ.get("region_name", "TEST")
    logger.info(f"=== REGION FROM ENV: '{REGION}' ===")

    # ========== DEBUGGING: Check region boundaries ==========
    import utils.region_boundaries
    boundaries = utils.region_boundaries.get_region_boundaries()
    logger.info(f"Available regions: {list(boundaries.keys())}")
    if REGION in boundaries:
        logger.info(f"Region '{REGION}' boundaries: {boundaries[REGION]}")
    else:
        logger.error(f"Region '{REGION}' NOT FOUND in boundaries!")
        logger.error(f"Available: {list(boundaries.keys())}")

    # ========== DEBUGGING: Check what's being passed to functions ==========
    logger.info("=" * 80)
    logger.info("FUNCTION CALL TRACING")
    logger.info("=" * 80)

    SHOULD_RUN = False

    summer_months = [6, 7, 8, 9]
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    if not all_dynamic_world_files:
        logger.error(f"No .nc files found in {dynamic_world_data_dir}")
        return

    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
    logger.debug(f"This is the most recent dynamic world file {original_most_recent_dynamic_world_file}")
    missing_dates_from_netcdf = download_new_dynamic_world_data.check_missing_data_in_netcdf(
        original_most_recent_dynamic_world_file
    )

    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    if TODAY_MONTH - 1 in summer_months:
        logger.debug(f"TODAY MONTH: {TODAY_MONTH}")
        logger.debug(f"Last month: {TODAY.month - 1} checking to see if we should run")
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    if SHOULD_RUN:
        timestamp_to_run = [pd.Timestamp(date(datetime.now().year, TODAY_MONTH - 1, 1))]
        date_to_run = [datetime(TODAY.year, TODAY_MONTH - 1, 1).strftime("%Y-%m")]
        logger.debug(f"timestamp_to_run: {timestamp_to_run}")

        # Verify downloads are complete
        downloads_complete = verify_downloads_complete(region=REGION, analysis_dates=date_to_run)
        logger.debug(downloads_complete)

        complete = downloads_complete.get('complete', False)
        complete_dates = downloads_complete.get('complete_dates', [])
        incomplete_dates = downloads_complete.get('incomplete_dates', [])
        summary = downloads_complete.get('summary', {})

        logger.debug(f"Total expected downloads {summary.get('total_expected_downloads', 0)}")
        logger.debug(f"Total successful downloads {summary.get('total_successful_downloads', 0)}")

        total_skipped = summary.get('total_skipped_downloads', 0)
        total_successful = summary.get('total_successful_downloads', 0)
        total_expected = summary.get('total_expected_downloads', 0)

        total_skipped_and_successful = total_skipped + total_successful

        if total_expected > 0:
            percent_downloaded = float(total_skipped_and_successful) / float(total_expected)
            logger.debug(f"Percent downloaded: {percent_downloaded:.4f}")

            if percent_downloaded > 0.99:
                complete = True

        if complete:
            logger.debug(f"Merge all the results for {REGION} and {date_to_run[0]}")

            # Create the merged file path
            merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{REGION}_{date_to_run[0]}.nc")

            # Merge the new results
            merge_result = merge_new_results(
                region=REGION,
                date_to_merge=date_to_run[0],
                merged_file_path=merged_file_path,
                env_path=env_path
            )

            logger.debug(f"Merge result: {merge_result}")

            if merge_result.get('success', False):
                logger.debug(f"Verifying all data is in the file")

                # Check if all data is in the file
                check_result = is_all_new_data_in_file(
                    region=REGION,
                    date_to_check=date_to_run[0],
                    merged_file_path=merged_file_path,
                    env_path=env_path
                )

                logger.debug(f"Verification result: {check_result}")

                if check_result.get('success', False):
                    logger.info(f"✅ SUCCESS: All data for {date_to_run[0]} merged and verified!")
                else:
                    logger.warning(f"⚠️ Verification failed for {date_to_run[0]}")
                    logger.warning(f"  Reason: {check_result.get('error', 'Unknown error')}")
            else:
                logger.error(f"❌ Merge failed for {date_to_run[0]}")
                logger.error(f"  Reason: {merge_result.get('error', 'Unknown error')}")
        else:
            logger.debug(f"Downloads not complete for {REGION} - incomplete dates: {incomplete_dates}")
    else:
        logger.debug(f"Too early in the month to run downloads for {REGION}")

    logger.info("=" * 80)
    logger.info("SCRIPT COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()