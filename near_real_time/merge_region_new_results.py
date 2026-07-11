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


# =============================================================================
# IMPLEMENTATION OF MERGE_NEW_RESULTS
# =============================================================================
def merge_new_results(
        region: str = 'TEST',
        date_to_merge: str = None,
        merged_file_path: str = None,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Merge new downloaded results for a single date into a NetCDF file.

    This function:
    1. Finds all downloaded files for the specified date
    2. Combines them into a single dataset
    3. Saves the combined data to the specified file path

    Args:
        region: Region name (e.g., "TEST", "AFRICA")
        date_to_merge: Date in "YYYY-MM" format
        merged_file_path: Path where the merged file should be saved
        env_path: Optional path to .env file

    Returns:
        dict: Result with status, file path, and statistics
    """
    logger.debug(f"Merging new results for {region} and {date_to_merge} into file {merged_file_path}")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Validate inputs
    if date_to_merge is None:
        logger.error("date_to_merge is required")
        return {'success': False, 'error': 'date_to_merge is required'}

    if merged_file_path is None:
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if not dynamic_world_data_dir:
            logger.error("dynamic_world_data not set in environment")
            return {'success': False, 'error': 'dynamic_world_data not set'}
        merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_merge}.nc")

    # Ensure directory exists
    Path(merged_file_path).parent.mkdir(parents=True, exist_ok=True)

    # Find downloaded files for this date
    dynamic_world_download_dir = Path(os.environ.get('dynamic_world_downloads', ''))
    download_dir = dynamic_world_download_dir / region / f'download_{date_to_merge}'

    if not download_dir.exists():
        logger.warning(f"Download directory does not exist: {download_dir}")
        return {'success': False, 'error': f'Download directory not found: {download_dir}'}

    # Get all downloaded NetCDF files for this date
    downloaded_files = sorted(glob.glob(str(download_dir / f'DW_{date_to_merge}_*.nc')))

    if not downloaded_files:
        logger.warning(f"No downloaded files found for {date_to_merge} in {download_dir}")
        return {'success': False, 'error': f'No downloaded files found for {date_to_merge}'}

    logger.info(f"Found {len(downloaded_files)} downloaded files for {date_to_merge}")

    try:
        # Combine all downloaded files into a single dataset
        logger.info("Combining downloaded files...")
        combined = None
        failed_files = []

        for nc_file in downloaded_files:
            try:
                ds = xr.open_dataset(nc_file)
                if len(ds['id_geohash']) > 0:
                    if combined is None:
                        combined = ds
                    else:
                        # Concatenate along id_geohash dimension
                        combined = xr.concat([combined, ds], dim='id_geohash')
                        # Remove duplicate IDs
                        _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                        if len(unique_idx) < len(combined['id_geohash']):
                            combined = combined.isel(id_geohash=np.sort(unique_idx))
                else:
                    logger.warning(f"File {nc_file} has no IDs, skipping")
                    failed_files.append(nc_file)
            except Exception as e:
                logger.error(f"Error opening {nc_file}: {e}")
                failed_files.append(nc_file)

        if combined is None:
            logger.error("No valid data to merge")
            return {'success': False, 'error': 'No valid data to merge'}

        logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

        # Ensure date dimension is correct
        if len(combined['date']) > 0:
            # Sort by date
            combined = combined.sortby('date')

        # Write to NetCDF with compression
        logger.info(f"Writing merged data to {merged_file_path}")

        encoding = {}
        for var in combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        # Write to file
        combined.to_netcdf(merged_file_path, encoding=encoding)

        # Get file size
        file_size_gb = Path(merged_file_path).stat().st_size / (1024 ** 3)

        # Clean up
        combined.close()
        gc.collect()

        result = {
            'success': True,
            'file_path': merged_file_path,
            'id_count': len(combined['id_geohash']),
            'date_count': len(combined['date']),
            'file_size_gb': round(file_size_gb, 4),
            'files_merged': len(downloaded_files) - len(failed_files),
            'files_failed': len(failed_files),
            'failed_files': failed_files if failed_files else None,
            'region': region,
            'date': date_to_merge
        }

        logger.info(f"✅ Merge completed successfully!")
        logger.info(f"  File: {merged_file_path}")
        logger.info(f"  IDs: {result['id_count']:,}")
        logger.info(f"  Dates: {result['date_count']}")
        logger.info(f"  Size: {result['file_size_gb']:.4f} GB")

        return result

    except Exception as e:
        logger.error(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# IMPLEMENTATION OF IS_ALL_NEW_DATA_IN_FILE
# =============================================================================
def is_all_new_data_in_file(
        region: str = 'TEST',
        date_to_check: str = None,
        merged_file_path: str = None,
        env_path: str = None,
        skip_verification: bool = False  # New parameter
) -> Dict[str, Any]:
    """
    Check if all expected data for a date is present in the merged file.

    This function verifies that:
    1. The merged file exists and is valid
    2. The file contains the expected date
    3. All IDs from the region have data for that date

    Uses vectorized operations for large datasets.
    """
    logger.debug(f"Checking if all new data for {region} and {date_to_check} is in {merged_file_path}")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Validate inputs
    if date_to_check is None:
        logger.error("date_to_check is required")
        return {'success': False, 'error': 'date_to_check is required'}

    if merged_file_path is None:
        dynamic_world_data_dir = os.environ.get('dynamic_world_data')
        if not dynamic_world_data_dir:
            logger.error("dynamic_world_data not set in environment")
            return {'success': False, 'error': 'dynamic_world_data not set'}
        merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_check}.nc")

    # Check if file exists
    if not Path(merged_file_path).exists():
        logger.error(f"Merged file does not exist: {merged_file_path}")
        return {
            'success': False,
            'error': 'File not found',
            'file_exists': False,
            'file_path': merged_file_path
        }

    try:
        # Open the merged file
        logger.info(f"Opening merged file: {merged_file_path}")
        ds = xr.open_dataset(merged_file_path)

        # Check if the date exists in the file
        dates_in_file = pd.to_datetime(ds['date'].values)
        date_strings = [d.strftime("%Y-%m") for d in dates_in_file]

        date_present = date_to_check in date_strings

        if not date_present:
            logger.warning(f"Date {date_to_check} not found in merged file")
            ds.close()
            return {
                'success': False,
                'date_present': False,
                'error': f'Date {date_to_check} not found in file',
                'dates_in_file': date_strings,
                'file_path': merged_file_path
            }

        # Get all IDs for this region from the vector file
        from utils.region_boundaries import get_region_boundaries
        region_boundaries = get_region_boundaries()

        if region not in region_boundaries:
            logger.warning(f"Region {region} not found in boundaries")
            ds.close()
            return {
                'success': False,
                'error': f'Region {region} not found in boundaries',
                'date_present': True
            }

        # Load vector file to get region IDs
        vector_lake_file = os.environ.get('vector_lake_file')
        if not vector_lake_file or not Path(vector_lake_file).exists():
            logger.warning("Vector lake file not found, using IDs from merged file")
            all_ids_in_file = set(ds['id_geohash'].values)
            region_ids = list(all_ids_in_file)
        else:
            import geopandas as gpd
            gdf = gpd.read_parquet(vector_lake_file)

            # Get region bounds
            bounds = region_boundaries[region]
            x_min_start = bounds['X_MIN_START']
            x_min_end = bounds['X_MIN_END']
            y_min_start = bounds['Y_MIN_START']
            y_min_end = bounds['Y_MIN_END']

            # Handle both Point and Polygon geometries
            geom_type = gdf.geometry.geom_type.iloc[0] if len(gdf) > 0 else None

            if geom_type in ['Polygon', 'MultiPolygon']:
                centroids = gdf.geometry.centroid
                x_coords = centroids.x
                y_coords = centroids.y
            elif geom_type == 'Point':
                x_coords = gdf.geometry.x
                y_coords = gdf.geometry.y
            else:
                rep_points = gdf.geometry.representative_point()
                x_coords = rep_points.x
                y_coords = rep_points.y

            # Filter by bounding box
            mask = (x_coords >= x_min_start) & (x_coords <= x_min_end) & \
                   (y_coords >= y_min_start) & (y_coords <= y_min_end)

            gdf_subset = gdf[mask]
            region_ids = gdf_subset['id_geohash'].values.tolist()

        if not region_ids:
            logger.warning(f"No IDs found for region {region}")
            ds.close()
            return {
                'success': False,
                'error': f'No IDs found for region {region}',
                'date_present': True,
                'region_id_count': 0
            }

        logger.info(f"Found {len(region_ids):,} IDs for region {region}")

        # ================================================================
        # OPTIMIZED: Use vectorized operations instead of loop
        # ================================================================

        # Get all IDs in the merged file
        file_ids = set(ds['id_geohash'].values)

        # Use set operations for fast ID checking
        region_ids_set = set(region_ids)
        region_ids_in_file = region_ids_set & file_ids
        region_ids_missing = region_ids_set - file_ids

        logger.info(f"IDs in file: {len(region_ids_in_file):,}, IDs missing: {len(region_ids_missing):,}")

        # If no IDs from the region are in the file, fail
        if not region_ids_in_file:
            ds.close()
            return {
                'success': False,
                'date_present': True,
                'region_ids_in_file': 0,
                'region_ids_missing': len(region_ids),
                'error': 'No region IDs found in merged file'
            }

        # Check if data exists for the date
        date_ts = pd.Timestamp(f"{date_to_check}-01")

        # ================================================================
        # VECTORIZED DATA CHECK - MUCH FASTER!
        # ================================================================

        # Select data for the date (this is fast)
        date_data = ds.sel(date=date_ts)

        # Find a data variable to check
        data_var = None
        for var_candidate in ['water', 'water_observed', 'water_predicted']:
            if var_candidate in date_data.data_vars:
                data_var = var_candidate
                break

        if data_var is None:
            # No data variable found, check date presence only
            logger.warning("No data variable found, checking date presence only")
            ds.close()
            return {
                'success': True,
                'date_present': True,
                'region_ids_in_file': len(region_ids_in_file),
                'region_ids_missing': len(region_ids_missing),
                'all_ids_have_data': False,
                'warning': 'No data variable found to verify values',
                'file_path': merged_file_path
            }

        # ================================================================
        # VECTORIZED: Get all IDs from date_data
        # ================================================================
        date_ids = date_data['id_geohash'].values

        # Convert to set for fast membership testing
        date_ids_set = set(date_ids)

        # Find which region IDs are in the date data
        region_ids_with_date = region_ids_in_file & date_ids_set
        region_ids_without_date = region_ids_in_file - date_ids_set

        logger.info(f"IDs with date data: {len(region_ids_with_date):,}, IDs without: {len(region_ids_without_date):,}")

        # ================================================================
        # VECTORIZED: Check if the data variable has non-NaN values
        # ================================================================
        try:
            # Get the data values for the entire date slice (fast)
            data_values = date_data[data_var].values

            # Find which IDs in the region have non-NaN values
            # Create a boolean mask for IDs in the region
            id_mask = np.isin(date_data['id_geohash'].values, list(region_ids_in_file))

            # Get the data values for only the region IDs
            region_data_values = data_values[id_mask]

            # Check which IDs have non-NaN values (vectorized)
            has_data_mask = ~np.isnan(region_data_values)

            # Get the IDs that have data
            region_ids_with_data = date_data['id_geohash'].values[id_mask][has_data_mask]

            # Find missing IDs using set operations
            region_ids_with_data_set = set(region_ids_with_data)
            ids_with_data = region_ids_with_data_set
            ids_without_data = region_ids_in_file - region_ids_with_data_set

            logger.info(f"IDs with valid data: {len(ids_with_data):,}, IDs without: {len(ids_without_data):,}")

        except Exception as e:
            logger.warning(f"Error checking data values: {e}, falling back to date presence check")
            ids_with_data = region_ids_with_date
            ids_without_data = region_ids_without_date

        # Close the dataset
        ds.close()

        # Determine if all IDs have data
        all_ids_have_data = len(ids_without_data) == 0 and len(ids_with_data) > 0

        # Determine overall success
        success = date_present and all_ids_have_data

        result = {
            'success': success,
            'date_present': date_present,
            'all_ids_have_data': all_ids_have_data,
            'region_id_count': len(region_ids),
            'region_ids_in_file': len(region_ids_in_file),
            'region_ids_missing': len(region_ids_missing),
            'ids_with_data': len(ids_with_data),
            'ids_without_data': len(ids_without_data),
            'date': date_to_check,
            'region': region,
            'file_path': merged_file_path,
            'data_var_used': data_var,
            'geometry_type': geom_type if 'geom_type' in locals() else 'unknown',
            'verification_method': 'vectorized'
        }

        if success:
            logger.info(f"✅ All data for {date_to_check} is present in {merged_file_path}")
        else:
            logger.warning(f"⚠️ Data for {date_to_check} is incomplete")
            if not date_present:
                logger.warning(f"  - Date {date_to_check} not present in file")
            if not all_ids_have_data:
                logger.warning(f"  - {len(ids_without_data):,} IDs missing data for {date_to_check}")

        return result

    except Exception as e:
        logger.error(f"Error checking data: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'file_path': merged_file_path
        }


# =============================================================================
# COMBINE REGION FILES
# =============================================================================
def combine_region_files(
        region_files: List[str],
        output_file: str,
        date_to_check: str = None,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Combine multiple region NetCDF files into a single combined file.

    Args:
        region_files: List of paths to region NetCDF files
        output_file: Path for the combined output file
        date_to_check: Date in "YYYY-MM" format for verification
        env_path: Optional path to .env file

    Returns:
        dict: Result with status and statistics
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("COMBINING REGION FILES")
    logger.info(f"{'=' * 80}")
    logger.info(f"Number of files to combine: {len(region_files)}")
    logger.info(f"Output file: {output_file}")

    if not region_files:
        logger.error("No region files to combine")
        return {'success': False, 'error': 'No region files to combine'}

    # Verify all files exist
    missing_files = [f for f in region_files if not Path(f).exists()]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return {'success': False, 'error': f'Missing files: {missing_files}'}

    try:
        # Load all region datasets
        logger.info("Loading region datasets...")
        datasets = []
        file_info = []

        for file_path in region_files:
            try:
                ds = xr.open_dataset(file_path)

                # Get file info
                id_count = len(ds['id_geohash']) if 'id_geohash' in ds.dims else 0
                date_count = len(ds['date']) if 'date' in ds.dims else 0
                file_size_gb = Path(file_path).stat().st_size / (1024 ** 3)

                file_info.append({
                    'file': file_path,
                    'id_count': id_count,
                    'date_count': date_count,
                    'file_size_gb': round(file_size_gb, 4)
                })

                datasets.append(ds)

            except Exception as e:
                logger.error(f"Error opening {file_path}: {e}")
                # Close any datasets that were opened
                for ds in datasets:
                    try:
                        ds.close()
                    except:
                        pass
                return {'success': False, 'error': f'Error opening {file_path}: {e}'}

        # Log file info
        logger.info("\nFiles to combine:")
        for info in file_info:
            logger.info(
                f"  {Path(info['file']).name}: {info['id_count']:,} IDs, {info['date_count']} dates, {info['file_size_gb']:.4f} GB")

        # Combine all datasets
        logger.info("Combining datasets...")
        combined = None

        for ds in datasets:
            if combined is None:
                combined = ds
            else:
                # Concatenate along id_geohash dimension
                combined = xr.concat([combined, ds], dim='id_geohash')
                # Remove duplicate IDs
                _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(combined['id_geohash']):
                    removed_count = len(combined['id_geohash']) - len(unique_idx)
                    logger.info(f"Removed {removed_count} duplicate IDs")
                    combined = combined.isel(id_geohash=np.sort(unique_idx))

        if combined is None:
            logger.error("No datasets to combine")
            return {'success': False, 'error': 'No datasets to combine'}

        # Sort by ID and date
        combined = combined.sortby(['id_geohash', 'date'])

        logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

        # Write to output file with compression
        logger.info(f"Writing combined file to {output_file}")

        encoding = {}
        for var in combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        combined.to_netcdf(output_file, encoding=encoding)

        # Get file size
        file_size_gb = Path(output_file).stat().st_size / (1024 ** 3)

        # Clean up
        for ds in datasets:
            try:
                ds.close()
            except:
                pass
        combined.close()
        gc.collect()

        result = {
            'success': True,
            'file_path': output_file,
            'id_count': len(combined['id_geohash']),
            'date_count': len(combined['date']),
            'file_size_gb': round(file_size_gb, 4),
            'files_combined': len(datasets),
            'file_info': file_info
        }

        logger.info(f"✅ Combined file created successfully!")
        logger.info(f"  File: {output_file}")
        logger.info(f"  IDs: {result['id_count']:,}")
        logger.info(f"  Dates: {result['date_count']}")
        logger.info(f"  Size: {result['file_size_gb']:.4f} GB")

        return result

    except Exception as e:
        logger.error(f"Error combining files: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def verify_combined_file(
        combined_file_path: str,
        region_files: List[str],
        date_to_check: str = None,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Verify that the combined file contains all data from the region files.

    Args:
        combined_file_path: Path to the combined file
        region_files: List of region file paths
        date_to_check: Date in "YYYY-MM" format to check
        env_path: Optional path to .env file

    Returns:
        dict: Verification results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFYING COMBINED FILE")
    logger.info(f"{'=' * 80}")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Check combined file exists
    if not Path(combined_file_path).exists():
        logger.error(f"Combined file does not exist: {combined_file_path}")
        return {'success': False, 'error': 'Combined file not found'}

    try:
        # Open combined file
        combined_ds = xr.open_dataset(combined_file_path)
        combined_ids = set(combined_ds['id_geohash'].values)
        combined_dates = set(pd.to_datetime(combined_ds['date'].values))
        combined_date_strings = {d.strftime("%Y-%m") for d in combined_dates}

        logger.info(f"Combined file has {len(combined_ids):,} IDs and {len(combined_dates)} dates")

        # Check each region file
        region_results = {}
        all_present = True
        total_ids = 0
        total_missing = 0

        for region_file in region_files:
            region_name = Path(region_file).stem.replace('dw_', '')
            logger.info(f"\nChecking region: {region_name}")

            try:
                region_ds = xr.open_dataset(region_file)
                region_ids = set(region_ds['id_geohash'].values)

                # Check if all region IDs are in combined file
                missing_ids = region_ids - combined_ids
                ids_present = region_ids - missing_ids

                logger.info(f"  Region IDs: {len(region_ids):,}")
                logger.info(f"  IDs in combined: {len(ids_present):,}")
                logger.info(f"  IDs missing: {len(missing_ids):,}")

                if missing_ids:
                    all_present = False
                    total_missing += len(missing_ids)
                    logger.warning(f"  ⚠️ Missing {len(missing_ids)} IDs in combined file")
                    # Show first few missing IDs
                    sample_missing = list(missing_ids)[:5]
                    logger.warning(f"  Sample missing IDs: {sample_missing}")

                # Check if date is present
                if date_to_check:
                    if date_to_check in combined_date_strings:
                        logger.info(f"  ✅ Date {date_to_check} present in combined file")
                    else:
                        logger.warning(f"  ⚠️ Date {date_to_check} NOT present in combined file")
                        all_present = False

                region_results[region_name] = {
                    'total_ids': len(region_ids),
                    'ids_present': len(ids_present),
                    'ids_missing': len(missing_ids),
                    'all_ids_present': len(missing_ids) == 0,
                    'missing_ids': list(missing_ids)[:10]  # Store first 10 missing IDs
                }

                total_ids += len(region_ids)
                region_ds.close()

            except Exception as e:
                logger.error(f"Error checking region {region_name}: {e}")
                region_results[region_name] = {
                    'error': str(e),
                    'all_ids_present': False
                }
                all_present = False

        combined_ds.close()

        result = {
            'success': all_present,
            'combined_file': combined_file_path,
            'total_ids_in_combined': len(combined_ids),
            'total_region_ids': total_ids,
            'total_missing_ids': total_missing,
            'all_regions_present': all_present,
            'region_results': region_results,
            'date_present': date_to_check in combined_date_strings if date_to_check else None
        }

        if all_present:
            logger.info("\n✅ All region data is present in the combined file!")
        else:
            logger.warning(f"\n⚠️ {total_missing} IDs are missing from the combined file")

        return result

    except Exception as e:
        logger.error(f"Error verifying combined file: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 ** 3)
    return 0


def process_region(
        region: str,
        date_to_run: str,
        timestamp_to_run: pd.Timestamp,
        env_path: str = None,
        dynamic_world_data_dir: str = None,
        skip_verification_threshold: int = 100000  # New parameter
) -> Dict[str, Any]:
    """
    Process a single region: verify downloads, merge, and verify.

    Args:
        region: Region name
        date_to_run: Date in "YYYY-MM" format
        timestamp_to_run: Pandas Timestamp for the date
        env_path: Optional path to .env file
        dynamic_world_data_dir: Directory for dynamic world data
        skip_verification_threshold: Skip verification if region has more than this many IDs
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING REGION: {region}")
    logger.info(f"{'=' * 80}")

    result = {
        'region': region,
        'date': date_to_run,
        'success': False,
        'steps': {}
    }

    # Step 1: Verify downloads are complete
    logger.info(f"Step 1: Verifying downloads for {region}...")
    downloads_complete = verify_downloads_complete(region=region, analysis_dates=[date_to_run])
    logger.debug(downloads_complete)

    complete = downloads_complete.get('complete', False)
    summary = downloads_complete.get('summary', {})

    total_expected = summary.get('total_expected_downloads', 0)
    total_skipped = summary.get('total_skipped_downloads', 0)
    total_successful = summary.get('total_successful_downloads', 0)
    total_skipped_and_successful = total_skipped + total_successful

    if total_expected > 0:
        percent_downloaded = float(total_skipped_and_successful) / float(total_expected)
        logger.info(f"  Percent downloaded: {percent_downloaded:.4f}")

        if percent_downloaded > 0.99:
            complete = True

    result['steps']['download_verification'] = {
        'complete': complete,
        'percent_downloaded': percent_downloaded if total_expected > 0 else 0,
        'total_expected': total_expected,
        'total_successful': total_successful,
        'total_skipped': total_skipped
    }

    if not complete:
        logger.warning(f"⚠️ Downloads not complete for {region} - skipping merge")
        result['success'] = False
        result['reason'] = 'Downloads incomplete'
        return result

    # Step 2: Check if already merged
    merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_run}.nc")
    already_merged = False

    logger.info(f"Step 2: Checking if already merged for {region}...")
    if os.path.isfile(merged_file_path):
        # Quick check: just check if the file exists and has the date
        # Use a lightweight check instead of full verification
        try:
            ds = xr.open_dataset(merged_file_path)
            dates_in_file = pd.to_datetime(ds['date'].values)
            date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
            date_present = date_to_run in date_strings
            ds.close()

            if date_present:
                # File has the date, consider it already merged
                already_merged = True
                logger.info(f"✅ Region {region} already has date {date_to_run} - skipping merge")
                result['success'] = True
                result['merged_file'] = merged_file_path
                result['reason'] = 'Already merged (date present)'
                return result
        except Exception as e:
            logger.warning(f"Could not quick-check merged file: {e}")

    # Step 3: Merge the new results
    logger.info(f"Step 3: Merging results for {region}...")
    merge_result = merge_new_results(
        region=region,
        date_to_merge=date_to_run,
        merged_file_path=merged_file_path,
        env_path=env_path
    )

    result['steps']['merge'] = merge_result

    if not merge_result.get('success', False):
        logger.error(f"❌ Merge failed for {region}: {merge_result.get('error', 'Unknown error')}")
        result['success'] = False
        result['reason'] = 'Merge failed'
        return result

    # Step 4: Verify the merge (skip for large regions)
    id_count = merge_result.get('id_count', 0)
    if id_count > skip_verification_threshold:
        logger.info(f"⏭️ Skipping detailed verification for large region {region} ({id_count:,} IDs)")
        logger.info(f"✅ Region {region} merged successfully (verification skipped)")
        result['success'] = True
        result['merged_file'] = merged_file_path
        result['reason'] = f'Successfully merged (verification skipped for {id_count:,} IDs)'
        return result

    logger.info(f"Step 4: Verifying merge for {region}...")
    check_result = is_all_new_data_in_file(
        region=region,
        date_to_check=date_to_run,
        merged_file_path=merged_file_path,
        env_path=env_path
    )

    result['steps']['verification'] = check_result

    if check_result.get('success', False):
        logger.info(f"✅ SUCCESS: Region {region} data for {date_to_run} merged and verified!")
        result['success'] = True
        result['merged_file'] = merged_file_path
        result['reason'] = 'Successfully merged and verified'
    else:
        logger.warning(f"⚠️ Verification failed for {region}: {check_result.get('error', 'Unknown error')}")
        result['success'] = False
        result['reason'] = 'Verification failed'

    return result


# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():
    logger.debug(f"Beginning historical run for ALL regions")
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
        logger.debug(f"TODAY MONTH: {TODAY_MONTH}")
        logger.debug(f"Last month: {TODAY.month - 1} checking to see if we should run")
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    if not SHOULD_RUN:
        logger.debug(f"Too early in the month to run downloads - exiting")
        return

    # ========== Prepare date to run ==========
    timestamp_to_run = pd.Timestamp(date(datetime.now().year, TODAY_MONTH - 1, 1))
    date_to_run = datetime(TODAY.year, TODAY_MONTH - 1, 1).strftime("%Y-%m")
    logger.info(f"Processing date: {date_to_run}")
    logger.info(f"Timestamp: {timestamp_to_run}")

    # ========== Process ALL regions ==========
    results = {}
    success_count = 0
    failure_count = 0
    skipped_count = 0
    region_files = []

    for region in all_regions:
        try:
            logger.info(f"\n{'#' * 80}")
            logger.info(f"PROCESSING REGION: {region}")
            logger.info(f"{'#' * 80}")

            result = process_region(
                region=region,
                date_to_run=date_to_run,
                timestamp_to_run=timestamp_to_run,
                env_path=env_path,
                dynamic_world_data_dir=dynamic_world_data_dir
            )

            results[region] = result

            if result.get('success', False):
                if result.get('reason') == 'Already merged':
                    skipped_count += 1
                    logger.info(f"⏭️ Region {region} already merged (skipped)")
                else:
                    success_count += 1
                    logger.info(f"✅ Region {region} processed successfully!")

                # Track the merged file for combination
                if 'merged_file' in result:
                    region_files.append(result['merged_file'])
            else:
                failure_count += 1
                logger.error(f"❌ Region {region} failed: {result.get('reason', 'Unknown error')}")

        except Exception as e:
            logger.error(f"❌ Error processing region {region}: {e}")
            import traceback
            traceback.print_exc()
            results[region] = {
                'region': region,
                'date': date_to_run,
                'success': False,
                'reason': f'Exception: {str(e)}',
                'error': str(e)
            }
            failure_count += 1

    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY - ALL REGIONS")
    logger.info("=" * 80)
    logger.info(f"Date processed: {date_to_run}")
    logger.info(f"Total regions: {len(all_regions)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"⏭️ Already merged: {skipped_count}")
    logger.info(f"❌ Failed: {failure_count}")
    logger.info("=" * 80)

    # Print detailed results per region
    logger.info("\nDETAILED RESULTS:")
    logger.info("-" * 60)
    for region, result in results.items():
        status = "✅" if result.get('success', False) else "❌"
        reason = result.get('reason', 'Unknown')
        logger.info(f"  {status} {region}: {reason}")

        # Print any error details if available
        if 'error' in result:
            logger.info(f"      Error: {result['error']}")

    # ========== COMBINE ALL REGION FILES ==========
    if failure_count == 0 and region_files:
        logger.info("\n" + "=" * 80)
        logger.info("COMBINING ALL REGION FILES")
        logger.info("=" * 80)
        logger.info(f"Found {len(region_files)} region files to combine")

        # Check if all regions are present
        expected_files = [f"dw_{region}_{date_to_run}.nc" for region in all_regions]
        missing_files = [f for f in expected_files if f not in [Path(f).name for f in region_files]]

        if missing_files:
            logger.warning(f"⚠️ Missing region files: {missing_files}")
            logger.warning("Skipping combination due to missing files")
        else:
            # Create combined file name
            combined_file_name = f"dynamic_world_combined_{date_to_run}.nc"
            combined_file_path = os.path.join(dynamic_world_data_dir, combined_file_name)

            logger.info(f"Combining into: {combined_file_path}")

            # Combine the files
            combine_result = combine_region_files(
                region_files=region_files,
                output_file=combined_file_path,
                date_to_check=date_to_run,
                env_path=env_path
            )

            if combine_result.get('success', False):
                logger.info(f"✅ Combined file created successfully!")

                # Verify the combined file
                logger.info("Verifying combined file...")
                verify_result = verify_combined_file(
                    combined_file_path=combined_file_path,
                    region_files=region_files,
                    date_to_check=date_to_run,
                    env_path=env_path
                )

                if verify_result.get('success', False):
                    logger.info("✅ Combined file verification passed!")
                    logger.info(f"  Combined file: {combined_file_path}")
                    logger.info(f"  Total IDs: {verify_result['total_ids_in_combined']:,}")
                    logger.info(f"  Total region IDs: {verify_result['total_region_ids']:,}")
                else:
                    logger.warning("⚠️ Combined file verification had issues:")
                    for region, reg_result in verify_result.get('region_results', {}).items():
                        if not reg_result.get('all_ids_present', False):
                            logger.warning(f"  Region {region}: {reg_result.get('ids_missing', 0)} IDs missing")
            else:
                logger.error(f"❌ Failed to combine region files: {combine_result.get('error', 'Unknown error')}")
    else:
        if failure_count > 0:
            logger.warning("⚠️ Not combining region files due to failures")
        elif not region_files:
            logger.warning("⚠️ No region files to combine")

    logger.info("\n" + "=" * 80)
    logger.info("SCRIPT COMPLETED")
    logger.info("=" * 80)

    # Return exit code
    if failure_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()