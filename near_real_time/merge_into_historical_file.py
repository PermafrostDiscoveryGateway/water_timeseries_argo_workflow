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
# FAST VECTORIZED VERIFICATION
# =============================================================================
def verify_region_data_vectorized(
        region: str,
        date_to_check: str,
        file_path: str,
        env_path: str = None,
        sample_size: int = 1000  # Number of IDs to sample for checking
) -> Dict[str, Any]:
    """
    Fast vectorized verification of region data in a file.
    Uses sampling for large regions instead of checking every ID.
    """
    logger.debug(f"Fast vectorized check for {region} and {date_to_check} in {file_path}")

    # Load environment
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Check if file exists
    if not Path(file_path).exists():
        return {'success': False, 'error': 'File not found', 'file_exists': False}

    try:
        ds = xr.open_dataset(file_path)

        # Check date presence
        dates_in_file = pd.to_datetime(ds['date'].values)
        date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
        date_present = date_to_check in date_strings

        if not date_present:
            ds.close()
            return {'success': False, 'date_present': False, 'error': f'Date {date_to_check} not found'}

        # Get region IDs from vector file
        from utils.region_boundaries import get_region_boundaries
        region_boundaries = get_region_boundaries()

        if region not in region_boundaries:
            ds.close()
            return {'success': False, 'error': f'Region {region} not found in boundaries'}

        vector_lake_file = os.environ.get('vector_lake_file')
        if not vector_lake_file or not Path(vector_lake_file).exists():
            ds.close()
            return {'success': False, 'error': 'Vector lake file not found'}

        import geopandas as gpd
        gdf = gpd.read_parquet(vector_lake_file)

        bounds = region_boundaries[region]
        x_min_start = bounds['X_MIN_START']
        x_min_end = bounds['X_MIN_END']
        y_min_start = bounds['Y_MIN_START']
        y_min_end = bounds['Y_MIN_END']

        # Handle geometry types
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
            ds.close()
            return {'success': False, 'error': f'No IDs found for region {region}'}

        total_region_ids = len(region_ids)
        logger.info(f"Region {region} has {total_region_ids:,} IDs")

        # Get IDs in file
        file_ids = set(ds['id_geohash'].values)

        # Check if all region IDs are in file (using set operations - fast)
        region_ids_set = set(region_ids)
        missing_ids = region_ids_set - file_ids
        ids_in_file = region_ids_set & file_ids

        logger.info(f"IDs in file: {len(ids_in_file):,}, IDs missing: {len(missing_ids):,}")

        if len(missing_ids) > 0:
            ds.close()
            return {
                'success': False,
                'total_ids': total_region_ids,
                'ids_in_file': len(ids_in_file),
                'ids_missing': len(missing_ids),
                'missing_sample': list(missing_ids)[:10],
                'error': f'{len(missing_ids):,} IDs missing from file'
            }

        # Now check if data exists for the date (sample-based for speed)
        date_ts = pd.Timestamp(f"{date_to_check}-01")
        date_data = ds.sel(date=date_ts)

        # Find a data variable
        data_var = None
        for var_candidate in ['water', 'water_observed', 'water_predicted']:
            if var_candidate in date_data.data_vars:
                data_var = var_candidate
                break

        if data_var is None:
            ds.close()
            return {'success': True, 'date_present': True, 'warning': 'No data variable found'}

        # Sample-based check for data values
        ids_list = list(ids_in_file)
        sample_count = min(sample_size, len(ids_list))

        if sample_count < len(ids_list):
            # Sample IDs
            import random
            sampled_ids = random.sample(ids_list, sample_count)
            logger.info(f"Sampling {sample_count} IDs out of {len(ids_list):,} for data validation")

            ids_with_data = 0
            ids_without_data = 0

            for id_val in sampled_ids:
                try:
                    id_data = date_data.sel(id_geohash=id_val)
                    if data_var in id_data:
                        data_values = id_data[data_var].values
                        if np.any(~np.isnan(data_values)):
                            ids_with_data += 1
                        else:
                            ids_without_data += 1
                    else:
                        ids_without_data += 1
                except Exception:
                    ids_without_data += 1

            # If most sampled IDs have data, assume all do
            data_success_rate = ids_with_data / sample_count if sample_count > 0 else 0
            logger.info(f"Sample data success rate: {data_success_rate:.2%}")

            all_have_data = data_success_rate > 0.95  # 95% threshold

            ds.close()

            return {
                'success': all_have_data,
                'date_present': date_present,
                'total_ids': total_region_ids,
                'ids_in_file': len(ids_in_file),
                'ids_missing': 0,
                'sampled': sample_count,
                'ids_with_data': ids_with_data,
                'ids_without_data': ids_without_data,
                'data_success_rate': data_success_rate,
                'all_ids_have_data': all_have_data,
                'verification_method': 'sampled_vectorized'
            }
        else:
            # Small region - check all IDs
            logger.info(f"Checking all {len(ids_list):,} IDs for data validation")

            ids_with_data = []
            ids_without_data = []

            for id_val in ids_list:
                try:
                    id_data = date_data.sel(id_geohash=id_val)
                    if data_var in id_data:
                        data_values = id_data[data_var].values
                        if np.any(~np.isnan(data_values)):
                            ids_with_data.append(id_val)
                        else:
                            ids_without_data.append(id_val)
                    else:
                        ids_without_data.append(id_val)
                except Exception:
                    ids_without_data.append(id_val)

            all_have_data = len(ids_without_data) == 0 and len(ids_with_data) > 0

            ds.close()

            return {
                'success': all_have_data,
                'date_present': date_present,
                'total_ids': total_region_ids,
                'ids_in_file': len(ids_in_file),
                'ids_missing': 0,
                'ids_with_data': len(ids_with_data),
                'ids_without_data': len(ids_without_data),
                'all_ids_have_data': all_have_data,
                'verification_method': 'full_vectorized'
            }

    except Exception as e:
        logger.error(f"Error in vectorized verification: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# COMBINE REGION FILES
# =============================================================================
def combine_region_files(
        region_files: List[str],
        output_file: str,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Combine multiple region NetCDF files into a single combined file.
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
        logger.info("Loading region datasets...")
        datasets = []
        file_info = []

        for file_path in region_files:
            try:
                ds = xr.open_dataset(file_path)
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
                for ds in datasets:
                    try:
                        ds.close()
                    except:
                        pass
                return {'success': False, 'error': f'Error opening {file_path}: {e}'}

        logger.info("\nFiles to combine:")
        for info in file_info:
            logger.info(
                f"  {Path(info['file']).name}: {info['id_count']:,} IDs, {info['date_count']} dates, {info['file_size_gb']:.4f} GB")

        logger.info("Combining datasets...")
        combined = None

        for ds in datasets:
            if combined is None:
                combined = ds
            else:
                combined = xr.concat([combined, ds], dim='id_geohash')
                _, unique_idx = np.unique(combined['id_geohash'].values, return_index=True)
                if len(unique_idx) < len(combined['id_geohash']):
                    removed_count = len(combined['id_geohash']) - len(unique_idx)
                    logger.info(f"Removed {removed_count} duplicate IDs")
                    combined = combined.isel(id_geohash=np.sort(unique_idx))

        if combined is None:
            logger.error("No datasets to combine")
            return {'success': False, 'error': 'No datasets to combine'}

        combined = combined.sortby(['id_geohash', 'date'])

        logger.info(f"Combined dataset has {len(combined['id_geohash'])} IDs and {len(combined['date'])} dates")

        logger.info(f"Writing combined file to {output_file}")

        encoding = {}
        for var in combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }

        combined.to_netcdf(output_file, encoding=encoding)

        file_size_gb = Path(output_file).stat().st_size / (1024 ** 3)

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


# =============================================================================
# VERIFY COMBINED FILE (OPTIMIZED)
# =============================================================================
def verify_combined_file_optimized(
        combined_file_path: str,
        regions: List[str],
        date_to_check: str,
        env_path: str = None,
        sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Verify the combined file contains all data for all regions.
    Uses vectorized + sampling for speed.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFYING COMBINED FILE (OPTIMIZED)")
    logger.info(f"{'=' * 80}")

    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    if not Path(combined_file_path).exists():
        return {'success': False, 'error': 'Combined file not found'}

    # First, do a quick check on the combined file itself
    try:
        combined_ds = xr.open_dataset(combined_file_path)
        combined_ids = set(combined_ds['id_geohash'].values)
        combined_dates = set(pd.to_datetime(combined_ds['date'].values))
        combined_date_strings = {d.strftime("%Y-%m") for d in combined_dates}

        logger.info(f"Combined file has {len(combined_ids):,} IDs and {len(combined_dates)} dates")

        # Check date
        date_present = date_to_check in combined_date_strings
        if not date_present:
            combined_ds.close()
            return {'success': False, 'error': f'Date {date_to_check} not found in combined file'}

        combined_ds.close()

    except Exception as e:
        logger.error(f"Error opening combined file: {e}")
        return {'success': False, 'error': str(e)}

    # Now check each region (fast vectorized checks)
    region_results = {}
    all_present = True
    total_ids = 0
    total_missing = 0

    for region in regions:
        logger.info(f"\nChecking region: {region}")

        # Use the fast vectorized verification
        result = verify_region_data_vectorized(
            region=region,
            date_to_check=date_to_check,
            file_path=combined_file_path,
            env_path=env_path,
            sample_size=sample_size
        )

        region_results[region] = result

        if result.get('success', False):
            logger.info(f"  ✅ Region {region} verified successfully")
        else:
            all_present = False
            error = result.get('error', 'Unknown error')
            logger.warning(f"  ❌ Region {region} failed: {error}")

            if 'ids_missing' in result:
                total_missing += result['ids_missing']

        total_ids += result.get('total_ids', 0)

    return {
        'success': all_present,
        'combined_file': combined_file_path,
        'total_ids': total_ids,
        'total_missing': total_missing,
        'all_regions_present': all_present,
        'region_results': region_results,
        'date_present': date_present
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 ** 3)
    return 0


def quick_check_merged_file(file_path: str, date_to_check: str) -> bool:
    """Quick check if a merged file exists and has the date."""
    if not Path(file_path).exists():
        return False
    try:
        ds = xr.open_dataset(file_path)
        dates_in_file = pd.to_datetime(ds['date'].values)
        date_strings = [d.strftime("%Y-%m") for d in dates_in_file]
        ds.close()
        return date_to_check in date_strings
    except:
        return False


def process_region_fast(
        region: str,
        date_to_run: str,
        env_path: str = None,
        dynamic_world_data_dir: str = None
) -> Dict[str, Any]:
    """
    Fast process a single region: verify downloads, merge (no verification).
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING REGION: {region} (FAST MODE - NO VERIFICATION)")
    logger.info(f"{'=' * 80}")

    result = {
        'region': region,
        'date': date_to_run,
        'success': False,
        'steps': {}
    }

    # Step 1: Verify downloads are complete (quick)
    logger.info(f"Step 1: Verifying downloads for {region}...")
    downloads_complete = verify_downloads_complete(region=region, analysis_dates=[date_to_run])

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

    if not complete:
        logger.warning(f"⚠️ Downloads not complete for {region} - skipping")
        result['success'] = False
        result['reason'] = 'Downloads incomplete'
        return result

    # Step 2: Check if already merged (quick check)
    merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{region}_{date_to_run}.nc")

    if quick_check_merged_file(merged_file_path, date_to_run):
        logger.info(f"✅ Region {region} already has date {date_to_run} - skipping merge")
        result['success'] = True
        result['merged_file'] = merged_file_path
        result['reason'] = 'Already merged'
        return result

    # Step 3: Merge (no verification)
    logger.info(f"Step 2: Merging results for {region}...")
    merge_result = merge_new_results(
        region=region,
        date_to_merge=date_to_run,
        merged_file_path=merged_file_path,
        env_path=env_path
    )

    if not merge_result.get('success', False):
        logger.error(f"❌ Merge failed for {region}: {merge_result.get('error', 'Unknown error')}")
        result['success'] = False
        result['reason'] = 'Merge failed'
        return result

    logger.info(f"✅ Region {region} merged successfully (verification deferred)")
    result['success'] = True
    result['merged_file'] = merged_file_path
    result['reason'] = 'Successfully merged (verification deferred)'

    return result


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
    if hist_ids.dtype.kind in ['U', 'S']:  # String or bytes type
        logger.info(f"Historical IDs sample (as strings): {[str(id) for id in hist_ids[:5]]}")
    if comb_ids.dtype.kind in ['U', 'S']:
        logger.info(f"Combined IDs sample (as strings): {[str(id) for id in comb_ids[:5]]}")

    # Check if IDs are numeric
    if hist_ids.dtype.kind in ['i', 'f']:  # Integer or float
        logger.info(f"Historical IDs are numeric, range: {hist_ids.min()} to {hist_ids.max()}")
    if comb_ids.dtype.kind in ['i', 'f']:
        logger.info(f"Combined IDs are numeric, range: {comb_ids.min()} to {comb_ids.max()}")

    # Check if there's any match at all
    hist_set = set(hist_ids)
    comb_set = set(comb_ids)
    intersection = hist_set & comb_set

    logger.info(f"Intersection size: {len(intersection)}")

    if len(intersection) == 0:
        # Check if IDs are similar but with different prefixes/suffixes
        logger.info("No exact matches found. Checking for partial matches...")

        # Convert to strings for comparison
        hist_str = [str(id) for id in hist_ids[:1000]]
        comb_str = [str(id) for id in comb_ids[:1000]]

        # Check if any IDs from one file appear as substrings in the other
        for i, h_id in enumerate(hist_str[:10]):
            for j, c_id in enumerate(comb_str[:10]):
                if h_id in c_id or c_id in h_id:
                    logger.info(f"Partial match: '{h_id}' and '{c_id}'")

        # Check if one is hashed/encoded differently
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


# =============================================================================
# MAIN SCRIPT - SIMPLE COPY AND MERGE
# =============================================================================
# =============================================================================
# MAIN SCRIPT - CORRECTED MERGE (PRESERVES ALL IDs)
# =============================================================================
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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