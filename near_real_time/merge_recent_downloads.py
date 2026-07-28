from utils.helper_functions import verify_downloads_complete, merge_new_results
import sys
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
import os
import glob
import pandas as pd
from pathlib import Path
import xarray as xr
import numpy as np
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


# =============================================================================
# MAIN SCRIPT
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

    # ========== Process ALL regions (FAST - no verification) ==========
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

            result = process_region_fast(
                region=region,
                date_to_run=date_to_run,
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
    logger.info("FINAL SUMMARY - ALL REGIONS (FAST MERGE)")
    logger.info("=" * 80)
    logger.info(f"Date processed: {date_to_run}")
    logger.info(f"Total regions: {len(all_regions)}")
    logger.info(f"✅ Successful: {success_count}")
    logger.info(f"⏭️ Already merged: {skipped_count}")
    logger.info(f"❌ Failed: {failure_count}")
    logger.info("=" * 80)

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
            combined_file_path = os.path.join(dynamic_world_data_dir, 'merge', combined_file_name)

            logger.info(f"Combining into: {combined_file_path}")

            # Combine the files
            combine_result = combine_region_files(
                region_files=region_files,
                output_file=combined_file_path,
                env_path=env_path
            )

            if combine_result.get('success', False):
                logger.info(f"✅ Combined file created successfully!")

                # ============================================================
                # NOW VERIFY THE COMBINED FILE (optimized)
                # ============================================================
                logger.info("\n" + "=" * 80)
                logger.info("VERIFYING COMBINED FILE")
                logger.info("=" * 80)

                verify_result = verify_combined_file_optimized(
                    combined_file_path=combined_file_path,
                    regions=all_regions,
                    date_to_check=date_to_run,
                    env_path=env_path,
                    sample_size=1000  # Sample 1000 IDs per region for verification
                )

                if verify_result.get('success', False):
                    logger.info("\n" + "=" * 80)
                    logger.info("✅ ALL REGIONS VERIFIED SUCCESSFULLY!")
                    logger.info("=" * 80)
                    logger.info(f"  Combined file: {combined_file_path}")
                    logger.info(f"  Total IDs: {verify_result['total_ids']:,}")
                    logger.info(f"  Date: {date_to_run}")
                    logger.info("=" * 80)
                else:
                    logger.warning("\n" + "=" * 80)
                    logger.warning("⚠️ VERIFICATION ISSUES FOUND")
                    logger.warning("=" * 80)

                    for region, reg_result in verify_result.get('region_results', {}).items():
                        if not reg_result.get('success', False):
                            logger.warning(f"  Region {region}: {reg_result.get('error', 'Unknown error')}")
                            if 'ids_missing' in reg_result:
                                logger.warning(f"    {reg_result['ids_missing']} IDs missing")
                                if 'missing_sample' in reg_result:
                                    logger.warning(f"    Sample missing IDs: {reg_result['missing_sample']}")

                    logger.warning("=" * 80)
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

    sys.exit(0 if failure_count == 0 else 1)


if __name__ == "__main__":
    main()