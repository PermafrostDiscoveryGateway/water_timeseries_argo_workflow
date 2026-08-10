from utils.helper_functions import verify_merged_netcdf, enable_memory_tracking, log_memory_usage
import sys
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
import os
import glob
import dask
import time
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


def _get_id_chunk_size(default: int = 2000) -> int:
    """Read the lake_chunk_size env var (already wired into every Argo workflow pod)
    so NetCDFs are opened as dask-backed, id_geohash-chunked datasets instead of
    fully into memory as numpy arrays."""
    try:
        return int(os.environ.get('lake_chunk_size', default))
    except (TypeError, ValueError):
        return default


def _configure_dask_for_low_memory():
    """Bound the default in-process dask scheduler.

    The Argo pod sets DASK_DISTRIBUTED__WORKER__MEMORY__* env vars, but nothing
    in this script starts a dask.distributed cluster, so those spill-to-disk
    thresholds are never actually consulted -- xarray falls back to the plain
    threaded scheduler, which has no memory awareness at all and will happily
    materialize as many chunks in memory at once as there are CPUs, then let
    the OS OOM-kill the pod. Capping worker count bounds how many chunks can
    be resident at once. split_large_chunks guards against xarray/dask
    silently collapsing an out-of-order reindex (aligning the new month's IDs
    onto the historical ID order) into one giant, unchunked array.
    """
    num_workers = int(os.environ.get('dask_num_workers', 2))
    dask.config.set(scheduler='threads', num_workers=num_workers)
    dask.config.set({'array.slicing.split_large_chunks': True})
    logger.info(f"Dask configured: threaded scheduler capped at {num_workers} concurrent workers")


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


def check_region_merges_completed(dynamic_world_data_dir: str, date_to_run: str, regions: list,
                                  require_all_success: bool = True) -> Dict[str, Any]:
    """
    Check if all region merge files exist AND are fully written/valid for the given date.

    Returns:
        dict: {
            'all_completed': bool,
            'missing_files': list,
            'invalid_files': list,
            'processing_files': list,
            'details': dict
        }
    """
    merge_dir = os.path.join(dynamic_world_data_dir, 'merge')

    # Pattern for region merge files
    date_merge_pattern = f"dw_*_{date_to_run}.nc"
    region_merge_files = glob.glob(os.path.join(merge_dir, date_merge_pattern))

    # Expected files
    expected_files = [f"dw_{region}_{date_to_run}.nc" for region in regions]
    existing_files = [os.path.basename(f) for f in region_merge_files]

    # Check for missing files
    missing_files = [f for f in expected_files if f not in existing_files]

    # Check for files that might be incomplete (being written)
    processing_files = []
    invalid_files = []
    valid_files = []

    for region_file in region_merge_files:
        try:
            # Check if file is still being written (use is_file_ready from helper_functions)
            from utils.helper_functions import is_file_ready
            filepath = Path(region_file)

            # First check: is the file accessible and non-empty?
            if not filepath.exists() or filepath.stat().st_size == 0:
                logger.warning(f"File {filepath.name} is empty or missing")
                invalid_files.append(filepath.name)
                continue

            # Second check: is the file being written to?
            if not is_file_ready(str(filepath), wait_seconds=0.5, checks=10):
                logger.warning(f"File {filepath.name} appears to be in the process of being written")
                processing_files.append(filepath.name)
                continue

            # Third check: verify the NetCDF file is valid and complete
            verify_result = verify_merged_netcdf(str(filepath))
            if verify_result.get('success', False):
                valid_files.append(filepath.name)
                logger.info(f"✅ {filepath.name}: valid ({verify_result.get('id_count', 0):,} IDs)")
            else:
                logger.warning(
                    f"❌ {filepath.name}: verification failed - {verify_result.get('error', 'Unknown error')}")
                invalid_files.append(filepath.name)

        except Exception as e:
            logger.warning(f"Could not verify {region_file}: {e}")
            invalid_files.append(os.path.basename(region_file))

    # Determine overall status
    all_files_present = len(missing_files) == 0
    no_processing_files = len(processing_files) == 0
    no_invalid_files = len(invalid_files) == 0

    all_completed = all_files_present and no_processing_files and no_invalid_files

    result = {
        'all_completed': all_completed,
        'missing_files': missing_files,
        'processing_files': processing_files,
        'invalid_files': invalid_files,
        'valid_files': valid_files,
        'details': {
            'total_expected': len(expected_files),
            'total_existing': len(existing_files),
            'total_valid': len(valid_files)
        }
    }

    if all_completed:
        logger.info(f"✅ All {len(regions)} region merge files are complete and valid for {date_to_run}")
    else:
        if missing_files:
            logger.warning(f"Missing region merge files for {date_to_run}: {missing_files}")
        if processing_files:
            logger.warning(f"Region merge files still being written for {date_to_run}: {processing_files}")
        if invalid_files:
            logger.warning(f"Invalid region merge files for {date_to_run}: {invalid_files}")

    return result


def wait_for_regions_to_complete(
        dynamic_world_data_dir: str,
        date_to_run: str,
        regions: list,
        max_wait_minutes: int = 30,
        check_interval_seconds: int = 30
) -> bool:
    """
    Wait for all regions to complete their processing for the given date.

    Args:
        dynamic_world_data_dir: Base directory containing merge folder
        date_to_run: Date in "YYYY-MM" format
        regions: List of region names
        max_wait_minutes: Maximum time to wait (in minutes)
        check_interval_seconds: How often to check (in seconds)

    Returns:
        bool: True if all regions completed within the time limit, False otherwise
    """
    logger.info("=" * 80)
    logger.info(f"WAITING FOR REGION PROCESSING TO COMPLETE FOR {date_to_run}")
    logger.info("=" * 80)
    logger.info(f"Monitoring {len(regions)} regions")
    logger.info(f"Max wait: {max_wait_minutes} minutes")
    logger.info(f"Check interval: {check_interval_seconds} seconds")

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60

    while True:
        elapsed = time.time() - start_time

        # Check if all regions are complete
        status = check_region_merges_completed(dynamic_world_data_dir, date_to_run, regions)

        if status['all_completed']:
            logger.info(f"✅ All regions completed processing for {date_to_run}!")
            logger.info(f"   Time elapsed: {elapsed / 60:.1f} minutes")
            return True

        # Check if we've exceeded max wait time
        if elapsed >= max_wait_seconds:
            logger.error(f"❌ Timeout: Regions did not complete within {max_wait_minutes} minutes")
            logger.error(f"   Missing: {status['missing_files']}")
            logger.error(f"   Processing: {status['processing_files']}")
            logger.error(f"   Invalid: {status['invalid_files']}")
            return False

        # Log current status
        logger.info(f"⏳ Waiting for regions... ({elapsed / 60:.1f}/{max_wait_minutes} min)")
        if status['missing_files']:
            logger.info(f"   Missing: {len(status['missing_files'])} files")
        if status['processing_files']:
            logger.info(f"   Still processing: {len(status['processing_files'])} files")
        if status['invalid_files']:
            logger.info(f"   Invalid: {len(status['invalid_files'])} files")

        # Wait before checking again
        time.sleep(check_interval_seconds)


def combine_region_files(
        region_files: List[str],
        output_file: str,
        env_path: str = None
) -> Dict[str, Any]:
    """
    Combine multiple region NetCDF files into a single combined file.
    Memory-optimized version.
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
        id_chunk = _get_id_chunk_size()
        logger.info(f"Loading region datasets (dask-chunked, id_geohash={id_chunk})...")

        # Open all datasets lazily with dask
        datasets = []
        file_info = []

        # Use a smaller chunk size for the combine operation
        combine_chunk = min(id_chunk, 500)  # Smaller chunks for combining

        for file_path in region_files:
            try:
                # Open with smaller chunks to reduce memory pressure
                ds = xr.open_dataset(
                    file_path,
                    chunks={'id_geohash': combine_chunk, 'date': -1}
                )
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
                f"  {Path(info['file']).name}: {info['id_count']:,} IDs, {info['date_count']} dates, {info['file_size_gb']:.4f} GB"
            )

        logger.info("Combining datasets lazily...")
        if not datasets:
            logger.error("No datasets to combine")
            return {'success': False, 'error': 'No datasets to combine'}

        # Use concat with dask to avoid loading everything into memory
        combined = xr.concat(datasets, dim='id_geohash', combine='nested')

        # Close the original datasets to free memory
        for ds in datasets:
            try:
                ds.close()
            except:
                pass
        datasets = None
        gc.collect()

        # Remove duplicates using dask operations (lazy)
        # Instead of loading all IDs into memory, use dask's unique operation
        logger.info("Removing duplicate IDs...")

        # Get unique IDs using dask - this is still memory intensive but less so
        # because dask can chunk the operation
        unique_ids = combined['id_geohash'].unique().compute()

        if len(unique_ids) < len(combined['id_geohash']):
            removed_count = len(combined['id_geohash']) - len(unique_ids)
            logger.info(f"Removed {removed_count} duplicate IDs")

            # Use where and drop to filter - this is more memory efficient
            mask = combined['id_geohash'].isin(unique_ids)
            # Note: This still loads data, but dask handles it in chunks

            # Alternative: Use groupby first to avoid loading all IDs
            # This is a more memory-efficient way to deduplicate
            combined = combined.drop_duplicates(dim='id_geohash')

        # Sort by IDs and date
        logger.info("Sorting combined dataset...")
        combined = combined.sortby(['id_geohash', 'date'])

        # Persist to disk with chunked writing to avoid OOM
        logger.info(f"Writing combined file to {output_file}")

        # Use encoding with compression
        # Chunk sizes must not exceed the actual dimension sizes (netCDF4 rejects that)
        n_ids = combined.sizes['id_geohash']
        n_dates = combined.sizes['date']
        encoding = {}
        for var in combined.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True,
                'chunksizes': (min(id_chunk, 500, n_ids), n_dates)  # Chunk for writing
            }

        # Write in chunks to avoid memory issues
        # Use compute with chunked writing
        combined.to_netcdf(
            output_file,
            encoding=encoding,
            unlimited_dims=['date']  # Allow date dimension to grow
        )

        # Get final file size
        file_size_gb = Path(output_file).stat().st_size / (1024 ** 3)

        # Clean up
        combined.close()
        gc.collect()

        result = {
            'success': True,
            'file_path': output_file,
            'id_count': len(unique_ids),
            'date_count': len(combined['date']) if 'date' in combined.dims else 0,
            'file_size_gb': round(file_size_gb, 4),
            'files_combined': len(file_info),
            'file_info': file_info
        }

        logger.info(f"✅ Combined file created successfully!")
        logger.info(f"  File: {output_file}")
        logger.info(f"  IDs: {result['id_count']:,}")
        logger.info(f"  Size: {result['file_size_gb']:.4f} GB")

        return result

    except Exception as e:
        logger.error(f"Error combining files: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def merge_historical_file(
        historical_file: str,
        combined_file: str,
        output_file: str,
        id_chunk: int = None
) -> Dict[str, Any]:
    """
    Memory-optimized merge of historical and combined files.
    Uses chunked processing to avoid loading entire files into memory.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("MERGING HISTORICAL AND COMBINED FILES")
    logger.info(f"{'=' * 80}")

    if id_chunk is None:
        id_chunk = _get_id_chunk_size()

    # Use smaller chunks for merging
    merge_chunk = min(id_chunk, 500)

    try:
        # Open files with dask and chunking
        logger.info(f"Opening historical file with chunk size {merge_chunk}...")
        hist_ds = xr.open_dataset(
            historical_file,
            chunks={'id_geohash': merge_chunk, 'date': -1}
        )

        logger.info(f"Opening combined file with chunk size {merge_chunk}...")
        comb_ds = xr.open_dataset(
            combined_file,
            chunks={'id_geohash': merge_chunk, 'date': -1}
        )

        # Get dates without loading all data
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
            return {'success': True, 'dates_added': [], 'message': 'No new dates'}

        # Filter combined to only new dates
        new_date_objects = [pd.Timestamp(f"{d}-01") for d in dates_to_add]
        comb_ds_filtered = comb_ds.sel(date=new_date_objects)

        # Close the original combined dataset to free memory
        comb_ds.close()
        gc.collect()

        # Get historical IDs without loading all data
        hist_ids = hist_ds['id_geohash'].values

        # OPTIMIZATION: Instead of reindexing the entire combined dataset,
        # we'll align during the concat operation
        logger.info(f"Historical has {len(hist_ids):,} IDs")
        logger.info(f"Combined filtered has {len(comb_ds_filtered['id_geohash']):,} IDs")

        # Ensure both datasets have the same IDs
        # Use align with join='inner' to keep only common IDs for the combined data
        # This is more memory efficient than reindex
        logger.info("Aligning datasets...")

        # First, align the combined data to historical IDs
        # This creates a new dataset with the same ID dimension as historical
        # Missing IDs will be NaN
        comb_ds_aligned = comb_ds_filtered.reindex(
            id_geohash=hist_ids,
            method=None
        )

        # Close filtered dataset
        comb_ds_filtered.close()
        gc.collect()

        # Verify IDs match (lazy check - only checks the first few)
        hist_ids_sample = hist_ids[:5]
        comb_ids_sample = comb_ds_aligned['id_geohash'].values[:5]

        if not np.array_equal(hist_ids_sample, comb_ids_sample):
            logger.error("ID mismatch after alignment!")
            hist_ds.close()
            comb_ds_aligned.close()
            return {'success': False, 'error': 'ID mismatch'}

        # Now concatenate along date dimension
        logger.info("Concatenating datasets...")
        merged_ds = xr.concat([hist_ds, comb_ds_aligned], dim='date')
        merged_ds = merged_ds.sortby('date')

        # Remove duplicate dates if any (lazy operation)
        dates = merged_ds['date'].values
        _, unique_idx = np.unique(dates, return_index=True)
        if len(unique_idx) < len(dates):
            removed = len(dates) - len(unique_idx)
            logger.info(f"Removed {removed} duplicate dates")
            merged_ds = merged_ds.isel(date=np.sort(unique_idx))

        # Close the component datasets
        hist_ds.close()
        comb_ds_aligned.close()
        gc.collect()

        # Write the merged file in chunks to avoid OOM
        logger.info(f"Writing merged file to {output_file}...")

        # Chunk sizes must not exceed the actual dimension sizes (netCDF4 rejects that)
        n_ids = merged_ds.sizes['id_geohash']
        n_dates = merged_ds.sizes['date']
        encoding = {}
        for var in merged_ds.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': 4,
                'shuffle': True,
                'chunksizes': (min(id_chunk, 500, n_ids), n_dates)  # Chunk by IDs
            }

        # Write with chunked output
        merged_ds.to_netcdf(
            output_file,
            encoding=encoding,
            unlimited_dims=['date']  # Allow date dimension to grow
        )

        # Get final size
        final_size_gb = Path(output_file).stat().st_size / (1024 ** 3)

        # Clean up
        merged_ds.close()
        gc.collect()

        result = {
            'success': True,
            'file_path': output_file,
            'id_count': len(hist_ids),
            'date_count': len(merged_ds['date']),
            'file_size_gb': round(final_size_gb, 4),
            'dates_added': dates_to_add
        }

        logger.info(f"✅ Merged file created successfully!")
        logger.info(f"  File: {output_file}")
        logger.info(f"  IDs: {result['id_count']:,}")
        logger.info(f"  Dates: {result['date_count']}")
        logger.info(f"  Size: {result['file_size_gb']:.4f} GB")
        logger.info(f"  Added dates: {dates_to_add}")

        return result

    except MemoryError as e:
        logger.error(f"Memory error while merging files: {e}")
        if Path(output_file).exists():
            Path(output_file).unlink()
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Error merging files: {e}")
        if Path(output_file).exists():
            Path(output_file).unlink()
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    logger.debug(f"Beginning historical run for ALL regions (fast mode)")
    enable_memory_tracking()
    log_memory_usage("Program start")
    _configure_dask_for_low_memory()

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
    if os.environ['test_run'] == 'True':
        all_regions = list(utils.region_boundaries.get_small_regions().keys())
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

    # ========== STEP 1: Wait for region merges to complete ==========
    logger.info("=" * 80)
    logger.info("STEP 1: Waiting for region processing to complete")
    logger.info("=" * 80)

    # Wait for all regions to complete processing
    # Adjust max_wait_minutes as needed (default: 30 minutes)
    max_wait_minutes = int(os.environ.get('merge_wait_minutes', 30))

    regions_completed = wait_for_regions_to_complete(
        dynamic_world_data_dir=dynamic_world_data_dir,
        date_to_run=date_to_run,
        regions=all_regions,
        max_wait_minutes=max_wait_minutes,
        check_interval_seconds=30
    )

    if not regions_completed:
        logger.error(f"❌ Not all regions completed processing for {date_to_run}. Aborting merge.")
        return

    # ========== STEP 2: Create combined file if it doesn't exist ==========
    logger.info("=" * 80)
    logger.info("STEP 2: Creating combined file from region files")
    logger.info("=" * 80)

    combined_file_name = f"dynamic_world_combined_{date_to_run}.nc"
    combined_file_path = os.path.join(dynamic_world_data_dir, 'merge', combined_file_name)

    if not os.path.exists(combined_file_path):
        # Get all region files for this date
        merge_dir = os.path.join(dynamic_world_data_dir, 'merge')
        region_files = glob.glob(os.path.join(merge_dir, f"dw_*_{date_to_run}.nc"))

        if not region_files:
            logger.error(f"No region files found for date {date_to_run}")
            return

        logger.info(f"Found {len(region_files)} region files to combine")

        # Combine the region files
        combine_result = combine_region_files(
            region_files=region_files,
            output_file=combined_file_path,
            env_path=env_path
        )

        if not combine_result.get('success', False):
            logger.error(f"Failed to create combined file: {combine_result.get('error', 'Unknown error')}")
            return

        logger.info("✅ Combined file created successfully!")
    else:
        logger.info(f"Combined file {combined_file_name} already exists. Proceeding to merge...")

    # ========== STEP 3: Merge with historical file ==========
    logger.info("=" * 80)
    logger.info("STEP 3: Merging combined file with historical data")
    logger.info("=" * 80)

    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))

    # Find the most recent .nc file (excluding the merge directory)
    # Use the helper from your earlier code
    from pathlib import Path
    def find_most_recent_nc_file(directory):
        nc_files = list(Path(directory).glob("*.nc"))
        if not nc_files:
            return None
        return max(nc_files, key=lambda f: f.stat().st_mtime)

    original_most_recent = find_most_recent_nc_file(dynamic_world_data_dir)
    if original_most_recent is None:
        logger.error("No .nc files found in the dynamic_world_data directory")
        return

    original_most_recent_dynamic_world_file = str(original_most_recent)

    # Get file sizes for logging
    historical_file_size_gb = Path(original_most_recent_dynamic_world_file).stat().st_size / (1024 ** 3)
    logger.info(f"Historical file: {os.path.basename(original_most_recent_dynamic_world_file)}")
    logger.info(f"Historical file size: {historical_file_size_gb:.2f} GB")

    if os.path.exists(combined_file_path):
        combined_file_size_gb = Path(combined_file_path).stat().st_size / (1024 ** 3)
        logger.info(f"Combined file size: {combined_file_size_gb:.2f} GB")

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

        id_chunk = _get_id_chunk_size()
        log_memory_usage("Before merge_historical_file")
        merge_result = merge_historical_file(
            historical_file=original_most_recent_dynamic_world_file,
            combined_file=combined_file_path,
            output_file=new_historical_file_path,
            id_chunk=id_chunk
        )
        log_memory_usage("After merge_historical_file")

        if not merge_result.get('success', False):
            logger.error(f"Failed to merge historical file: {merge_result.get('error', 'Unknown error')}")
            return

        if not merge_result.get('dates_added'):
            logger.info(merge_result.get('message', 'No new dates to add.'))
            return

        # Also run the standard verification from helper_functions
        verify_result = verify_merged_netcdf(new_historical_file_path)
        if verify_result.get('success', False):
            logger.info("✅ Merged file verification passed")
        else:
            logger.warning(f"⚠️ Merged file verification failed: {verify_result.get('error', 'Unknown error')}")

        expected_ids = merge_result['id_count']
        verified_ids = verify_result.get('id_count', 'unknown')
        if verified_ids != 'unknown' and verified_ids != expected_ids:
            logger.warning(f"⚠️ ID count mismatch! Expected {expected_ids:,}, got {verified_ids:,}")

        logger.info("=" * 80)
        logger.info("✅ SUCCESS: New historical file created!")
        logger.info("=" * 80)
        logger.info(f"  Original historical file (KEPT): {original_most_recent_dynamic_world_file}")
        logger.info(f"  Combined file (KEPT): {combined_file_path}")
        logger.info(f"  NEW merged file: {new_historical_file_path}")
        logger.info(f"  File size: {merge_result['file_size_gb']:.4f} GB")
        logger.info(f"  Added dates: {merge_result['dates_added']}")
        logger.info(f"  Total IDs: {expected_ids:,}")
        logger.info(f"  Total dates: {merge_result['date_count']}")
        logger.info("=" * 80)
        logger.info("⚠️  No files were deleted. All original files are preserved.")
        logger.info("=" * 80)
    else:
        logger.info(f"Combined file {combined_file_name} does not exist. Nothing to merge.")


if __name__ == "__main__":
    main()