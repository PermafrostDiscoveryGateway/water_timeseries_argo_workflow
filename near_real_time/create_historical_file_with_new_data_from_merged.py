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
    # ... (keep your existing debug function)


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
    # ... (keep your existing combine function)


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
    # ... (keep your existing merge function)


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