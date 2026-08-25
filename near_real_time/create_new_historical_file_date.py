"""Create a new historical file for a single, explicitly-named month.

Unlike create_new_historical_file.py (which always targets two months back, gated to
only run during the summer processing window), this script takes an explicit target
month (e.g. "2026-06") as a command-line argument, so it can be used to backfill/fix
a month that was missed or came out incomplete.

It also lets you pin the historical file to merge into, instead of always picking the
most recently modified .nc file in dynamic_world_data: if the .env sets
`original_most_recent_dynamic_world_file` to a path, that file is used directly.

Usage:
    python create_new_historical_file_date.py 2026-06 [path/to/.env]
"""
from create_new_historical_file import (
    _get_id_chunk_size,
    _configure_dask_for_low_memory,
    wait_for_regions_to_complete,
    combine_region_files,
    merge_historical_file,
)
from utils.helper_functions import verify_merged_netcdf, enable_memory_tracking, log_memory_usage
import sys
import re
import os
import glob
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATE_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def find_most_recent_nc_file(directory):
    nc_files = list(Path(directory).glob("*.nc"))
    if not nc_files:
        return None
    return max(nc_files, key=lambda f: f.stat().st_mtime)


def resolve_historical_file(dynamic_world_data_dir: str):
    """Use the historical file pinned via the .env `original_most_recent_dynamic_world_file`
    key if present, otherwise fall back to the most recently modified .nc file."""
    pinned_path = os.environ.get('original_most_recent_dynamic_world_file')
    if pinned_path:
        if not os.path.exists(pinned_path):
            logger.error(f"original_most_recent_dynamic_world_file is set but does not exist: {pinned_path}")
            return None
        logger.info(f"Using pinned historical file from .env: {pinned_path}")
        return pinned_path

    most_recent = find_most_recent_nc_file(dynamic_world_data_dir)
    if most_recent is None:
        return None
    return str(most_recent)


def main():
    logger.debug("=" * 80)
    logger.debug("CREATE_NEW_HISTORICAL_FILE_DATE.PY STARTED")
    logger.debug("=" * 80)
    enable_memory_tracking()
    log_memory_usage("Program start")
    _configure_dask_for_low_memory()

    if len(sys.argv) < 2:
        logger.error('Usage: python create_new_historical_file_date.py <YYYY-MM> [path/to/.env]')
        sys.exit(1)

    date_to_run = sys.argv[1]
    if not DATE_PATTERN.match(date_to_run):
        logger.error(f"Invalid date '{date_to_run}' - expected format YYYY-MM, e.g. 2026-06")
        sys.exit(1)

    env_path = None
    if len(sys.argv) > 2:
        env_path = sys.argv[2]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    import utils.region_boundaries
    boundaries = utils.region_boundaries.get_region_boundaries()
    all_regions = list(boundaries.keys())
    logger.info(f"Available regions: {all_regions}")

    dynamic_world_data_dir = os.environ['dynamic_world_data']

    logger.info(f"🎯 Target date: {date_to_run}")

    # ========== STEP 1: Wait for region merges to complete ==========
    logger.info("=" * 80)
    logger.info("STEP 1: Waiting for region processing to complete")
    logger.info("=" * 80)

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
        merge_dir = os.path.join(dynamic_world_data_dir, 'merge')
        region_files = glob.glob(os.path.join(merge_dir, f"dw_*_{date_to_run}.nc"))

        if not region_files:
            logger.error(f"No region files found for date {date_to_run}")
            return

        logger.info(f"Found {len(region_files)} region files to combine")

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

    original_most_recent_dynamic_world_file = resolve_historical_file(dynamic_world_data_dir)
    if original_most_recent_dynamic_world_file is None:
        logger.error("No .nc files found in the dynamic_world_data directory")
        return

    historical_file_size_gb = Path(original_most_recent_dynamic_world_file).stat().st_size / (1024 ** 3)
    logger.info(f"Historical file: {os.path.basename(original_most_recent_dynamic_world_file)}")
    logger.info(f"Historical file size: {historical_file_size_gb:.2f} GB")

    if not os.path.exists(combined_file_path):
        logger.info(f"Combined file {combined_file_name} does not exist. Nothing to merge.")
        return

    combined_file_size_gb = Path(combined_file_path).stat().st_size / (1024 ** 3)
    logger.info(f"Combined file size: {combined_file_size_gb:.2f} GB")

    new_historical_file_name = f"dynamic_world_historical_{date_to_run}.nc"
    new_historical_file_path = os.path.join(dynamic_world_data_dir, new_historical_file_name)

    try:
        statvfs = os.statvfs(dynamic_world_data_dir)
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 ** 3)
        logger.info(f"Free disk space: {free_space_gb:.2f} GB")

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


if __name__ == "__main__":
    main()
