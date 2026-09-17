from utils.helper_functions import verify_downloads_complete , download_near_real_time_region_dates, compare_netcdf_files
from utils.date_gate import is_test_run, most_recent_summer_month
import sys
import utils.download_utils as download_new_dynamic_world_data
from loguru import logger
from datetime import date, datetime
from dotenv import load_dotenv
import os
import glob
import pandas as pd
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# After merging, compare original vs new
def verify_merge_result(original_file, merged_file):
    """
    Compare original and merged files and log the results.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("VERIFYING MERGE RESULT")
    logger.info(f"{'=' * 80}")
    logger.info(f"Original: {original_file}")
    logger.info(f"Merged:   {merged_file}")

    result = compare_netcdf_files(
        file1_path=original_file,
        file2_path=merged_file,
        sample_ids=5,
        verbose=True
    )

    # Log summary
    if result['summary']['successful']:
        logger.info("✅ MERGE VERIFICATION PASSED")
        logger.info(f"   Size: {result['summary']['file1_size_gb']:.2f}GB → {result['summary']['file2_size_gb']:.2f}GB")
        if result['summary'].get('new_dates_added', 0) > 0:
            logger.info(f"   New dates added: {result['summary']['new_dates_added']}")
        if result['summary'].get('new_ids_added', 0) > 0:
            logger.info(f"   New IDs added: {result['summary']['new_ids_added']}")
    else:
        logger.error("❌ MERGE VERIFICATION FAILED")
        for issue in result['summary']['issues']:
            logger.error(f"   Issue: {issue}")

    return result


# Usage



def main():
    exit_code = 0
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
    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
    logger.debug(f"This is the most recent dynamic world file {original_most_recent_dynamic_world_file}")
    missing_dates_from_netcdf = utils.download_utils.check_missing_data_in_netcdf(original_most_recent_dynamic_world_file)

    downloads_dir = Path(dynamic_world_data_dir) / 'downloads' / REGION
    merge_dir = Path(dynamic_world_data_dir) / 'merge' / REGION

    downloads_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ensured directories exist: {downloads_dir}, {merge_dir}")



    TODAY =  datetime.now()
    TODAY_MONTH = TODAY.month
    target_month = None
    if is_test_run():
        target_month = most_recent_summer_month(TODAY)
        SHOULD_RUN = True
        logger.debug(f"test_run=True - bypassing day-of-month/season gate, using {target_month.strftime('%Y-%m')}")
    elif TODAY_MONTH - 1 in summer_months:
        logger.debug(f"TODAY MONTH: {TODAY_MONTH}")
        logger.debug(f"Last month: {TODAY.month - 1} checking to see if we should run")
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 2:
            SHOULD_RUN = True
            target_month = datetime(TODAY.year, TODAY_MONTH - 1, 1)
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    if SHOULD_RUN:
        timestamp_to_run = [pd.Timestamp(target_month)]
        date_to_run = [target_month.strftime("%Y-%m")]
        logger.debug(f"timestamp_to_run: {timestamp_to_run}")

        downloads_complete = verify_downloads_complete(region=REGION, analysis_dates=date_to_run)
        logger.debug(downloads_complete)
        if downloads_complete.get('need_download', False) or not downloads_complete.get('complete', False):
            logger.debug(f"We should run downloads for {REGION} for {timestamp_to_run}")
            if env_path:
                download_result = download_near_real_time_region_dates(region=REGION, dates_to_download=timestamp_to_run, env_path=env_path )
                logger.debug(f"Download result: {download_result}")
            else:
                logger.debug(f"Missing env path")
                download_result = download_near_real_time_region_dates(region=REGION, dates_to_download=timestamp_to_run)
                logger.debug(f"Download result: {download_result}")

            if not download_result.get('success', False):
                logger.error(f"Download for {REGION} reported failure: {download_result}")
                exit_code = 1
        else:
            logger.debug(f"Already done downloading {REGION} for {date_to_run}")

    else:
        logger.debug(f"Too early in the month to run downloads for {REGION}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())