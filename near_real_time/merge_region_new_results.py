from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region , \
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
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))





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
    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
    logger.debug(f"This is the most recent dynamic world file {original_most_recent_dynamic_world_file}")
    missing_dates_from_netcdf = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_most_recent_dynamic_world_file)



    TODAY =  datetime.now()
    TODAY_MONTH = TODAY.month
    if TODAY_MONTH - 1 in summer_months:
        logger.debug(f"TODAY MONTH: {TODAY_MONTH}")
        logger.debug(f"Last month: {TODAY.month - 1} checking to see if we should run")
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    if SHOULD_RUN:
        timestamp_to_run = [pd.Timestamp(date(datetime.now().year, TODAY_MONTH -1, 1))]
        date_to_run = [datetime(TODAY.year, TODAY_MONTH -1, 1).strftime("%Y-%m")]
        logger.debug(f"timestamp_to_run: {timestamp_to_run}")
        downloads_complete = verify_downloads_complete(region=REGION, analysis_dates=date_to_run)
        logger.debug(downloads_complete)
        complete = downloads_complete['complete']
        complete_dates = downloads_complete['complete_dates']
        incomplete_dates = downloads_complete['incomplete_dates']
        summary = downloads_complete['summary']
        logger.debug(f"Total expected downloads {summary['total_expected_downloads']}")
        logger.debug(f"Total successful downloads {summary['total_successful_downloads']}")
        total_skipped_and_successful_downloads = summary['total_skipped_downloads'] + summary['total_successful_downloads']
        percent_downloaded = float(total_skipped_and_successful_downloads)/float(summary['total_expected_downloads'])
        logger.debug(f"Percent downloaded: {percent_downloaded}")
        if percent_downloaded > 0.99:
            complete = True
            if complete:
                logger.debug(f"Merge all the results for {REGION} and {date_to_run[0]}")
                # TODO need a method to merge them (just new results
                merged_file_path = os.path.join(dynamic_world_data_dir, f"dw_{REGION}_{date_to_run[0]}.nc")
                merge_new_results(region=REGION, date_to_merge=date_to_run[0], merged_file_path=merged_file_path, env_path=env_path)
                # this should go into a file that is this path
                logger.debug(f"Verify all are the same")
                check = is_all_new_data_in_file(region=REGION, date_to_check=date_to_run[0], merged_file_path=merged_file_path, env_path=env_path)

    else:
        logger.debug(f"Too early in the month to run downloads for {REGION}")




if __name__ == "__main__":
    main()