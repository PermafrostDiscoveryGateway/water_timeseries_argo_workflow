from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region , \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
                                    merge_near_real_time_region_v2, merge_near_real_time_region_v3_simple, \
                 compare_netcdf_files, verify_merged_netcdf, verify_merged_data
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

def get_creation_time(filepath):
    """Get file creation time on Linux (birth time) if available"""
    stat_info = os.stat(filepath)
    try:
        # st_birthtime is the actual creation time on Linux
        creation_time = stat_info.st_birthtime
    except AttributeError:
        # Fallback to ctime if birthtime not available
        creation_time = stat_info.st_ctime
    return creation_time

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
    logger.debug(f"Beginning historical run")
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    REGION = os.environ.get("region_name", "TEST")

    SHOULD_RUN = False

    summer_months = [6, 7, 8, 9]
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    for file in all_dynamic_world_files:
        time_created = get_creation_time(file)
        readable_time = datetime.fromtimestamp(time_created)
        logger.debug(f"Netcdf file {file} has creation date of {readable_time}")
        most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: f.stat().st_mtime)
    logger.debug(f"Most recent dynamic world file {most_recent_dynamic_world_file}")
    # missing_dates_from_netcdf = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_most_recent_dynamic_world_file)



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
        date_to_run = [datetime(TODAY.year, TODAY_MONTH -1, 1).strftime("%Y-%m")]
        dates_to_run_string = date_to_run[0].replace('-', '_')
        logger.debug(f"New netcdf file will end with string {dates_to_run_string}")
        logger.debug(f"Checking if we should merge")
        logger.debug(f"Merge if {date_to_run} are downloaded for all regions")
        REGIONS = utils.region_boundaries.get_region_boundaries()
        REGION_NAMES = list(REGIONS.keys())

        regions_downloaded = 0

        for region in REGION_NAMES:
            logger.info(f"Checking if we should merge region: {region}")
            check_result = verify_downloads_complete(region=region, analysis_dates=date_to_run)
            logger.debug(f"Result for region {region}: {check_result['complete']}")
            if check_result['complete']:
                regions_downloaded += 1
        logger.debug(f"How many regions are finished downloading?")
        logger.debug(f"{regions_downloaded} regions downloaded")

        successfully_merged_region_count = 0
        regions_still_to_merge = []

        if regions_downloaded == len(REGION_NAMES):
            for region in REGION_NAMES:
                logger.debug(f"Checking if we already merged region {region} for {date_to_run}")
            for region in REGION_NAMES:
                merge_result = merge_near_real_time_region_v3_simple(region=region, dates_to_merge=date_to_run)
                # TODO use this method to check merged data
                # check_region_data_in_merged_file
                logger.debug(f"Merge result for region {region}: {merge_result['complete']}")
            logger.debug(f"Verifying merge finished for all regions properly")
        else:
            logger.debug(f"not all regions downloaded, do not merge")







if __name__ == "__main__":
    main()