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
    original_file = "/Users/helium/Desktop/dynamic_world/data/lakes_dw_V2d.nc"
    merged_file = "/Users/helium/Desktop/dynamic_world/data/historical_data_20260703_140615.nc"

    verify_result = verify_merge_result(original_file, merged_file)
    logger.debug(f"After verify results")
    time.sleep(60*10)

    logger.debug(f"Beginning historical run")
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)

    CURRENT_REGION = sys.argv[1]

    SHOULD_RUN = False

    summer_months = [6, 7, 8, 9]
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    missing_dates_from_netcdf = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_most_recent_dynamic_world_file)

    # result_test = verify_merged_data(
    #     file_path="/Users/helium/Desktop/dynamic_world/data/historical_data_20260702_170636.nc",
    #     region="TEST",
    #     dates=["2025-07", "2025-08", "2025-09", "2026-06"]
    # )
    #
    # result_eurasia3 = verify_merged_data(
    #     file_path="/Users/helium/Desktop/dynamic_world/data/historical_data_20260702_170636.nc",
    #     region="EURASIA3",
    #     dates=["2025-07", "2025-08", "2025-09", "2026-06"]
    # )

    # TODO remove later
    # After merging, compare original vs new

    # Usage
    # original_file = "/Users/helium/Desktop/dynamic_world/data/lakes_dw_V2d.nc"
    # merged_file = "/Users/helium/Desktop/dynamic_world/data/historical_data_20260702_170636.nc"
    #
    # verify_result = verify_merge_result(original_file, merged_file)
    logger.debug(f"Got verify result")
    # TODO remove later
    expected_dates = generate_expected_dates(start_year=2025)
    expected_dates = expected_dates[:-2]

    TODAY =  datetime.now()
    TODAY_MONTH = TODAY.month
    if TODAY_MONTH - 1 in summer_months:
        logger.debug(f"TODAY MONTH: {TODAY_MONTH}")
        logger.debug(f"Last month: {TODAY.month - 1} checking to see if we should run")
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    # TODO get timestamp value of last month
    timestamp_to_run = [pd.Timestamp(date(datetime.now().year, TODAY_MONTH -1, 1))]
    date_to_run = [datetime(TODAY.year, TODAY_MONTH -1, 1).strftime("%Y-%m")]
    logger.debug(f"timestamp_to_run: {timestamp_to_run}")


if __name__ == "__main__":
    main()