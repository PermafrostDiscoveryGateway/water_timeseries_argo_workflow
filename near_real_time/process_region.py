from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region , \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
                                     merge_near_real_time_region_v3_simple, \
                 compare_netcdf_files, verify_merged_netcdf, verify_merged_data, merge_near_real_time_region_v3_smart
import sys
import shutil
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

def is_file_ready(filepath, wait_seconds=0.5, checks=20):
    sizes = []
    for _ in range(checks):
        size = os.path.getsize(filepath)
        sizes.append(size)
        time.sleep(wait_seconds)

    # If size hasn't changed, assume writing is done
    return len(set(sizes)) == 1

def main():
    logger.debug(f"Running processing cron job")
    SHOULD_RUN = False
    REGION = os.environ.get("region_name", "TEST")
    summer_months = [6, 7, 8, 9]
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
    logger.debug(f"This is the most recent dynamic world file {original_most_recent_dynamic_world_file}")

    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    if TODAY_MONTH - 1 in summer_months:
        logger.debug(f"TODAY MONTH: {TODAY_MONTH}")
        logger.debug(f"Last month: {TODAY.month - 1} checking to see if we should run")
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    if SHOULD_RUN:
        timestamp_to_run = [pd.Timestamp(date(datetime.now().year, TODAY_MONTH - 1, 1))]
        date_to_run = [datetime(TODAY.year, TODAY_MONTH - 1, 1).strftime("%Y-%m")]
        logger.debug(f"timestamp_to_run: {timestamp_to_run}")
        dates_to_run_string = date_to_run[0].replace('-', '_')
        name_of_final_merge_file = f"{dynamic_world_data_dir}/lakes_dw_Vdc_v2_{dates_to_run_string}.nc"
        logger.debug(f"Checking if the file exists: {name_of_final_merge_file}")
        if os.path.isfile(name_of_final_merge_file):
            logger.debug(f"File for the most recent month exists")
            file_ready_check = is_file_ready(name_of_final_merge_file)
            if file_ready_check:
                logger.debug(f"We will process for region {REGION} and date {date_to_run}")
                process_result = process_near_real_time_region_dates_zarr(region=REGION, current_analysis_dates=timestamp_to_run)
                logger.debug(f"Result is {process_result}")
    else:
        logger.debug(f"Too early in the month to run")