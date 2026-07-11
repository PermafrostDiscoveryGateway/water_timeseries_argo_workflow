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
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")
    logger.debug("=" * 80)
    logger.debug("PROCESS_REGION.PY STARTED")
    logger.debug("=" * 80)

    # Log all environment variables that matter
    env_vars_to_check = [
        'dynamic_world_data',
        'dynamic_world_downloads',
        'vector_lake_file',
        'output_dir',
        'project',
        'region_name'
    ]

    for var in env_vars_to_check:
        value = os.environ.get(var, 'NOT SET')
        logger.debug(f"{var} = {value}")

    REGION = os.environ.get("region_name", "TEST")

    # Also check if the download directory actually exists
    dynamic_world_download_dir = os.environ.get('dynamic_world_downloads')
    if dynamic_world_download_dir:
        download_path = Path(dynamic_world_download_dir) / REGION / 'download_2026-06'
        logger.debug(f"Checking for downloads at: {download_path}")
        logger.debug(f"Path exists: {download_path.exists()}")
        if download_path.exists():
            files = list(download_path.glob('*.nc'))
            logger.debug(f"Found {len(files)} NetCDF files")
            if files:
                logger.debug(f"First few files: {files[:3]}")
    logger.debug(f"Running processing cron job")
    logger.debug(f"=== STARTING PROCESSING ===")
    logger.debug(f"Current date: {datetime.now()}")
    logger.debug(f"Environment variables: {dict(os.environ)}")

    # Log all important env vars
    logger.debug(f"REGION: {os.environ.get('region_name', 'NOT SET')}")
    logger.debug(f"dynamic_world_data: {os.environ.get('dynamic_world_data', 'NOT SET')}")
    logger.debug(f"dynamic_world_downloads: {os.environ.get('dynamic_world_downloads', 'NOT SET')}")
    logger.debug(f"vector_lake_file: {os.environ.get('vector_lake_file', 'NOT SET')}")
    SHOULD_RUN = False
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
        # TODO check if the downloads for the region are complete
        downloads_complete = verify_downloads_complete(region=REGION, analysis_dates=date_to_run)
        complete = downloads_complete['complete']
        complete_dates = downloads_complete['complete_dates']
        incomplete_dates = downloads_complete['incomplete_dates']
        summary = downloads_complete['summary']
        logger.debug(f"Total expected downloads {summary['total_expected_downloads']}")
        logger.debug(f"Total successful downloads {summary['total_successful_downloads']}")
        total_skipped_and_successful_downloads = summary['total_skipped_downloads'] + summary[
            'total_successful_downloads']
        percent_downloaded = float(total_skipped_and_successful_downloads) / float(summary['total_expected_downloads'])
        logger.debug(f"Percent downloaded: {percent_downloaded}")
        if percent_downloaded > 0.99:
            complete = True
        if complete:
            process_result = process_near_real_time_region_dates_zarr(region=REGION,
                                                                      current_analysis_dates=timestamp_to_run)
            logger.debug(f"Result is {process_result}")
    else:
        logger.debug(f"Too early in the month to run")

if __name__ == '__main__':
    main()