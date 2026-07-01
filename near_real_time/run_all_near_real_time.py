from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region , \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates
import sys
from loguru import logger
from datetime import date, datetime
from dotenv import load_dotenv
import os
import glob
import pandas as pd
import utils.region_boundaries
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    logger.debug(f"Beginning historical run")
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    REGIONS = utils.region_boundaries.get_region_boundaries()
    REGION_NAMES = list(REGIONS.keys())

    SHOULD_RUN = False

    summer_months = [6, 7, 8, 9]

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
    for REGION in REGION_NAMES:
        dynamic_world_data_dir = os.environ['dynamic_world_data']
        all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
        most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
        downloads_complete = verify_downloads_complete(region=REGION, analysis_dates=date_to_run)
        if downloads_complete['complete']:
            logger.debug(f"Downloads already complete for region {REGION}")
        else:
            logger.debug(f"downloads_complete: {downloads_complete}")
            logger.debug(f"downloading data for   REGION: {REGION}")
            download_near_real_time_region_dates(region=REGION, dates_to_download=timestamp_to_run)
            logger.debug(f"Finished downloading data for   REGION: {REGION}")
            result = merge_near_real_time_region(region="TEST", dates_to_merge=date_to_run)
            logger.debug(f"Merging finished for region : {REGION}")
    for REGION in REGION_NAMES:
        logger.debug(f"Processing region {REGION}")
        # TODO check if we should process

    logger.debug("Finished NRT for all regions")


if __name__ == "__main__":
    main()