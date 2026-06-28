from near_real_time.near_real_time_grid_v2 import download_near_real_time_region, download_new_dynamic_world_data, \
    verify_downloads_complete
from near_real_time_grid import near_real_time_region
import sys
from loguru import logger
import glob
from datetime import date, datetime
from dotenv import load_dotenv
import os
import utils.region_boundaries
import utils.download_new_dynamic_world_data

def main():
    logger.debug(f"Beginning historical run")
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)

    now = datetime.now()
    start_label = now.strftime("%Y%m%d%H%M%S")

    analysis_dates = []
    original_missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_most_recent_dynamic_world_file)
    for date_idx, date in enumerate(original_missing_dates):
        ANALYSIS_DATE = date.strftime("%Y-%m")
        analysis_dates.append(ANALYSIS_DATE)

    download_near_real_time_region(region="TEST", run_start_label=start_label)
    verify_downloads_complete(region="TEST", run_start_label=start_label, analysis_dates=analysis_dates)
    logger.debug(f"Done for test")


    # near_real_time_region(region='TEST')


if __name__ == "__main__":
    main()