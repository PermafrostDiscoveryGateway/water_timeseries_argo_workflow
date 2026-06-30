from near_real_time.near_real_time_grid_v2 import download_near_real_time_region, download_new_dynamic_world_data, \
    verify_downloads_complete, merge_near_real_time_region, compare_netcdf_files, process_near_real_time_region_dates, \
    generate_expected_dates, download_near_real_time_region_dates
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
    original_most_recent_dynamic_world_file = min(all_dynamic_world_files, key=os.path.getctime)

    now = datetime.now()
    start_label = now.strftime("%Y%m%d%H%M%S")

    analysis_dates = []
    original_missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_most_recent_dynamic_world_file)
    for date_idx, date in enumerate(original_missing_dates):
        date_type = type(date)
        print(date_type)
        ANALYSIS_DATE = date.strftime("%Y-%m")
        analysis_dates.append(ANALYSIS_DATE)

    expected_dates = generate_expected_dates(start_year=2025)

    download_near_real_time_region_dates(region="TEST", run_start_label=start_label, dates_to_download=expected_dates)
    verify_downloads_complete(region="TEST", run_start_label=start_label, analysis_dates=analysis_dates)
    result = merge_near_real_time_region(region="TEST", run_start_label=start_label, dates_to_merge=analysis_dates)
    logger.debug(result)
    new_file_path = result['result']
    new_file_path_string = str(new_file_path)
    logger.debug(f"Done for test")

    # new_netcdf_file_path = '/Users/helium/Desktop/dynamic_world/data/historical_data_20260629_104025.nc'
    # original_netcdf_file_path = '/Users/helium/Desktop/dynamic_world/data/lakes_dw_V2d.nc'
    logger.debug(f"Comparing the netcdf files {new_file_path_string} to original {original_most_recent_dynamic_world_file}")
    compare_netcdf_files(original_most_recent_dynamic_world_file, new_file_path_string)
    logger.debug(f"Comparison finished")

    process_near_real_time_region_dates(region="TEST", current_analysis_dates=original_missing_dates)
    logger.debug(f"Finished processing")

if __name__ == "__main__":
    main()