from h5py.h5z import FLAG_SKIP_EDC
from networkx import all_simple_edge_paths

from near_real_time.near_real_time_grid_v2 import download_near_real_time_region, download_new_dynamic_world_data, \
    verify_downloads_complete, merge_near_real_time_region, compare_netcdf_files, process_near_real_time_region_dates, \
    generate_expected_dates, download_near_real_time_region_dates, process_near_real_time_region_dates_zarr, verify_process_complete
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

    SHOULD_RUN = False
    RUN_DOWNLOAD = False
    RUN_PROCESS = False

    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    original_most_recent_dynamic_world_file = max(all_dynamic_world_files, key=os.path.getctime)
    logger.debug(f"Original most recent dynamic world file: {original_most_recent_dynamic_world_file}")

    now = datetime.now()
    start_label = now.strftime("%Y%m%d%H%M%S")


    analysis_dates = []
    original_missing_dates = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_most_recent_dynamic_world_file)
    last_date = max(original_missing_dates)
    logger.debug(f"Last date: {last_date} of type {type(last_date)}")
    # TODO get rid of the last date, if today is within the same month
    last_date_year = last_date.year
    last_date_month = last_date.month
    last_date_day = last_date.day

    current_year = now.year
    current_month = now.month
    current_day = now.day

    if last_date_year == current_year and last_date_month == current_month and current_day > 3:
        logger.debug(f"We should run the previous month now")
        analysis_dates.remove(last_date)
        SHOULD_RUN = True
    else:
        logger.debug(f"We should not run")
    logger.debug(f"Analysis dates: {analysis_dates}")

    # if year and month are not less than today
    # and if today is not after the 4th of the month

    for date_idx, date in enumerate(original_missing_dates):
        date_type = type(date)
        print(date_type)
        ANALYSIS_DATE = date.strftime("%Y-%m")
        analysis_dates.append(ANALYSIS_DATE)

    expected_dates = generate_expected_dates(start_year=2025)
    expected_dates= expected_dates[:-2]
    # TODO check if the downloads are done
    verify_downloads =verify_downloads_complete(region="EURASIA3", run_start_label=start_label, analysis_dates=expected_dates)
    all_downloads_complete = verify_downloads['complete']
    # TODO check if processing is done
    verify_process = verify_process_complete(region='EURASIA3', analysis_dates=expected_dates)
    all_process_complete = verify_process['complete']
    logger.debug(f"All download complete: {all_downloads_complete} and all process complete: {all_process_complete}")
    if SHOULD_RUN:
        if not all_downloads_complete:
            RUN_DOWNLOAD = True
        if not all_process_complete:
            SHOULD_RUN = True
    logger.debug(f"Should run {SHOULD_RUN} and should download {RUN_PROCESS} and process {RUN_DOWNLOAD}")
    # download_near_real_time_region_dates(region="TEST", run_start_label=start_label, dates_to_download=expected_dates)
    logger.debug(f"Expected dates: {expected_dates}")
    verify_results =verify_downloads_complete(region="TEST", run_start_label=start_label, analysis_dates=expected_dates)
    logger.debug(f"Verify results")
    logger.debug(verify_results)

    # verify_process = verify_process_complete(region='TEST', analysis_dates=expected_dates)

    result = merge_near_real_time_region(region="TEST", run_start_label=start_label, dates_to_merge=analysis_dates)

    if result is not None and 'result' in result:
        new_file_path = result['result']
        new_file_path_string = str(new_file_path)
        logger.debug(f"Comparing the netcdf files {new_file_path_string} to original {original_most_recent_dynamic_world_file}")
        compare_netcdf_files(original_most_recent_dynamic_world_file, new_file_path_string)
        logger.debug(f"Comparison finished")
    else:
        logger.debug(f"No result no merge required")

    process_near_real_time_region_dates_zarr(region="TEST", current_analysis_dates=expected_dates)
    logger.debug(f"Finished processing")

if __name__ == "__main__":
    main()