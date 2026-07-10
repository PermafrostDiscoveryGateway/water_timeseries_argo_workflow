from near_real_time_grid_v2 import verify_downloads_complete, verify_process_complete, merge_near_real_time_region , \
    process_near_real_time_region_dates_zarr, download_near_real_time_region_dates, generate_expected_dates, \
                                     merge_near_real_time_region_v3_simple, \
                 compare_netcdf_files, verify_merged_netcdf, verify_merged_data, merge_near_real_time_region_v3_smart, \
            enable_memory_tracking, log_memory_usage, has_region_been_merged_for_dates, merge_near_real_time_region_v4_smart
import sys
import shutil
import gc
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
    stat_info = os.stat(Path(filepath))
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


def is_file_ready(filepath, wait_seconds=0.5, checks=10):
    sizes = []
    for _ in range(checks):
        size = os.path.getsize(filepath)
        sizes.append(size)
        time.sleep(wait_seconds)

    # If size hasn't changed, assume writing is done
    return len(set(sizes)) == 1



def main():
    logger.debug(f"Beginning historical run")
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    enable_memory_tracking()
    log_memory_usage("Program start")

    try:
        import dask
        dask.config.set(scheduler='threads')
        dask.config.set({'array.chunk-size': '128MiB'})
    except:
        pass

    REGION = os.environ.get("region_name", "TEST")

    SHOULD_RUN = False

    summer_months = [6, 7, 8, 9]
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    all_dynamic_world_files = glob.glob(os.path.join(dynamic_world_data_dir, "*.nc"))
    most_recent_dynamic_world_file = None
    for file in all_dynamic_world_files:
        time_created = get_creation_time(file)
        readable_time = datetime.fromtimestamp(time_created)
        logger.debug(f"Netcdf file {file} has creation date of {readable_time}")
        most_recent_dynamic_world_file = max(all_dynamic_world_files, key=lambda f: Path(f).stat().st_mtime)
    logger.debug(f"Most recent dynamic world file {most_recent_dynamic_world_file}")
    # missing_dates_from_netcdf = utils.download_new_dynamic_world_data.check_missing_data_in_netcdf(original_most_recent_dynamic_world_file)



    TODAY =  datetime.now()
    TODAY_MONTH = TODAY.month
    if TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            logger.debug(f"TODAY_DAY: {TODAY_DAY} should we run and check: {SHOULD_RUN}")

    HISTORICAL_DATA_FILE = os.path.join(dynamic_world_data_dir, "lakes_dw_V2d_2016-2025.nc")

    if SHOULD_RUN:
        date_to_run = [datetime(TODAY.year, TODAY_MONTH -1, 1).strftime("%Y-%m")]
        dates_to_run_string = date_to_run[0].replace('-', '_')
        NAME_OF_FINAL_MERGE_FILE = f"{dynamic_world_data_dir}_lakes_dw_Vdc{dates_to_run_string}.nc"
        logger.debug(f"New netcdf file will be {NAME_OF_FINAL_MERGE_FILE}")
        logger.debug(f"Checking if we should merge")
        logger.debug(f"Merge if {date_to_run} are downloaded for all regions")
        REGIONS = utils.region_boundaries.get_region_boundaries()
        REGION_NAMES = list(REGIONS.keys())

        regions_downloaded = 0
        regions_downloaded_names = []

        for region in REGION_NAMES:
            downloads_complete = verify_downloads_complete(region=region, analysis_dates=date_to_run)
            logger.debug(downloads_complete)
            complete = downloads_complete['complete']
            complete_dates = downloads_complete['complete_dates']
            incomplete_dates = downloads_complete['incomplete_dates']
            summary = downloads_complete['summary']
            logger.debug(f"Total expected downloads {summary['total_expected_downloads']}")
            logger.debug(f"Total successful downloads {summary['total_successful_downloads']}")
            total_skipped_and_successful_downloads = summary['total_skipped_downloads'] + summary[
                'total_successful_downloads']
            total_expected_downloads = summary['total_expected_downloads']
            percent_downloaded = float(total_skipped_and_successful_downloads) / float(total_expected_downloads)
            logger.debug(f"Percent downloaded for {region}: {percent_downloaded}")
            logger.debug(f"Percent downloaded: {percent_downloaded}")
            if downloads_complete['complete'] or percent_downloaded > 0.99:
                regions_downloaded += 1
                regions_downloaded_names.append(region)
        logger.debug(f"How many regions are finished downloading?")
        logger.debug(f"{regions_downloaded} regions downloaded")
        logger.debug(f"These regions are fully downloaded")

        for region in regions_downloaded_names:
            logger.debug(region)

        successfully_merged_region_count = 0

        # TODO check this is it right copy method?
        logger.debug(f"Copying older data to {NAME_OF_FINAL_MERGE_FILE}")
        shutil.copy(HISTORICAL_DATA_FILE, NAME_OF_FINAL_MERGE_FILE)

        if regions_downloaded == len(REGION_NAMES):
            for region in REGION_NAMES:
                logger.debug(f"Checking if we already merged region {region} for {date_to_run}")

            if regions_downloaded == len(REGION_NAMES):
                successfully_merged_region_count = 0

                for region in REGION_NAMES:
                    logger.debug(f"Merging region {region} into {NAME_OF_FINAL_MERGE_FILE}")

                    # Use V4 smart merge that merges INTO the target file
                    merge_result = merge_near_real_time_region_v4_smart(
                        region=region,
                        dates_to_merge=date_to_run,
                        target_file_path=NAME_OF_FINAL_MERGE_FILE,  # ← Merge INTO this file
                        skip_if_already_merged=True,
                        verify_downloads_first=False  # Already verified
                    )

                    if merge_result.get('success', False):
                        if merge_result.get('skipped', False):
                            logger.info(f"✅ Region {region} already had data (skipped)")
                        else:
                            successfully_merged_region_count += 1
                            logger.info(f"✅ Merge successful for region {region}")
                            logger.info(f"  Target file now has {merge_result.get('id_count', 0):,} IDs")
                            logger.info(f"  File size: {merge_result.get('file_size_gb', 0):.2f} GB")

                        gc.collect()
                        log_memory_usage(f"After merging region {region}")
                    else:
                        logger.error(f"❌ Merge failed for region {region}: {merge_result.get('error')}")

                logger.debug(f"Verifying merge finished for all regions properly")
            else:
                logger.debug(f"Not all regions downloaded, do not merge")

            logger.debug(f"Successfully merged {successfully_merged_region_count}")
            if successfully_merged_region_count == len(REGION_NAMES):
                logger.debug("All regions merged successfully!")
                # The file is already at NAME_OF_FINAL_MERGE_FILE
                logger.info(f"Final merged file: {NAME_OF_FINAL_MERGE_FILE}")
            else:
                logger.warning(f"Only {successfully_merged_region_count}/{len(REGION_NAMES)} regions merged")

            logger.debug(f"Merging now finished")







if __name__ == "__main__":
    main()