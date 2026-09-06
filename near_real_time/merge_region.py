"""
Merge one region's recent downloads into that region's merged NetCDF file.

merge_recent_downloads.py loops over every region in a single pod, so nothing
downstream (process_NRT.py) can start on any region until the slowest
region's merge in that pod finishes. This script does exactly one region's
merge - reusing the same process_region_fast() step merge_recent_downloads.py
already calls per region - so it can be triggered as soon as that region's
own download step completes, letting process_NRT.py start on that region
without waiting on any other region's download or merge.

Does not perform the "combine all regions into one file" step - that still
needs every region's merged file to exist, and stays in
merge_recent_downloads.py until/unless a separate combine step is split out.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.date_gate import is_test_run, most_recent_summer_month
from near_real_time.merge_recent_downloads import process_region_fast, _configure_dask_for_low_memory


def main():
    _configure_dask_for_low_memory()

    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    REGION = os.environ.get("region_name", "TEST")
    logger.info(f"=== REGION FROM ENV: '{REGION}' ===")

    dynamic_world_data_dir = os.environ['dynamic_world_data']

    # ========== DETERMINE DATE TO RUN (same window as merge_recent_downloads.py) ==========
    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    target_month = None

    if is_test_run():
        SHOULD_RUN = True
        target_month = most_recent_summer_month(TODAY)
        logger.debug(f"test_run=True - bypassing the summer-month/day-of-month gate, using {target_month.strftime('%Y-%m')}")
    elif TODAY_MONTH - 1 in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            # Last complete (previous) month, computed this way so it also
            # handles the January wraparound correctly.
            target_month = TODAY.replace(day=1) - timedelta(days=1)

    if not SHOULD_RUN:
        logger.debug("Too early in the month to run - exiting")
        return 0

    date_to_run = target_month.strftime("%Y-%m")
    logger.info(f"Merging {REGION} for {date_to_run}")

    result = process_region_fast(
        region=REGION,
        date_to_run=date_to_run,
        env_path=env_path,
        dynamic_world_data_dir=dynamic_world_data_dir
    )

    logger.info("\n" + "=" * 80)
    logger.info(f"MERGE SUMMARY for {REGION} / {date_to_run}")
    logger.info("=" * 80)

    if result.get('success', False):
        if result.get('partial', False):
            logger.info(f"⚠️ {REGION} merged partially (acceptable): {result.get('reason')}")
        else:
            logger.info(f"✅ {REGION} merged successfully: {result.get('reason')}")
        logger.info(f"  Merged file: {result.get('merged_file')}")
        return 0

    logger.error(f"❌ {REGION} merge failed: {result.get('reason', 'Unknown error')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
