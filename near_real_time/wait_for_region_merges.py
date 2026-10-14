"""
Cheap pre-check for the create-new-historical-file step.

Polls for every region's merge output (dw_{region}_{date}.nc) the same way
create_new_historical_file.py's wait_for_regions_to_complete() does, but on
its own, low-resource pod - so the heavy create-new-historical-file container
(large memory request) only gets scheduled once merges are actually ready,
instead of sitting reserved for however long the region pipelines take.

Exit code 0: merges are ready (or nothing to do this month - see
get_target_month_to_run()). Exit code 1: timed out waiting.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from near_real_time.create_new_historical_file import (
    get_all_regions, get_target_month_to_run, wait_for_regions_to_complete,
)


def main():
    env_path = sys.argv[1] if len(sys.argv) > 1 else None
    if env_path:
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    target_month = get_target_month_to_run()
    if target_month is None:
        logger.info("Too early in the month to run - nothing to wait for.")
        sys.exit(0)

    date_to_run = target_month.strftime("%Y-%m")
    all_regions = get_all_regions()
    dynamic_world_data_dir = os.environ['dynamic_world_data']
    max_wait_minutes = int(os.environ.get('merge_wait_minutes', 2880))

    completed = wait_for_regions_to_complete(
        dynamic_world_data_dir=dynamic_world_data_dir,
        date_to_run=date_to_run,
        regions=all_regions,
        max_wait_minutes=max_wait_minutes,
        check_interval_seconds=30,
    )
    sys.exit(0 if completed else 1)


if __name__ == "__main__":
    main()
