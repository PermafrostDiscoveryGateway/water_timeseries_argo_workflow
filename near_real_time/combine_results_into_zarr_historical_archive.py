from dotenv import load_dotenv
import os
import glob
import time
import pandas as pd
from datetime import datetime
import sys
from loguru import logger
from pathlib import Path
import xarray as xr
from utils.region_boundaries import get_region_boundaries
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# A breakpoint zarr store that was written to within this window is assumed to
# still be mid-write, so it's excluded from the merge rather than folded in as a partial.
STALE_THRESHOLD_SECONDS = 60 * 60


def get_most_recent_zarr(zarr_dir):
    """Return the most recently created .zarr store in zarr_dir, or None if there isn't one."""
    zarr_paths = glob.glob(os.path.join(zarr_dir, "*.zarr"))
    if not zarr_paths:
        return None
    return max(zarr_paths, key=os.path.getctime)


def zarr_store_age_seconds(zarr_path):
    """Seconds since the most recently modified file inside a zarr store."""
    files = glob.glob(os.path.join(zarr_path, "**", "*"), recursive=True)
    last_modified = max((os.path.getmtime(f) for f in files), default=os.path.getmtime(zarr_path))
    return time.time() - last_modified


def main():
    logger.debug(f"Combining the exiting zarr dataset with all new")
    env_path = None
    if len(sys.argv) > 1:
        env_path = sys.argv[1]
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loading environment from: {env_path}")
    else:
        load_dotenv()
        logger.info("Loading environment from default .env file")

    combined_zarr_datasets = os.path.join(project_root, "data", "full_datasets")

    most_recent_full_dataset = get_most_recent_zarr(combined_zarr_datasets)
    if most_recent_full_dataset:
        logger.info(f"Most recent existing full dataset: {most_recent_full_dataset}")
    else:
        logger.warning(f"No existing full dataset found in {combined_zarr_datasets}")

    SHOULD_RUN = False
    summer_months = [6, 7, 8, 9]
    TODAY = datetime.now()
    TODAY_MONTH = TODAY.month
    TODAY_YEAR = TODAY.year

    if (TODAY_MONTH - 1) in summer_months:
        TODAY_DAY = TODAY.day
        if TODAY_DAY > 3:
            SHOULD_RUN = True
            target_month = TODAY_MONTH - 1
            target_date = f"{TODAY_YEAR}-{target_month:02d}"
            logger.debug(f"TODAY_DAY: {TODAY_DAY} - Should run: {SHOULD_RUN}")
            logger.debug(f"Target date: {target_date}")

    if not SHOULD_RUN:
        logger.info("Skipping processing - conditions not met")
        return

    output_dir = os.environ['output_dir']

    regions = list(get_region_boundaries().keys())

    latest_results_paths = []

    for region in regions:
        path_to_latest_data = os.path.join(output_dir, region, 'breakpoint_zarr', f"breakpoints_{target_date}_{region}.zarr" )
        latest_results_paths.append(path_to_latest_data)

    results_paths_to_merge = []
    for path in latest_results_paths:
        if not os.path.exists(path):
            logger.warning(f"Breakpoint zarr does not exist, skipping: {path}")
            continue
        age_seconds = zarr_store_age_seconds(path)
        if age_seconds < STALE_THRESHOLD_SECONDS:
            logger.warning(
                f"Breakpoint zarr was modified {age_seconds:.0f}s ago (< {STALE_THRESHOLD_SECONDS}s) "
                f"and may still be writing, skipping: {path}"
            )
            continue
        results_paths_to_merge.append(path)

    if not results_paths_to_merge:
        logger.error("No complete breakpoint zarr datasets available to merge - aborting")
        return

    new_zarr_dataset_name = f'complete_lake_drainage_{target_date}.zarr'
    new_zarr_dataset_path = os.path.join(combined_zarr_datasets, new_zarr_dataset_name)

    logger.info(f"Opening {len(results_paths_to_merge)} region breakpoint zarr datasets...")
    ds_new_merged = xr.open_mfdataset(
        results_paths_to_merge, engine="zarr", combine="nested", concat_dim="id_geohash"
    )

    if most_recent_full_dataset:
        logger.info(f"Opening existing full dataset: {most_recent_full_dataset}")
        ds_full = xr.open_zarr(most_recent_full_dataset)
        ds_full['id_geohash'] = ds_full['id_geohash'].astype(str)

        logger.info("Aligning new data coordinates to the master lake list...")
        ds_new_aligned = ds_new_merged.reindex(id_geohash=ds_full.id_geohash)

        logger.info("Concatenating existing history with the new month...")
        ds_combined = xr.concat([ds_full, ds_new_aligned], dim="date")
    else:
        logger.info("No existing full dataset found - new archive will start from this month")
        ds_combined = ds_new_merged

    logger.info(f"Saving combined result to {new_zarr_dataset_path}")
    ds_combined.to_zarr(new_zarr_dataset_path, mode='w', align_chunks=True)

    logger.success(f"Combined zarr dataset written to {new_zarr_dataset_path}")


if __name__ == "__main__":
    main()

